using Sdcb.GcdrNet.Engine;
using Sdcb.GcdrNet.Onnx;

namespace Sdcb.GcdrNet.Backends.Vulkan;

/// <summary>
/// Fused GPU worker: BGR u8 upload → bgr2nhwc shader → GC-Net → div6 (division
/// correction, interleaved 6ch NHWC) → DR-Net → nhwc2bgr → u8 readback. Both
/// plans keep physical (NHWC) I/O, so neither net contains a Transpose. One
/// command buffer, one submit per frame.
///
/// Size-agnostic: per-size state is only the cached graphs (plan + dispatch
/// list over the model's shared const store). The worker owns grow-only
/// activation arenas and BGR staging buffers and re-records its command buffer
/// when the source size changes, so a new size costs a re-record instead of a
/// pipeline build (arena allocation, weight upload, first-touch). Not
/// thread-safe; callers pool workers.
/// </summary>
internal sealed unsafe class VkGcdrPipeline : IDisposable
{
    private readonly VkDevice _dev;
    private readonly VkBackend _be;
    private readonly OnnxModel _gcModel, _drModel;
    private readonly IntPtr _cmdPool, _cmd, _fence;
    private IntPtr _descPool;              // !PushDescriptors: rebuilt on each record
    private VkBuffer? _gcArena, _drArena, _bgrIn, _bgrOut;
    private byte* _bgrInMap, _bgrOutMap;
    private VkGraphModel? _gcG, _drG;      // graphs of the bound padded size (refs held)
    private int _h, _w;                    // padded dims
    private int _ih, _iw;                  // source dims
    private int _pt, _pl;
    private bool _recorded;
    private readonly bool _f16;
    private readonly VkPipeline _pPre, _pDiv6, _pPost;

    internal VkGcdrPipeline(VkBackend be, OnnxModel gc, OnnxModel dr)
    {
        _be = be;
        _dev = be.Dev;
        _gcModel = gc;
        _drModel = dr;
        _f16 = be.F16;
        string sfx = _f16 ? "_h" : "";
        _pPre = _dev.GetPipeline("bgr2nhwc" + sfx, 2, 7 * 4);
        _pDiv6 = _dev.GetPipeline("div6" + sfx, 3, 4);
        _pPost = _dev.GetPipeline("nhwc2bgr" + sfx, 2, 7 * 4);
        _cmdPool = _dev.NewCommandPool();
        try
        {
            _cmd = _dev.AllocateCommandBuffer(_cmdPool);
            _fence = _dev.NewFence();
        }
        catch
        {
            _dev.DestroyCommandPool(_cmdPool);
            throw;
        }
    }

    private int Es => _f16 ? 2 : 4;
    private ExecPlan GcPlan => _gcG!.Plan;
    private ExecPlan DrPlan => _drG!.Plan;
    private ulong InOff(ExecPlan p) => (ulong)p.BufOffset[p.Values[p.InputValue].Buf] * (ulong)Es;
    private ulong OutOff(ExecPlan p) => (ulong)p.BufOffset[p.Values[p.OutputValue].Buf] * (ulong)Es;
    private ulong OutBytes(ExecPlan p) => (ulong)p.Values[p.OutputValue].ElementCount * (ulong)Es;
    private ulong InBytes(ExecPlan p) => (ulong)p.Values[p.InputValue].ElementCount * (ulong)Es;

    /// <summary>Bind the worker to a source size: fetch (cached) graphs for its padded
    /// size, grow buffers as needed, re-record when anything changed.</summary>
    internal void Bind(int ih, int iw)
    {
        if (_recorded && ih == _ih && iw == _iw) return;
        _recorded = false;   // stays false if anything below throws (buffers may be gone)
        int pt = (-ih) & 31, pl = (-iw) & 31;
        int h = ih + pt, w = iw + pl;
        if (_gcG is null || h != _h || w != _w)
        {
            VkGraphModel gg = _be.AcquireGraph(_gcModel, [1, 3, h, w], physIo: true);
            VkGraphModel gd;
            try { gd = _be.AcquireGraph(_drModel, [1, 6, h, w], physIo: true); }
            catch { gg.Release(); throw; }
            if (gg.F16 != _f16 || gd.F16 != _f16)
            {
                gg.Release(); gd.Release();
                throw new InvalidOperationException("gc/dr storage precision mismatch");
            }
            _gcG?.Release(); _drG?.Release();
            _gcG = gg; _drG = gd;
            _h = h; _w = w;
        }
        _ih = ih; _iw = iw; _pt = pt; _pl = pl;

        ulong es = (ulong)Es;
        ulong needGc = (ulong)GcPlan.ArenaFloats * es + 64, needDr = (ulong)DrPlan.ArenaFloats * es + 64;
        // arenas only grow while within ArenaBudget; an over-budget pair (one huge image)
        // is dropped to exact size as soon as a size that fits the budget comes along
        bool shrink = _gcArena is not null && _drArena is not null
            && _gcArena.Size + _drArena.Size > ArenaBudget && needGc + needDr <= ArenaBudget;
        bool headroom = _gcArena is not null && (needGc + needDr) / 4 * 5 <= ArenaBudget;
        Arena(ref _gcArena, needGc, shrink, headroom);
        Arena(ref _drArena, needDr, shrink, headroom);
        ulong bgrBytes = (ulong)ih * (ulong)iw * 3 + 4;
        Staging(ref _bgrIn, ref _bgrInMap, bgrBytes, preferHost: false);
        Staging(ref _bgrOut, ref _bgrOutMap, bgrBytes, preferHost: true);
        Record((uint)(_iw * 3));   // _bgrIn is always tightly packed (we de-stride on upload)
        _recorded = true;
    }

    /// <summary>Per-worker device-arena budget (gc + dr). Within it arenas are grow-only
    /// (25% headroom on regrowth) so mixed sizes don't reallocate.</summary>
    internal const ulong ArenaBudget = 3UL << 30;

    /// <summary>Device bytes held by this worker (arenas + BGR staging).</summary>
    internal long DeviceBytes => (long)((_gcArena?.Size ?? 0) + (_drArena?.Size ?? 0)
        + (_bgrIn?.Size ?? 0) + (_bgrOut?.Size ?? 0));

    private void Arena(ref VkBuffer? buf, ulong need, bool shrink, bool headroom)
    {
        if (buf is not null && buf.Size >= need && !shrink) return;
        buf?.Free();
        buf = null;
        buf = _dev.NewStorageBuffer(headroom && !shrink ? need / 4 * 5 : need, hostVisible: false);
    }

    /// <summary>Grow-only mapped staging buffer; the map pointer is cleared with the buffer
    /// it points into and re-established whenever missing.</summary>
    private void Staging(ref VkBuffer? buf, ref byte* map, ulong need, bool preferHost)
    {
        if (buf is not null && buf.Size < need)
        {
            if (map != null) buf.Unmap();
            map = null;
            buf.Free();
            buf = null;
        }
        buf ??= _dev.NewStorageBuffer(need + need / 4, hostVisible: true, preferHost);
        if (map == null) map = (byte*)buf.Map();
    }

    private static byte[] Pc(params uint[] u)
    {
        byte[] b = new byte[u.Length * 4];
        Buffer.BlockCopy(u, 0, b, 0, b.Length);
        return b;
    }

    private void Record(uint srcStride)
    {
        VkGraphModel gcG = _gcG!, drG = _drG!;
        ExecPlan gcPlan = gcG.Plan, drPlan = drG.Plan;
        if (!_dev.PushDescriptors)
        {
            if (_descPool != IntPtr.Zero) Vk.vkDestroyDescriptorPool(_dev.Device, _descPool, null);
            _descPool = IntPtr.Zero;
            _descPool = _dev.NewDescriptorPool((uint)(gcG.Dispatches.Length + drG.Dispatches.Length + 3));
        }

        uint h = (uint)_h, w = (uint)_w;
        Vk.Check(Vk.vkResetCommandBuffer(_cmd, 0), "vkResetCommandBuffer");
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(_cmd, &begin), "vkBeginCommandBuffer");

        if (Environment.GetEnvironmentVariable("GCDR_ZERO_ARENA") == "1")
        {
            Vk.vkCmdFillBuffer(_cmd, _gcArena!.Buffer, 0, _gcArena.Size, 0);
            Vk.vkCmdFillBuffer(_cmd, _drArena!.Buffer, 0, _drArena.Size, 0);
            Vk.VkMemoryBarrier fb = new()
            {
                SType = VkConst.StMemoryBarrier,
                SrcAccessMask = VkConst.AccessTransferWrite, DstAccessMask = VkConst.AccessShaderRead | VkConst.AccessShaderWrite,
            };
            Vk.vkCmdPipelineBarrier(_cmd, VkConst.PipelineStageTransfer, VkConst.PipelineStageComputeShader, 0,
                1, &fb, 0, null, 0, null);
        }

        // bgr2nhwc -> gc arena input
        Dispatch(_pPre,
            [(_bgrIn!, 0UL, _bgrIn!.Size), (_gcArena!, InOff(gcPlan), InBytes(gcPlan))],
            Pc(h, w, (uint)_ih, (uint)_iw, (uint)_pt, (uint)_pl, srcStride),
            (w + 255) / 256, h, 1);
        _dev.CmdComputeBarrier(_cmd);
        RecordGraph(gcG, _gcArena!);

        // div6: gc input + gc output -> dr arena input
        uint hw = h * w;
        ulong es = (ulong)Es;
        _dev.CmdComputeBarrier(_cmd);
        Dispatch(_pDiv6,
            [(_gcArena!, InOff(gcPlan), 3UL * hw * es),
             (_gcArena!, OutOff(gcPlan), 3UL * hw * es),
             (_drArena!, InOff(drPlan), 6UL * hw * es)],
            Pc(hw), (hw + 255) / 256, 1, 1);
        _dev.CmdComputeBarrier(_cmd);
        RecordGraph(drG, _drArena!);

        // nhwc2bgr -> bgrOut
        _dev.CmdComputeBarrier(_cmd);
        uint nwords = ((uint)_ih * (uint)_iw * 3u + 3u) / 4u;
        Dispatch(_pPost,
            [(_drArena!, OutOff(drPlan), OutBytes(drPlan)), (_bgrOut!, 0UL, _bgrOut!.Size)],
            Pc(h, w, (uint)_ih, (uint)_iw, (uint)_pt, (uint)_pl, nwords),   // PC layout must match nhwc2bgr.comp
            (nwords + 255) / 256, 1, 1);
        Vk.Check(Vk.vkEndCommandBuffer(_cmd), "vkEndCommandBuffer");
    }

    /// <summary>All graph dispatches with barriers between them (none before the first).</summary>
    private void RecordGraph(VkGraphModel g, VkBuffer arena)
    {
        var binds = new (VkBuffer buf, ulong off, ulong size)[8];
        for (int i = 0; i < g.Dispatches.Length; i++)
        {
            VkDispatch d = g.Dispatches[i];
            if (i > 0) _dev.CmdComputeBarrier(_cmd);
            if (binds.Length < d.Binds.Length) binds = new (VkBuffer, ulong, ulong)[d.Binds.Length];
            for (int b = 0; b < d.Binds.Length; b++)
            {
                VkBind vb = d.Binds[b];
                VkBuffer buf = vb.Slot switch
                {
                    VkSlot.Arena => arena,
                    VkSlot.Consts => g.Consts,
                    VkSlot.Dummy => g.Dummy,
                    VkSlot.Direct => vb.Buf!,
                    _ => throw new InvalidOperationException($"pipeline graph dispatch binds {vb.Slot}"),
                };
                binds[b] = (buf, vb.Offset, vb.Size);
            }
            Dispatch(d.Pipe, binds.AsSpan(0, d.Binds.Length), d.Pc, d.Gx, d.Gy, d.Gz);
        }
    }

    private void Dispatch(VkPipeline pipe, ReadOnlySpan<(VkBuffer buf, ulong off, ulong size)> binds,
        byte[] pc, uint gx, uint gy, uint gz)
    {
        Vk.vkCmdBindPipeline(_cmd, VkConst.BindPointCompute, pipe.Pipeline);
        if (_dev.PushDescriptors)
        {
            var writes = stackalloc Vk.VkWriteDescriptorSet[8];
            var infos = stackalloc Vk.VkDescriptorBufferInfo[8];
            for (int b = 0; b < binds.Length; b++)
            {
                infos[b] = new Vk.VkDescriptorBufferInfo { Buffer = binds[b].buf.Buffer, Offset = binds[b].off, Range = binds[b].size };
                writes[b] = new Vk.VkWriteDescriptorSet
                {
                    SType = VkConst.StWriteDescriptorSet, DstBinding = (uint)b,
                    DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer,
                    PBufferInfo = infos + b,
                };
            }
            _dev.CmdPushDescriptors(_cmd, pipe.Layout, writes, (uint)binds.Length);
        }
        else
        {
            IntPtr set = _dev.AllocateDescriptorSet(_descPool, pipe.SetLayout);
            for (int b = 0; b < binds.Length; b++)
                _dev.BindBuffer(set, (uint)b, binds[b].buf, binds[b].off, binds[b].size);
            Vk.vkCmdBindDescriptorSets(_cmd, VkConst.BindPointCompute, pipe.Layout, 0, 1, &set, 0, null);
        }
        if (pc.Length > 0)
            fixed (byte* p = pc)
                Vk.vkCmdPushConstants(_cmd, pipe.Layout, VkConst.StageComputeShader, 0, (uint)pc.Length, p);
        Vk.vkCmdDispatch(_cmd, gx, gy, gz);
    }

    /// <summary>Enhance one image of any size (binds/re-records first when the size changed).</summary>
    public void Enhance(ReadOnlySpan<byte> bgrSrc, int ih, int iw, int srcStride, Span<byte> bgrDst, int dstStride)
    {
        Bind(ih, iw);
        Enhance(bgrSrc, srcStride, bgrDst, dstStride);
    }

    /// <summary>Enhance one image of the currently bound size.</summary>
    public void Enhance(ReadOnlySpan<byte> bgrSrc, int srcStride, Span<byte> bgrDst, int dstStride)
    {
        if (!_recorded) throw new InvalidOperationException("worker not bound to a size");
        ulong inBytes = (ulong)_ih * (ulong)_iw * 3;
        if (srcStride == _iw * 3)
            bgrSrc.Slice(0, (int)inBytes).CopyTo(new Span<byte>(_bgrInMap, (int)inBytes));
        else
            for (int y = 0; y < _ih; y++)
                bgrSrc.Slice(y * srcStride, _iw * 3).CopyTo(
                    new Span<byte>(_bgrInMap + y * _iw * 3, _iw * 3));
        _bgrIn!.Flush(0, inBytes);

        _dev.Submit(_cmd, _fence);
        _dev.WaitFence(_fence);
        _bgrOut!.Invalidate(0, inBytes);

        if (dstStride == _iw * 3)
            new ReadOnlySpan<byte>(_bgrOutMap, (int)inBytes).CopyTo(bgrDst);
        else
            for (int y = 0; y < _ih; y++)
                new ReadOnlySpan<byte>(_bgrOutMap + y * _iw * 3, _iw * 3)
                    .CopyTo(bgrDst.Slice(y * dstStride));
    }

    /// <summary>Debug view of one stage (gc or dr) of the bound size.</summary>
    internal sealed class DebugStage(VkGcdrPipeline p, bool gc)
    {
        private VkGraphModel G => (gc ? p._gcG : p._drG) ?? throw new InvalidOperationException("worker not bound");
        private VkBuffer Arena => (gc ? p._gcArena : p._drArena)!;
        public ExecPlan DebugPlan => G.Plan;
        /// <summary>Download the plan's arena prefix (ArenaFloats elements) into dst.</summary>
        public void DebugArena(Span<float> dst) => p.Download(Arena, 0, dst.Slice(0, G.Plan.ArenaFloats));
        public (VkBuffer consts, VkBuffer arena) DebugBuffers => (G.Consts, Arena);
    }

    internal DebugStage DebugGc => new(this, true);
    internal DebugStage DebugDr => new(this, false);

    private void Download(VkBuffer buf, ulong off, Span<float> dst)
    {
        ulong bytes = (ulong)dst.Length * (ulong)Es;
        byte[] tmp = new byte[bytes];
        fixed (byte* t = tmp)
        {
            _dev.Download(buf, t, bytes, off);
            VkInferenceSession.FromStorage(t, dst, _f16);
        }
    }

    /// <summary>Debug: download intermediate tensors (gc input, gc shadow out, dr 6ch input — padded, NHWC).</summary>
    internal void DebugIntermediates(Span<float> gcIn, Span<float> gcOut, Span<float> drIn)
    {
        int hw = _h * _w;
        Download(_gcArena!, InOff(GcPlan), gcIn.Slice(0, 3 * hw));
        Download(_gcArena!, OutOff(GcPlan), gcOut.Slice(0, 3 * hw));
        Download(_drArena!, InOff(DrPlan), drIn.Slice(0, 6 * hw));
    }

    public void Dispose()
    {
        // Enhance() is synchronous (already waits the fence); callers must not race Dispose
        _dev.DestroyCommandPool(_cmdPool);
        if (_descPool != IntPtr.Zero) { Vk.vkDestroyDescriptorPool(_dev.Device, _descPool, null); _descPool = IntPtr.Zero; }
        _dev.DestroyFence(_fence);
        if (_bgrIn is not null) { if (_bgrInMap != null) _bgrIn.Unmap(); _bgrIn.Free(); _bgrIn = null; _bgrInMap = null; }
        if (_bgrOut is not null) { if (_bgrOutMap != null) _bgrOut.Unmap(); _bgrOut.Free(); _bgrOut = null; _bgrOutMap = null; }
        _gcArena?.Free(); _gcArena = null;
        _drArena?.Free(); _drArena = null;
        _gcG?.Release(); _gcG = null;
        _drG?.Release(); _drG = null;
        _recorded = false;
    }
}
