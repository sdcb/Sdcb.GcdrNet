using Sdcb.GcdrNet.Engine;

namespace Sdcb.GcdrNet.Backends.Vulkan;

/// <summary>
/// GPU counterpart of <see cref="InferenceSession"/>: one activation arena in
/// device-local memory, persistent host-visible staging buffers, and a single
/// pre-recorded command buffer (staging copy-in → all dispatches → copy-out).
/// Not thread-safe; one per thread or serialize externally.
/// </summary>
public sealed unsafe class VkInferenceSession : IDisposable
{
    private readonly VkDevice _dev;
    private readonly VkGraphModel _graph;
    private readonly ExecPlan _plan;

    internal readonly VkBuffer Arena;     // device-local activation arena
    private readonly VkBuffer _stageIn;   // host-visible float input staging
    private readonly VkBuffer _stageOut;  // host-visible float output staging
    private float* _stageInMap, _stageOutMap;
    private readonly float[] _inHost, _outHost;
    private readonly IntPtr _cmd, _fence;
    private readonly IntPtr[] _descSets;  // per-dispatch sets when !PushDescriptors
    private readonly ulong _inOff, _outOff;
    private readonly int _inFloats, _outFloats;

    internal VkInferenceSession(VkDevice dev, ExecPlan plan, VkGraphModel graph)
    {
        _dev = dev;
        _plan = plan;
        _graph = graph;

        Arena = dev.NewStorageBuffer((ulong)plan.ArenaFloats * 4 + 64, hostVisible: false);
        _inFloats = checked((int)plan.Values[plan.InputValue].ElementCount);
        _outFloats = checked((int)plan.Values[plan.OutputValue].ElementCount);
        _inOff = (ulong)plan.BufOffset[plan.Values[plan.InputValue].Buf] * 4;
        _outOff = (ulong)plan.BufOffset[plan.Values[plan.OutputValue].Buf] * 4;
        _stageIn = dev.NewStorageBuffer((ulong)_inFloats * 4, hostVisible: true, preferHost: true);
        _stageOut = dev.NewStorageBuffer((ulong)_outFloats * 4, hostVisible: true, preferHost: true);
        _stageInMap = (float*)_stageIn.Map();
        _stageOutMap = (float*)_stageOut.Map();
        _inHost = new float[_inFloats];
        _outHost = new float[_outFloats];
        _fence = dev.NewFence();

        if (!dev.PushDescriptors)
        {
            _descSets = new IntPtr[_graph.Dispatches.Length];
            for (int i = 0; i < _graph.Dispatches.Length; i++)
            {
                var d = _graph.Dispatches[i];
                IntPtr set = dev.NewDescriptorSet(d.Pipe.SetLayout);
                for (uint b = 0; b < d.Binds.Length; b++)
                {
                    (VkBuffer buf, _, _) = Resolve(d.Binds[b]);
                    dev.BindBuffer(set, b, buf, d.Binds[b].Offset, d.Binds[b].Size);
                }
                _descSets[i] = set;
            }
        }
        else _descSets = [];

        _cmd = dev.NewCommandBuffer();
        Record();
    }

    internal VkBuffer ResolveBuf(VkBind b) => Resolve(b).buf;
    internal ulong ResolveOff(VkBind b) => b.Offset;

    internal (VkBuffer buf, ulong off, ulong size) Resolve(VkBind b) => b.Slot switch
    {
        VkSlot.Arena => (Arena, b.Offset, b.Size),
        VkSlot.Consts => (_graph.Consts, b.Offset, b.Size),
        VkSlot.StageIn => (_stageIn, b.Offset, b.Size),
        VkSlot.StageOut => (_stageOut, b.Offset, b.Size),
        VkSlot.Direct => (b.Buf!, b.Offset, b.Size),
        _ => (_graph.Dummy, b.Offset, b.Size),
    };

    /// <summary>Input tensor shape (e.g. [1,3,H,W]).</summary>
    public int[] InputShape => _plan.Values[_plan.InputValue].Shape;
    public int[] OutputShape => _plan.Values[_plan.OutputValue].Shape;

    /// <summary>The writable input window (host copy staged on <see cref="Run"/>).</summary>
    public Span<float> Input => _inHost;
    /// <summary>The output tensor after <see cref="Run"/>.</summary>
    public ReadOnlySpan<float> Output => _outHost;

    internal ulong InputOffset => _inOff;
    internal ulong OutputOffset => _outOff;
    internal ulong InputBytes => (ulong)_inFloats * 4;
    internal ulong OutputBytes => (ulong)_outFloats * 4;

    /// <summary>Record one dispatch (push-descriptor path only; caller emits barriers).</summary>
    internal static void RecordOne(VkDevice dev, IntPtr cmd, VkPipeline pipe,
        (VkBuffer buf, ulong off, ulong size)[] binds, byte[] pc, uint gx, uint gy)
    {
        Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, pipe.Pipeline);
        var writes = stackalloc Vk.VkWriteDescriptorSet[8];
        var infos = stackalloc Vk.VkDescriptorBufferInfo[8];
        for (uint b = 0; b < binds.Length; b++)
        {
            infos[b] = new Vk.VkDescriptorBufferInfo { Buffer = binds[b].buf.Buffer, Offset = binds[b].off, Range = binds[b].size };
            writes[b] = new Vk.VkWriteDescriptorSet
            {
                SType = VkConst.StWriteDescriptorSet, DstBinding = b,
                DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer,
                PBufferInfo = infos + b,
            };
        }
        dev.CmdPushDescriptors(cmd, pipe.Layout, writes, (uint)binds.Length);
        if (pc.Length > 0)
            fixed (byte* p = pc)
                Vk.vkCmdPushConstants(cmd, pipe.Layout, VkConst.StageComputeShader, 0, (uint)pc.Length, p);
        Vk.vkCmdDispatch(cmd, gx, gy, 1);
    }

    /// <summary>Record all dispatches with barriers into an open command buffer.</summary>
    internal void RecordCompute(IntPtr cmd)
    {
        var writes = stackalloc Vk.VkWriteDescriptorSet[8];
        var infos = stackalloc Vk.VkDescriptorBufferInfo[8];
        bool first = true;
        for (int i = 0; i < _graph.Dispatches.Length; i++)
        {
            var d = _graph.Dispatches[i];
            if (!first) _dev.CmdComputeBarrier(cmd);
            first = false;
            Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, d.Pipe.Pipeline);
            if (_dev.PushDescriptors)
            {
                for (uint b = 0; b < d.Binds.Length; b++)
                {
                    (VkBuffer buf, ulong off, ulong size) = Resolve(d.Binds[b]);
                    infos[b] = new Vk.VkDescriptorBufferInfo { Buffer = buf.Buffer, Offset = off, Range = size };
                    writes[b] = new Vk.VkWriteDescriptorSet
                    {
                        SType = VkConst.StWriteDescriptorSet, DstBinding = b,
                        DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer,
                        PBufferInfo = infos + b,
                    };
                }
                _dev.CmdPushDescriptors(cmd, d.Pipe.Layout, writes, (uint)d.Binds.Length);
            }
            else
            {
                IntPtr set = _descSets[i];
                Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, d.Pipe.Layout, 0, 1, &set, 0, null);
            }
            if (d.Pc.Length > 0)
                fixed (byte* pc = d.Pc)
                    Vk.vkCmdPushConstants(cmd, d.Pipe.Layout, VkConst.StageComputeShader, 0, (uint)d.Pc.Length, pc);
            Vk.vkCmdDispatch(cmd, d.Gx, d.Gy, d.Gz);
        }
    }

    private void Record()
    {
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(_cmd, &begin), "vkBeginCommandBuffer");
        // staging -> arena(input)
        Vk.VkBufferCopy cin = new() { Size = InputBytes, DstOffset = _inOff };
        Vk.vkCmdCopyBuffer(_cmd, _stageIn.Buffer, Arena.Buffer, 1, &cin);
        // barrier: transfer-write -> shader-read
        Vk.VkMemoryBarrier mb = new()
        {
            SType = VkConst.StMemoryBarrier,
            SrcAccessMask = VkConst.AccessTransferWrite, DstAccessMask = VkConst.AccessShaderRead,
        };
        Vk.vkCmdPipelineBarrier(_cmd, VkConst.PipelineStageTransfer, VkConst.PipelineStageComputeShader, 0,
            1, &mb, 0, null, 0, null);
        RecordCompute(_cmd);
        // barrier: shader-write -> transfer-read
        Vk.VkMemoryBarrier mb2 = new()
        {
            SType = VkConst.StMemoryBarrier,
            SrcAccessMask = VkConst.AccessShaderWrite, DstAccessMask = VkConst.AccessTransferRead,
        };
        Vk.vkCmdPipelineBarrier(_cmd, VkConst.PipelineStageComputeShader, VkConst.PipelineStageTransfer, 0,
            1, &mb2, 0, null, 0, null);
        Vk.VkBufferCopy cout = new() { Size = OutputBytes, SrcOffset = _outOff };
        Vk.vkCmdCopyBuffer(_cmd, Arena.Buffer, _stageOut.Buffer, 1, &cout);
        Vk.Check(Vk.vkEndCommandBuffer(_cmd), "vkEndCommandBuffer");
    }

    /// <summary>Debug/test: record a single dispatch list slice and run it.</summary>
    internal void DebugRunRange(int dStart, int dCount)
    {
        IntPtr cmd = _dev.NewCommandBuffer();
        IntPtr fence = _dev.NewFence();
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
        var writes = stackalloc Vk.VkWriteDescriptorSet[8];
        var infos = stackalloc Vk.VkDescriptorBufferInfo[8];
        for (int i = dStart; i < dStart + dCount; i++)
        {
            var d = _graph.Dispatches[i];
            if (i > dStart) _dev.CmdComputeBarrier(cmd);
            Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, d.Pipe.Pipeline);
            for (uint b = 0; b < d.Binds.Length; b++)
            {
                (VkBuffer buf, ulong off, ulong size) = Resolve(d.Binds[b]);
                infos[b] = new Vk.VkDescriptorBufferInfo { Buffer = buf.Buffer, Offset = off, Range = size };
                writes[b] = new Vk.VkWriteDescriptorSet
                {
                    SType = VkConst.StWriteDescriptorSet, DstBinding = b,
                    DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer,
                    PBufferInfo = infos + b,
                };
            }
            _dev.CmdPushDescriptors(cmd, d.Pipe.Layout, writes, (uint)d.Binds.Length);
            if (d.Pc.Length > 0)
                fixed (byte* pc = d.Pc)
                    Vk.vkCmdPushConstants(cmd, d.Pipe.Layout, VkConst.StageComputeShader, 0, (uint)d.Pc.Length, pc);
            Vk.vkCmdDispatch(cmd, d.Gx, d.Gy, d.Gz);
        }
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        _dev.Submit(cmd, fence);
        _dev.WaitFence(fence);
        _dev.DestroyFence(fence);
        _dev.FreeCommandBuffer(cmd);
    }

    /// <summary>
    /// Debug/test: time each dispatch by submitting it alone (barrier + dispatch) and
    /// wall-timing the fenced submission. Includes ~20-40us launch overhead per dispatch;
    /// accurate enough to rank ms-scale kernels. Returns name→best-of-2 ms.
    /// </summary>
    internal Dictionary<string, double> DebugProfile()
    {
        int n = _graph.Dispatches.Length;
        IntPtr cmd = _dev.NewCommandBuffer();
        IntPtr fence = _dev.NewFence();
        var writes = stackalloc Vk.VkWriteDescriptorSet[8];
        var infos = stackalloc Vk.VkDescriptorBufferInfo[8];
        var best = new double[n];
        Array.Fill(best, double.MaxValue);
        void RecordDispatch(VkDispatch d)
        {
            _dev.CmdComputeBarrier(cmd);
            Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, d.Pipe.Pipeline);
            for (uint b = 0; b < d.Binds.Length; b++)
            {
                (VkBuffer buf, ulong off, ulong size) = Resolve(d.Binds[b]);
                infos[b] = new Vk.VkDescriptorBufferInfo { Buffer = buf.Buffer, Offset = off, Range = size };
                writes[b] = new Vk.VkWriteDescriptorSet
                {
                    SType = VkConst.StWriteDescriptorSet, DstBinding = b,
                    DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer,
                    PBufferInfo = infos + b,
                };
            }
            _dev.CmdPushDescriptors(cmd, d.Pipe.Layout, writes, (uint)d.Binds.Length);
            if (d.Pc.Length > 0)
                fixed (byte* pc = d.Pc)
                    Vk.vkCmdPushConstants(cmd, d.Pipe.Layout, VkConst.StageComputeShader, 0, (uint)d.Pc.Length, pc);
            Vk.vkCmdDispatch(cmd, d.Gx, d.Gy, d.Gz);
        }
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        // warm pass to keep state identical to production (input already in arena)
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
        foreach (var d in _graph.Dispatches) RecordDispatch(d);
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        _dev.Submit(cmd, fence);
        _dev.WaitFence(fence);
        // timed passes: one dispatch per submit, min of 2
        var sw = new System.Diagnostics.Stopwatch();
        for (int rep = 0; rep < 2; rep++)
            for (int i = 0; i < n; i++)
            {
                var d = _graph.Dispatches[i];
                Vk.vkResetCommandBuffer(cmd, 0);
                Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
                RecordDispatch(d);
                Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
                sw.Restart();
                _dev.Submit(cmd, fence);
                _dev.WaitFence(fence);
                double ms = sw.Elapsed.TotalMilliseconds;
                if (ms < best[i]) best[i] = ms;
            }
        _dev.DestroyFence(fence);
        _dev.FreeCommandBuffer(cmd);
        var agg = new Dictionary<string, double>();
        PerDispatch = new (string, double)[n];
        for (int i = 0; i < n; i++)
        {
            string name = _graph.Dispatches[i].Name ?? $"d{i}";
            PerDispatch[i] = (name, best[i]);
            name = name[..(name.IndexOf('(') is int cut && cut > 0 ? cut : name.Length)];
            agg[name] = agg.TryGetValue(name, out double v) ? v + best[i] : best[i];
        }
        return agg;
    }

    /// <summary>Per-dispatch timings from the last <see cref="DebugProfile"/> call.</summary>
    internal (string Name, double Ms)[]? PerDispatch;

    /// <summary>Debug/test: download an activation value's region into dst.</summary>
    internal void DebugDownload(int valueId, float[] dst)
    {
        CValue v = _plan.Values[valueId];
        ulong off = (ulong)_plan.BufOffset[v.Buf] * 4;
        fixed (float* p = dst)
            _dev.Download(Arena, p, (ulong)v.ElementCount * 4, off);
    }

    /// <summary>Debug/test: copy host input into the arena's input region.</summary>
    internal void DebugUploadInput()
    {
        _inHost.CopyTo(new Span<float>(_stageInMap, _inFloats));
        _stageIn.Flush(0, InputBytes);
        IntPtr cmd = _dev.NewCommandBuffer();
        IntPtr fence = _dev.NewFence();
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
        Vk.VkBufferCopy cin = new() { Size = InputBytes, DstOffset = _inOff };
        Vk.vkCmdCopyBuffer(cmd, _stageIn.Buffer, Arena.Buffer, 1, &cin);
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        _dev.Submit(cmd, fence);
        _dev.WaitFence(fence);
        _dev.DestroyFence(fence);
        _dev.FreeCommandBuffer(cmd);
    }

    /// <summary>Copy host input into staging and run the recorded pipeline.</summary>
    public void Run()
    {
        _inHost.CopyTo(new Span<float>(_stageInMap, _inFloats));
        _stageIn.Flush(0, InputBytes);
        _dev.Submit(_cmd, _fence);
        _dev.WaitFence(_fence);
        _stageOut.Invalidate(0, OutputBytes);
        new ReadOnlySpan<float>(_stageOutMap, _outFloats).CopyTo(_outHost);
    }

    public void Dispose()
    {
        if (_stageInMap != null) { _stageIn.Unmap(); _stageInMap = null; }
        if (_stageOutMap != null) { _stageOut.Unmap(); _stageOutMap = null; }
        _dev.DestroyFence(_fence);
        Arena.Free(); _stageIn.Free(); _stageOut.Free();
    }
}

