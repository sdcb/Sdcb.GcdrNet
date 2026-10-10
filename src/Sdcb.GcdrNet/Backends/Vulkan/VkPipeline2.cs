using Sdcb.GcdrNet.Engine;

namespace Sdcb.GcdrNet.Backends.Vulkan;

/// <summary>
/// Fused GPU pipeline for one padded size: BGR u8 upload → bgr2nchw shader →
/// GC-Net → div6 (division correction + 6ch build) → DR-Net → nchw2bgr →
/// u8 readback. One command buffer, one submit per frame.
/// </summary>
internal sealed unsafe class VkGcdrPipeline : IDisposable
{
    private readonly VkDevice _dev;
    private readonly VkInferenceSession _gc, _dr;
    private readonly VkBuffer _bgrIn, _bgrOut;
    private byte* _bgrInMap, _bgrOutMap;
    private readonly IntPtr _cmd, _fence;
    private readonly int _h, _w;          // padded dims
    private readonly int _ih, _iw;        // original dims
    private readonly int _pt, _pl;
    private readonly VkPipeline _pPre, _pDiv6, _pPost;

    internal VkGcdrPipeline(VkDevice dev, ExecPlan gcPlan, ExecPlan drPlan,
        VkGraphModel gcGraph, VkGraphModel drGraph, int ih, int iw)
    {
        _dev = dev;
        _ih = ih; _iw = iw;
        _pl = (-iw) & 31; _pt = (-ih) & 31;
        _h = ih + _pt; _w = iw + _pl;
        _gc = new VkInferenceSession(dev, gcPlan, gcGraph);
        _dr = new VkInferenceSession(dev, drPlan, drGraph);

        ulong inBytes = (ulong)ih * (ulong)iw * 3;
        _bgrIn = dev.NewStorageBuffer(inBytes + 4, hostVisible: true, preferHost: false);
        _bgrOut = dev.NewStorageBuffer(inBytes + 4, hostVisible: true, preferHost: true);
        _bgrInMap = (byte*)_bgrIn.Map();
        _bgrOutMap = (byte*)_bgrOut.Map();
        _fence = dev.NewFence();

        _pPre = dev.GetPipeline("bgr2nchw", 2, 7 * 4);
        _pDiv6 = dev.GetPipeline("div6", 3, 4);
        _pPost = dev.GetPipeline("nchw2bgr", 2, 7 * 4);
        _cmd = dev.NewCommandBuffer();
        Record((uint)(_iw * 3));   // _bgrIn is always tightly packed (we de-stride on upload)
    }

    private static byte[] Pc(params uint[] u)
    {
        byte[] b = new byte[u.Length * 4];
        Buffer.BlockCopy(u, 0, b, 0, b.Length);
        return b;
    }

    private void Record(uint srcStride)
    {
        uint h = (uint)_h, w = (uint)_w;
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(_cmd, &begin), "vkBeginCommandBuffer");

        if (Environment.GetEnvironmentVariable("GCDR_ZERO_ARENA") == "1")
        {
            Vk.vkCmdFillBuffer(_cmd, _gc.Arena.Buffer, 0, _gc.Arena.Size, 0);
            Vk.vkCmdFillBuffer(_cmd, _dr.Arena.Buffer, 0, _dr.Arena.Size, 0);
            Vk.VkMemoryBarrier fb = new()
            {
                SType = VkConst.StMemoryBarrier,
                SrcAccessMask = VkConst.AccessTransferWrite, DstAccessMask = VkConst.AccessShaderRead | VkConst.AccessShaderWrite,
            };
            Vk.vkCmdPipelineBarrier(_cmd, VkConst.PipelineStageTransfer, VkConst.PipelineStageComputeShader, 0,
                1, &fb, 0, null, 0, null);
        }

        // bgr2nchw -> gc arena input
        VkInferenceSession.RecordOne(_dev, _cmd, _pPre,
            [(_bgrIn, 0UL, _bgrIn.Size), (_gc.Arena, _gc.InputOffset, _gc.InputBytes)],
            Pc(h, w, (uint)_ih, (uint)_iw, (uint)_pt, (uint)_pl, (uint)srcStride),
            (w + 255) / 256, h, 3);
        _dev.CmdComputeBarrier(_cmd);
        _gc.RecordCompute(_cmd);

        // div6: gc input + gc output -> dr arena input
        uint hw = h * w;
        _dev.CmdComputeBarrier(_cmd);
        VkInferenceSession.RecordOne(_dev, _cmd, _pDiv6,
            [(_gc.Arena, _gc.InputOffset, 3UL * hw * 4),
             (_gc.Arena, _gc.OutputOffset, 3UL * hw * 4),
             (_dr.Arena, _dr.InputOffset, 6UL * hw * 4)],
            Pc(hw), (3u * hw + 255) / 256, 1);
        _dev.CmdComputeBarrier(_cmd);
        _dr.RecordCompute(_cmd);

        // nchw2bgr -> bgrOut
        _dev.CmdComputeBarrier(_cmd);
        uint nwords = ((uint)_ih * (uint)_iw * 3u + 3u) / 4u;
        VkInferenceSession.RecordOne(_dev, _cmd, _pPost,
            [(_dr.Arena, _dr.OutputOffset, _dr.OutputBytes), (_bgrOut, 0UL, _bgrOut.Size)],
            Pc(h, w, (uint)_ih, (uint)_iw, (uint)_pt, (uint)_pl, nwords),   // PC layout must match nchw2bgr.comp
            (nwords + 255) / 256, 1);
        Vk.Check(Vk.vkEndCommandBuffer(_cmd), "vkEndCommandBuffer");
    }

    public void Enhance(ReadOnlySpan<byte> bgrSrc, int srcStride, Span<byte> bgrDst, int dstStride)
    {
        ulong inBytes = (ulong)_ih * (ulong)_iw * 3;
        if (srcStride == _iw * 3)
            bgrSrc.Slice(0, (int)inBytes).CopyTo(new Span<byte>(_bgrInMap, (int)inBytes));
        else
            for (int y = 0; y < _ih; y++)
                bgrSrc.Slice(y * srcStride, _iw * 3).CopyTo(
                    new Span<byte>(_bgrInMap + y * _iw * 3, _iw * 3));
        _bgrIn.Flush(0, inBytes);

        _dev.Submit(_cmd, _fence);
        _dev.WaitFence(_fence);
        _bgrOut.Invalidate(0, inBytes);

        if (dstStride == _iw * 3)
            new ReadOnlySpan<byte>(_bgrOutMap, (int)inBytes).CopyTo(bgrDst);
        else
            for (int y = 0; y < _ih; y++)
                new ReadOnlySpan<byte>(_bgrOutMap + y * _iw * 3, _iw * 3)
                    .CopyTo(bgrDst.Slice(y * dstStride));
    }
    internal VkInferenceSession DebugGc => _gc;
    internal VkInferenceSession DebugDr => _dr;

    /// <summary>Debug: download intermediate tensors (gc nchw input, gc shadow out, dr 6ch input — all padded).</summary>
    internal void DebugIntermediates(Span<float> gcIn, Span<float> gcOut, Span<float> drIn)
    {
        uint hw = (uint)(_h * _w);
        fixed (float* p = gcIn) _dev.Download(_gc.Arena, p, 3UL * hw * 4, _gc.InputOffset);
        fixed (float* p = gcOut) _dev.Download(_gc.Arena, p, 3UL * hw * 4, _gc.OutputOffset);
        fixed (float* p = drIn) _dev.Download(_dr.Arena, p, 6UL * hw * 4, _dr.InputOffset);
    }

    public void Dispose()
    {
        // Enhance() is synchronous (already waits the fence); callers must not race Dispose
        _dev.FreeCommandBuffer(_cmd);
        _bgrIn.Unmap(); _bgrOut.Unmap();
        _dev.DestroyFence(_fence);
        _bgrIn.Free(); _bgrOut.Free();
        _gc.Dispose(); _dr.Dispose();
    }
}
