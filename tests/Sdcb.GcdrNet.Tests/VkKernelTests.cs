using Sdcb.GcdrNet.Backends.Vulkan;
using Sdcb.GcdrNet.Engine;
using Sdcb.GcdrNet.Kernels;
using Xunit;

namespace Sdcb.GcdrNet.Tests;

/// <summary>Standalone GPU kernel vs CPU kernel checks (bilinear, conv, gemm).</summary>
public unsafe class VkKernelTests
{
    [SkippableFact]
    public void Bilinear_MatchesCpu()
    {
        Skip.IfNot(VkOk(), "no Vulkan device");
        var dev = VkBackend.Default.Dev;
        var rng = new Random(42);
        int c = 4, ih = 3, iw = 4, oh = 6, ow = 8;
        float[] src = Enumerable.Range(0, c * ih * iw).Select(_ => (float)rng.NextDouble() * 4 - 2).ToArray();
        float[] dstCpu = new float[c * oh * ow];
        Ops.ResizeBilinear(src, [1, c, ih, iw], dstCpu, [1, c, oh, ow], [1, 1, 2, 2]);

        VkBuffer xb = dev.NewStorageBuffer((ulong)src.Length * 4, true);
        VkBuffer ob = dev.NewStorageBuffer((ulong)dstCpu.Length * 4, true, preferHost: true);
        fixed (float* sp = src) dev.Upload(xb, sp, (ulong)src.Length * 4);

        VkPipeline pipe = dev.GetPipeline("bilinear", 2, 8 * 4);
        IntPtr cmd = dev.NewCommandBuffer();
        IntPtr fence = dev.NewFence();
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "begin");
        uint[] pc = [(uint)c, (uint)ih, (uint)iw, (uint)oh, (uint)ow, 0u, 0u, 0u];
        // PC layout: 5 uints + relu uint + 2 floats
        byte[] pcb = new byte[32];
        Buffer.BlockCopy(pc, 0, pcb, 0, 24);
        Buffer.BlockCopy(new float[] { 2f, 2f }, 0, pcb, 24, 8);
        VkInferenceSession.RecordOne(dev, cmd, pipe, [(xb, 0UL, xb.Size), (ob, 0UL, ob.Size)],
            pcb, (uint)(dstCpu.Length + 255) / 256, 1);
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "end");
        dev.Submit(cmd, fence); dev.WaitFence(fence);
        dev.DestroyFence(fence); dev.FreeCommandBuffer(cmd);

        float[] got = new float[dstCpu.Length];
        fixed (float* gp = got) dev.Download(ob, gp, (ulong)got.Length * 4);
        float maxd = 0; int at = -1;
        for (int i = 0; i < got.Length; i++)
            if (Math.Abs(got[i] - dstCpu[i]) > maxd) { maxd = Math.Abs(got[i] - dstCpu[i]); at = i; }
        Assert.True(maxd < 1e-4, $"bilinear maxd {maxd} @{at}: gpu={got[at]} cpu={dstCpu[at]}");
        xb.Free(); ob.Free();
    }

    private static bool VkOk()
    {
        try { _ = VkBackend.Default.Dev.DeviceName; return true; }
        catch { return false; }
    }
}
