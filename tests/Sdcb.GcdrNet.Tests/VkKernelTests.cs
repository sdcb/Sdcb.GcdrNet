using Sdcb.GcdrNet.Backends.Vulkan;
using Sdcb.GcdrNet.Engine;
using Sdcb.GcdrNet.Kernels;
using Xunit;

namespace Sdcb.GcdrNet.Tests;

/// <summary>Standalone GPU kernel vs CPU kernel checks (bilinear, fused upsample tail).</summary>
public unsafe class VkKernelTests
{
    [SkippableTheory]
    [InlineData(0u)]   // plain
    [InlineData(7u)]   // relu(resize) + relu(addend)
    [InlineData(2u)]   // resize + addend
    public void Bilinear_MatchesCpu(uint flags)
    {
        Skip.IfNot(VkOk(), "no Vulkan device");
        var dev = VkBackend.Default.Dev;
        var rng = new Random(42);
        int c = 4, ih = 3, iw = 4, oh = 6, ow = 8;
        float[] src = Rand(rng, c * ih * iw);
        float[] add = Rand(rng, c * oh * ow);
        float[] dstCpu = new float[c * oh * ow];
        bool hasAdd = (flags & 2u) != 0;
        // shaders address NHWC storage: src is [ih][iw][c]
        Nhwc.ResizeBilinear(src, ih, iw, c, dstCpu, oh, ow, 2f, 2f,
            (flags & 1u) != 0 ? ActivationKind.Relu : ActivationKind.None,
            hasAdd ? add : default, (flags & 4u) != 0);

        // PC layout: 5 uints + flags uint + 2 floats
        byte[] pcb = new byte[32];
        Buffer.BlockCopy(new uint[] { (uint)c, (uint)ih, (uint)iw, (uint)oh, (uint)ow, flags }, 0, pcb, 0, 24);
        Buffer.BlockCopy(new float[] { 2f, 2f }, 0, pcb, 24, 8);
        float[] got = RunGpu(dev, "bilinear", [src, add], dstCpu.Length, pcb, (uint)(dstCpu.Length + 255) / 256);
        AssertClose(got, dstCpu, 1e-4f, "bilinear");
    }

    [SkippableTheory]
    [InlineData(3, 16, 2u, 2u)]    // gc tail: relu(skip), bias, sigmoid
    [InlineData(3, 32, 3u, 2u)]    // + relu(resize)
    [InlineData(5, 20, 0u, 1u)]    // odd oc, relu act
    public void UpHead_MatchesCpu(int oc, int c, uint reluFlags, uint act)
    {
        Skip.IfNot(VkOk(), "no Vulkan device");
        var dev = VkBackend.Default.Dev;
        var rng = new Random(7);
        int ih = 5, iw = 7, oh = 10, ow = 14;
        float[] x = Rand(rng, ih * iw * c), skip = Rand(rng, oh * ow * c);
        float[] w = Rand(rng, oc * c), b = Rand(rng, oc);
        var p = new UpHeadP
        {
            Scales = [1f, 1f, 2f, 2f],
            ResizeRelu = (reluFlags & 1u) != 0, SkipRelu = (reluFlags & 2u) != 0,
            Act = act == 2u ? ActivationKind.Sigmoid : act == 1u ? ActivationKind.Relu : ActivationKind.None,
        };
        float[] cpu = new float[oh * ow * oc];
        Nhwc.UpHead(x, ih, iw, c, skip, cpu, oh, ow, 2f, 2f, p, Nhwc.PackUpHead(w, oc, c, b));

        uint flags = reluFlags | 4u | (act << 3);
        byte[] pcb = new byte[36];
        Buffer.BlockCopy(new uint[] { (uint)c, (uint)ih, (uint)iw, (uint)oh, (uint)ow, (uint)oc, flags }, 0, pcb, 0, 28);
        Buffer.BlockCopy(new float[] { 2f, 2f }, 0, pcb, 28, 8);
        float[] got = RunGpu(dev, "uphead", [x, skip, w, b], cpu.Length, pcb, (uint)(oh * ow + 255) / 256);
        AssertClose(got, cpu, 1e-4f, "uphead");
    }

    private static float[] Rand(Random rng, int n) =>
        Enumerable.Range(0, n).Select(_ => (float)rng.NextDouble() * 4 - 2).ToArray();

    private static void AssertClose(float[] got, float[] want, float tol, string tag)
    {
        float maxd = 0; int at = 0;
        for (int i = 0; i < got.Length; i++)
            if (Math.Abs(got[i] - want[i]) > maxd) { maxd = Math.Abs(got[i] - want[i]); at = i; }
        Assert.True(maxd < tol, $"{tag} maxd {maxd} @{at}: gpu={got[at]} cpu={want[at]}");
    }

    /// <summary>Runs one fp32-storage shader: inputs bound 0..n-1, output last.</summary>
    private static float[] RunGpu(VkDevice dev, string shader, float[][] inputs, int outLen, byte[] pc, uint gx)
    {
        var bufs = new List<VkBuffer>();
        foreach (float[] a in inputs)
        {
            VkBuffer bb = dev.NewStorageBuffer((ulong)a.Length * 4, true);
            fixed (float* sp = a) dev.Upload(bb, sp, (ulong)a.Length * 4);
            bufs.Add(bb);
        }
        VkBuffer ob = dev.NewStorageBuffer((ulong)outLen * 4, true, preferHost: true);
        bufs.Add(ob);
        VkPipeline pipe = dev.GetPipeline(shader, bufs.Count, pc.Length);
        IntPtr cmd = dev.NewCommandBuffer();
        IntPtr fence = dev.NewFence();
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "begin");
        VkInferenceSession.RecordOne(dev, cmd, pipe, bufs.Select(bb => (bb, 0UL, bb.Size)).ToArray(), pc, gx, 1);
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "end");
        dev.Submit(cmd, fence); dev.WaitFence(fence);
        dev.DestroyFence(fence); dev.FreeCommandBuffer(cmd);
        float[] got = new float[outLen];
        fixed (float* gp = got) dev.Download(ob, gp, (ulong)outLen * 4);
        foreach (VkBuffer bb in bufs) bb.Free();
        return got;
    }
    private static bool VkOk()
    {
        try { _ = VkBackend.Default.Dev.DeviceName; return true; }
        catch { return false; }
    }
}
