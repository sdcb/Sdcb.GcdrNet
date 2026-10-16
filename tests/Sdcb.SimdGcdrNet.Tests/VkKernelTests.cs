using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdGcdrNet.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdGcdrNet.Extensions;
using Sdcb.SimdOnnx.Kernels;
using Sdcb.SimdGcdrNet.Kernels;
using Xunit;

namespace Sdcb.SimdGcdrNet.Tests;

/// <summary>Standalone GPU kernel vs CPU kernel checks (bilinear, fused upsample tail).</summary>
public unsafe class VkKernelTests
{

    [SkippableTheory]
    [InlineData("shift", 2)]
    [InlineData("shift4", 2)]
    [InlineData("shift4", 3)]
    public void Shift_MatchesCpu(string shader, int axis)
    {
        Skip.IfNot(VkOk(), "no Vulkan device");
        var dev = VkBackend.Default.Dev;
        var rng = new Random(6);
        int c = 16, h = 7, w = 9;
        int[] ge = [4, 8, 12, 16], of = [-2, -1, 1, 2];
        float[] x = Rand(rng, h * w * c), cpu = new float[h * w * c];
        GcdrKernels.ChannelShift(x, h, w, c, cpu, new ShiftP { Axis = axis, Offsets = of, GroupEnds = ge });
        var pcu = new List<uint> { (uint)c, (uint)h, (uint)w, (uint)axis };
        for (int i = 0; i < 5; i++) pcu.Add((uint)(i < ge.Length ? ge[i] : c));
        for (int i = 0; i < 5; i++) pcu.Add((uint)(i < of.Length ? of[i] : 0));
        byte[] pc = new byte[pcu.Count * 4];
        Buffer.BlockCopy(pcu.ToArray(), 0, pc, 0, pc.Length);
        uint lanes = shader == "shift4" ? 4u : 1u;
        float[] got = RunGpu(dev, shader, [x], cpu.Length, pc, ((uint)cpu.Length / lanes + 255) / 256);
        AssertClose(got, cpu, 1e-6f, shader);
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
        GcdrKernels.UpHead(x, ih, iw, c, skip, cpu, oh, ow, 2f, 2f, p, GcdrKernels.PackUpHead(w, oc, c, b));

        uint flags = reluFlags | 4u | (act << 3);
        byte[] pcb = new byte[36];
        Buffer.BlockCopy(new uint[] { (uint)c, (uint)ih, (uint)iw, (uint)oh, (uint)ow, (uint)oc, flags }, 0, pcb, 0, 28);
        Buffer.BlockCopy(new float[] { 2f, 2f }, 0, pcb, 28, 8);
        float[] got = RunGpu(dev, "uphead", [x, skip, w, b], cpu.Length, pcb, (uint)(oh * ow + 255) / 256);
        AssertClose(got, cpu, 1e-4f, "uphead");
    }

    private static VkBuffer Half16(VkDevice dev, float[] a)
    {
        Half[] h = a.Select(v => (Half)v).ToArray();
        VkBuffer bb = dev.NewStorageBuffer((ulong)h.Length * 2 + 16, true);
        fixed (Half* hp = h) dev.Upload(bb, hp, (ulong)h.Length * 2);
        return bb;
    }

    private static VkBuffer F32(VkDevice dev, float[] a)
    {
        VkBuffer bb = dev.NewStorageBuffer((ulong)a.Length * 4, true);
        fixed (float* fp = a) dev.Upload(bb, fp, (ulong)a.Length * 4);
        return bb;
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
        VkPipeline pipe = dev.GetPipeline(shader, bufs.Count, pc.Length, owner: GcdrExtensions.Owner);
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
