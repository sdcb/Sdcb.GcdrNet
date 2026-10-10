using Sdcb.GcdrNet.Backends.Vulkan;
using Sdcb.GcdrNet.Engine;
using Sdcb.GcdrNet.Kernels;
using Xunit;

namespace Sdcb.GcdrNet.Tests;

/// <summary>Standalone GPU kernel vs CPU kernel checks (bilinear, fused upsample tail).</summary>
public unsafe class VkKernelTests
{
    [SkippableTheory]
    [InlineData(0u, "bilinear")]   // plain
    [InlineData(7u, "bilinear")]   // relu(resize) + relu(addend)
    [InlineData(2u, "bilinear")]   // resize + addend
    [InlineData(0u, "bilinear4")]  // 4-channel variant (integrated GPUs)
    [InlineData(7u, "bilinear4")]
    public void Bilinear_MatchesCpu(uint flags, string shader)
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
        uint lanes = shader == "bilinear4" ? 4u : 1u;
        float[] got = RunGpu(dev, shader, [src, add], dstCpu.Length, pcb, ((uint)dstCpu.Length / lanes + 255) / 256);
        AssertClose(got, dstCpu, 1e-4f, shader);
    }

    [SkippableTheory]
    [InlineData("maxpool", 2, 2, 0, 0u)]
    [InlineData("maxpool4", 2, 2, 0, 0u)]
    [InlineData("maxpool4", 3, 2, 1, 1u)]   // padded taps, fused relu
    public void MaxPool_MatchesCpu(string shader, int k, int st, int pad, uint relu)
    {
        Skip.IfNot(VkOk(), "no Vulkan device");
        var dev = VkBackend.Default.Dev;
        var rng = new Random(5);
        int c = 8, ih = 9, iw = 11, oh = (ih + 2 * pad - k) / st + 1, ow = (iw + 2 * pad - k) / st + 1;
        float[] x = Rand(rng, ih * iw * c), cpu = new float[oh * ow * c];
        Nhwc.MaxPool(x, ih, iw, c, cpu, oh, ow,
            new PoolP { Kernel = [k, k], Strides = [st, st], Pads = [pad, pad, pad, pad], FuseRelu = relu != 0 });
        byte[] pc = new byte[48];
        Buffer.BlockCopy(new uint[] { (uint)c, (uint)ih, (uint)iw, (uint)oh, (uint)ow, (uint)k, (uint)k, (uint)st, (uint)st,
            (uint)pad, (uint)pad, relu }, 0, pc, 0, 48);
        uint lanes = shader == "maxpool4" ? 4u : 1u;
        float[] got = RunGpu(dev, shader, [x], cpu.Length, pc, ((uint)cpu.Length / lanes + 255) / 256);
        AssertClose(got, cpu, 1e-6f, shader);
    }

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
        Nhwc.ChannelShift(x, h, w, c, cpu, new ShiftP { Axis = axis, Offsets = of, GroupEnds = ge });
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
    [InlineData("depthwise", true, 1)]
    [InlineData("depthwise4", true, 1)]
    [InlineData("depthwise4", false, 3)]   // constant pad, gelu
    public void Depthwise_MatchesCpu(string shader, bool reflect, int act)
    {
        Skip.IfNot(VkOk(), "no Vulkan device");
        var dev = VkBackend.Default.Dev;
        var rng = new Random(8);
        int c = 12, h = 10, w = 13;
        float[] x = Rand(rng, h * w * c), wt = Rand(rng, c * 9), b = Rand(rng, c), cpu = new float[h * w * c];
        var p = new ConvP
        {
            Pads = [1, 1, 1, 1], Strides = [1, 1], Dilations = [1, 1], Kernel = [3, 3], Group = c,
            PadMode = reflect ? PadMode.Reflect : PadMode.Constant, Act = (ActivationKind)act,
        };
        Nhwc.Conv(x, h, w, c, cpu, h, w, c, p, Nhwc.Pack(wt, [c, 1, 3, 3], b, c, c), default);
        float[] wp = new float[9 * c];   // [tap][C]
        for (int ch = 0; ch < c; ch++)
            for (int t = 0; t < 9; t++) wp[t * c + ch] = wt[ch * 9 + t];
        uint flags = 1u | (reflect ? 2u : 0u) | (act == 1 ? 4u : 0u) | (act == 3 ? 16u : 0u);
        byte[] pc = new byte[52];
        Buffer.BlockCopy(new uint[] { (uint)c, (uint)h, (uint)w, (uint)h, (uint)w, 3u, 3u, 1u, 1u, 1u, 1u, flags }, 0, pc, 0, 48);
        Buffer.BlockCopy(new float[] { 0f }, 0, pc, 48, 4);
        uint lanes = shader == "depthwise4" ? 4u : 1u;
        float[] got = RunGpu(dev, shader, [x, wp, b], cpu.Length, pc, ((uint)cpu.Length / lanes + 255) / 256);
        AssertClose(got, cpu, 2e-3f, shader);   // GPU erf polynomial vs CPU erf for gelu
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

    // lite: false = the device's variant (LITE on wave64 / 32 KB parts), true = force the LITE build
    [SkippableTheory]
    [InlineData(6, 32, 1, false, "conv_cm_n32s", false, false, 1)]    // dr stem (scalar k gather)
    [InlineData(3, 16, 1, true, "conv_cm_n32s", false, false, 1)]     // gc stem, reflect
    [InlineData(32, 64, 1, true, "conv_cm_n64", false, false, 1)]
    [InlineData(64, 128, 2, false, "conv_cm", false, false, 1)]       // stride 2
    [InlineData(16, 32, 1, false, "conv_cm_n32", false, false, 1)]
    [InlineData(6, 32, 1, false, "conv_cm_n32s", true, false, 1)]
    [InlineData(32, 64, 1, true, "conv_cm_n64", true, true, 2)]       // residual, gelu
    [InlineData(64, 128, 2, false, "conv_cm", true, true, 0)]
    [InlineData(16, 160, 1, false, "conv_cm_n32", true, false, 1)]    // 5 N tiles: partial raster group
    [InlineData(32, 256, 1, false, "conv_cm", false, true, 1)]
    public void ConvCm_MatchesCpu(int ic, int oc, int stride, bool reflect, string shader, bool lite, bool residual, int act)
    {
        Skip.IfNot(VkOk() && VkBackend.Default.Dev.CoopMat16, "no coopmat device");
        var dev = VkBackend.Default.Dev;
        var rng = new Random(ic * 7 + oc);
        int ih = 29, iw = 37, oh = (ih + 2 - 3) / stride + 1, ow = (iw + 2 - 3) / stride + 1;
        // fp16-representable inputs so CPU and GPU see identical operands
        float[] x = Rand(rng, ih * iw * ic).Select(v => (float)(Half)v).ToArray();
        float[] w = Rand(rng, oc * ic * 9).Select(v => (float)(Half)(v * 0.25f)).ToArray();
        float[] b = Rand(rng, oc);
        var p = new ConvP
        {
            Pads = [1, 1, 1, 1], Strides = [stride, stride], Dilations = [1, 1], Kernel = [3, 3],
            PadMode = reflect ? PadMode.Reflect : PadMode.Constant,
            Act = act == 1 ? ActivationKind.Relu : act == 2 ? ActivationKind.Gelu : ActivationKind.None,
        };
        float[] res = residual ? Rand(rng, oh * ow * oc).Select(v => (float)(Half)v).ToArray() : [];
        float[] cpu = new float[oh * ow * oc];
        NhwcConvPlan kp = Nhwc.Pack(w, [oc, ic, 3, 3], b, 1, ic);
        Nhwc.Conv(x, ih, iw, ic, cpu, oh, ow, oc, p, kp, res);

        int kr = 9 * ic, kd = (kr + 7) & ~7;
        float[] wt = new float[oc * kd];
        for (int o = 0; o < oc; o++)
            for (int c = 0; c < ic; c++)
                for (int t = 0; t < 9; t++) wt[o * kd + t * ic + c] = w[(o * ic + c) * 9 + t];
        uint flags = 1u | (residual ? 2u : 0u) | (reflect ? 4u : 0u) | ((uint)act << 4);
        uint m = (uint)(oh * ow);
        byte[] pc = new byte[56];
        Buffer.BlockCopy(new uint[] { m, (uint)oc, (uint)kd, flags, (uint)ow, (uint)iw, (uint)ih, (uint)ic, 3u,
            (uint)stride, (uint)stride, 1u, 1u, (uint)kr }, 0, pc, 0, 56);
        uint bn = shader == "conv_cm" ? 128u : shader == "conv_cm_n64" ? 64u : 32u;
        var bufs = new List<VkBuffer> { Half16(dev, x), Half16(dev, wt), F32(dev, b), Half16(dev, residual ? res : new float[8]) };
        VkBuffer ob = dev.NewStorageBuffer((ulong)cpu.Length * 2 + 16, true, preferHost: true);
        bufs.Add(ob);
        VkPipeline pipe = dev.GetCoopPipeline(shader, 5, 56, lite ? true : null);
        IntPtr cmd = dev.NewCommandBuffer();
        IntPtr fence = dev.NewFence();
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "begin");
        VkInferenceSession.RecordOne(dev, cmd, pipe, bufs.Select(bb => (bb, 0UL, bb.Size)).ToArray(), pc,
            (m + 127) / 128, ((uint)oc + bn - 1) / bn);
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "end");
        dev.Submit(cmd, fence); dev.WaitFence(fence);
        dev.DestroyFence(fence); dev.FreeCommandBuffer(cmd);
        Half[] gh = new Half[cpu.Length];
        fixed (Half* gp = gh) dev.Download(ob, gp, (ulong)gh.Length * 2);
        foreach (VkBuffer bb in bufs) bb.Free();
        float[] got = gh.Select(h => (float)h).ToArray();
        float scale = Math.Max(1f, cpu.Max(MathF.Abs));
        AssertClose(got, cpu, 2e-3f * scale, shader);
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
