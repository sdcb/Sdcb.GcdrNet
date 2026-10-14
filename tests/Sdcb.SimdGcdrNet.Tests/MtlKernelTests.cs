using Sdcb.SimdGcdrNet.Backends.Metal;
using Sdcb.SimdGcdrNet.Engine;
using Sdcb.SimdGcdrNet.Kernels;
using Xunit;

namespace Sdcb.SimdGcdrNet.Tests;

/// <summary>Standalone Metal kernel vs CPU kernel checks.</summary>
public unsafe class MtlKernelTests
{
    private static float[] RunGpu(MtlDevice dev, string shader, float[][] ins, int outLen, byte[] pc, uint groups, int[]? pcs = null)
    {
        using var pool = AutoReleasePool.Create();
        var bufs = new List<MtlBuffer>();
        foreach (var f in ins)
        {
            var h = new Half[f.Length];
            for (int i = 0; i < f.Length; i++) h[i] = (Half)f[i];
            fixed (Half* hp = h) bufs.Add(dev.NewBuffer(hp, (nuint)(h.Length * 2)));
        }
        // extra inputs that must stay fp32 (bias, layernorm params) are passed via pcs tag
        var bufsF = new List<MtlBuffer>();
        var ob = dev.NewBuffer((nuint)(outLen * 2 + 64));
        var enc = dev.BeginCommands();
        enc.SetPipeline(dev.Pipe(shader));
        int bi = 0;
        foreach (var b in bufs) { enc.SetBuffer(b, 0, bi++); }
        enc.SetBuffer(ob, 0, bi++);
        if (pcs != null) { /* placeholder */ }
        fixed (byte* p = pc) enc.SetPc(p, (nuint)pc.Length, bi);
        enc.Dispatch(groups, 1, 1, 256, 1, 1);
        enc.End();
        enc.CommitAndWait();
        var dst = new float[outLen];
        var sp = new ReadOnlySpan<Half>(ob.Contents.ToPointer(), outLen);
        for (int i = 0; i < outLen; i++) dst[i] = (float)sp[i];
        foreach (var b in bufs) b.Dispose();
        ob.Dispose();
        return dst;
    }

    // generic: ins as fp16 tensors plus an fp32 tensor slot for bias
    private static float[] RunGpuMixed(MtlDevice dev, string shader,
        (float[] data, bool f32)[] ins, int outLen, byte[] pc, uint gx, uint gy = 1, uint gz = 1)
    {
        using var pool = AutoReleasePool.Create();
        var bufs = new List<MtlBuffer>();
        foreach (var (f, f32) in ins)
        {
            if (f32)
                fixed (float* fp = f) bufs.Add(dev.NewBuffer(fp, (nuint)(f.Length * 4)));
            else
            {
                var h = new Half[f.Length];
                for (int i = 0; i < f.Length; i++) h[i] = (Half)f[i];
                fixed (Half* hp = h) bufs.Add(dev.NewBuffer(hp, (nuint)(h.Length * 2)));
            }
        }
        var ob = dev.NewBuffer((nuint)(outLen * 2 + 64));
        var enc = dev.BeginCommands();
        enc.SetPipeline(dev.Pipe(shader));
        int bi = 0;
        foreach (var b in bufs) enc.SetBuffer(b, 0, bi++);
        enc.SetBuffer(ob, 0, bi++);
        fixed (byte* p = pc) enc.SetPc(p, (nuint)pc.Length, bi);
        enc.Dispatch(gx, gy, gz, 256, 1, 1);
        enc.End();
        enc.CommitAndWait();
        var dst = new float[outLen];
        var sp = new ReadOnlySpan<Half>(ob.Contents.ToPointer(), outLen);
        for (int i = 0; i < outLen; i++) dst[i] = (float)sp[i];
        foreach (var b in bufs) b.Dispose();
        ob.Dispose();
        return dst;
    }

    private static float[] Rand(Random r, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = r.NextSingle() * 2 - 1;
        return a;
    }

    [SkippableTheory]
    [InlineData("convpool", 3, 16, true, 0f, 1)]
    [InlineData("convpool_k3", 3, 16, true, 0f, 1)]
    [InlineData("convpool_k3", 6, 32, false, 0f, 0)]
    [InlineData("convpool_k3", 7, 8, false, 0.25f, 1)]
    [InlineData("convpool_k3", 1, 24, true, 0f, 0)]
    public void ConvPool_MatchesCpu(string shader, int ic, int oc, bool reflect, float padValue, int act)
    {
        Skip.IfNot(MetalOk(), "no Metal device");
        var dev = MetalBackend.Default.Dev;
        var rng = new Random(ic * 13 + oc);
        int h = 22, w = 30, ph = h / 2, pw = w / 2;
        float[] x = Rand(rng, h * w * ic), wt = Rand(rng, oc * ic * 9), b = Rand(rng, oc);
        var p = new ConvP
        {
            Pads = [1, 1, 1, 1], Strides = [1, 1], Dilations = [1, 1], Kernel = [3, 3],
            PadMode = reflect ? PadMode.Reflect : PadMode.Constant, PadValue = padValue,
            Act = act == 1 ? ActivationKind.Relu : ActivationKind.None,
            Pool = new PoolP { Kernel = [2, 2], Strides = [2, 2], Pads = [0, 0, 0, 0], FuseRelu = act == 0 },
        };
        float[] cpu = new float[ph * pw * oc];
        Nhwc.ConvPool(x, h, w, ic, cpu, ph, pw, oc, p, Nhwc.Pack(wt, [oc, ic, 3, 3], b, 1, ic));
        float[] wk = new float[9 * ic * oc];
        for (int o = 0; o < oc; o++)
            for (int c = 0; c < ic; c++)
                for (int t = 0; t < 9; t++) wk[(t * ic + c) * oc + o] = wt[(o * ic + c) * 9 + t];
        uint flags = 1u | (reflect ? 2u : 0u) | (act == 1 ? 4u : 0u) | (act == 0 ? 8u : 0u);
        byte[] pc = new byte[48];
        Buffer.BlockCopy(new uint[] { (uint)ic, (uint)h, (uint)w, (uint)oc, (uint)ph, (uint)pw, 3u, 3u, 1u, 1u, flags }, 0, pc, 0, 44);
        Buffer.BlockCopy(new float[] { reflect ? 0f : padValue }, 0, pc, 44, 4);
        float[] got = RunGpuMixed(dev, shader, [(x, false), (wk, false), (b, true)], cpu.Length, pc,
            (uint)(ph * pw * (oc / 8) + 255) / 256);
        AssertClose(got, cpu, 2e-2f, shader);
    }


    [SkippableTheory]
    [InlineData(6, 32, 1, false, "conv_mm_32_1", false, 1)]     // dr stem (scalar k gather)
    [InlineData(3, 16, 1, true, "conv_mm_32_1", false, 1)]      // gc stem, reflect
    [InlineData(32, 64, 1, true, "conv_mm_64_0", false, 1)]
    [InlineData(64, 128, 2, false, "conv_mm_128_0", false, 1)]
    [InlineData(16, 32, 1, false, "conv_mm_32_0", false, 1)]
    [InlineData(32, 64, 1, true, "conv_mm_64_0", true, 2)]      // residual, gelu
    [InlineData(64, 128, 2, false, "conv_mm_128_0", true, 0)]
    [InlineData(16, 160, 1, false, "conv_mm_32_0", false, 1)]
    [InlineData(32, 256, 1, false, "conv_mm_128_0", true, 1)]
    [InlineData(16, 16, 1, true, "conv_mm_32_0", false, 1)]
    [InlineData(16, 48, 1, false, "conv_mm_32_0", true, 2)]
    [InlineData(8, 64, 2, true, "conv_mm_64_0", true, 1)]
    [InlineData(3, 32, 1, false, "conv_mm_32_1", true, 2)]
    [InlineData(128, 128, 1, true, "conv_mm_128_0", true, 2)]
    public void ConvMm_MatchesCpu(int ic, int oc, int stride, bool reflect, string shader, bool residual, int act)
    {
        Skip.IfNot(MetalOk(), "no Metal device");
        var dev = MetalBackend.Default.Dev;
        var rng = new Random(ic * 7 + oc);
        int ih = 29, iw = 37, oh = (ih + 2 - 3) / stride + 1, ow = (iw + 2 - 3) / stride + 1;
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
        uint bn = shader.StartsWith("conv_mm_128") ? 128u : shader.StartsWith("conv_mm_64") ? 64u : 32u;
        float[] got = RunGpuMixed(dev, shader,
            [(x, false), (wt, false), (b, true), (residual ? res : new float[8], false)],
            cpu.Length, pc, (m + 63) / 64, ((uint)oc + bn - 1) / bn);
        float scale = Math.Max(1f, cpu.Max(MathF.Abs));
        AssertClose(got, cpu, 2e-3f * scale, shader);
    }


    [SkippableTheory]
    [InlineData(160)]
    [InlineData(256)]
    [InlineData(64)]
    [InlineData(96)]
    public void LayerNorm_MatchesCpu(int len)
    {
        Skip.IfNot(MetalOk(), "no Metal device");
        var dev = MetalBackend.Default.Dev;
        var rng = new Random(len);
        int rows = 97;
        float[] x = Rand(rng, rows * len).Select(v => (float)(Half)v).ToArray();
        float[] w = Rand(rng, len), b = Rand(rng, len);
        float eps = 1e-5f;
        float[] want = new float[rows * len];
        for (int r = 0; r < rows; r++)
        {
            double m = 0, q = 0;
            for (int i = 0; i < len; i++) { m += x[r * len + i]; q += x[r * len + i] * x[r * len + i]; }
            m /= len; q = Math.Sqrt(q / len - m * m + eps);
            for (int i = 0; i < len; i++) want[r * len + i] = (float)((x[r * len + i] - m) / q * w[i] + b[i]);
        }
        byte[] pc = new byte[12];
        Buffer.BlockCopy(new uint[] { (uint)len, (uint)rows }, 0, pc, 0, 8);
        Buffer.BlockCopy(new float[] { eps }, 0, pc, 8, 4);
        float[] got = RunGpuMixed(dev, "layernorm", [(x, false), (w, true), (b, true)], want.Length, pc, (uint)((rows + 7) / 8));
        AssertClose(got, want, 1e-2f, "layernorm");
    }

    private static void AssertClose(float[] got, float[] want, float tol, string tag)
    {
        double mx = 0, rmse = 0;
        for (int i = 0; i < want.Length; i++)
        {
            double d = Math.Abs(got[i] - want[i]);
            if (d > mx) mx = d;
            rmse += d * d;
        }
        rmse = Math.Sqrt(rmse / want.Length);
        Assert.True(mx < tol, $"{tag}: max {mx} rmse {rmse} (tol {tol})");
    }

    private static bool MetalOk()
    {
        try { return GcdrEnhancer.MetalUsable; }
        catch { return false; }
    }
}
