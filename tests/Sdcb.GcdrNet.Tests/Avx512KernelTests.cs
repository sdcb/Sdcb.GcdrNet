using System.Runtime.Intrinsics.X86;
using Sdcb.GcdrNet.Engine;
using Sdcb.GcdrNet.Kernels;

namespace Sdcb.GcdrNet.Tests;

/// <summary>
/// AVX-512 tier vs the AVX2 tier in-process (<see cref="Nhwc.UseAvx512"/> toggled): the wider
/// kernels keep every output element's reduction order and epilogue math, so results must be
/// bit-identical. Skipped on machines (or JIT settings) without AVX-512.
/// </summary>
[Collection(TierToggle.Name)]
public class Avx512KernelTests
{
    private static readonly bool s_tier = Nhwc.UseAvx512;

    private static float[] Both(Action<float[]> run, int n)
    {
        Skip.IfNot(s_tier && Avx2.IsSupported && Fma.IsSupported, "AVX-512 tier not active");
        float[] avx2 = new float[n], avx512 = new float[n];
        try
        {
            Nhwc.UseAvx512 = false;
            run(avx2);
            Nhwc.UseAvx512 = true;
            run(avx512);
        }
        finally { Nhwc.UseAvx512 = s_tier; }
        for (int i = 0; i < n; i++)
            if (BitConverter.SingleToInt32Bits(avx2[i]) != BitConverter.SingleToInt32Bits(avx512[i]))
                Assert.Fail($"element {i}: avx2 {avx2[i]:R} vs avx512 {avx512[i]:R}");
        return avx512;
    }

    [SkippableTheory]
    [InlineData(3, 16, 3, 1, 37, 23, 1, 0f, false, 0)]   // gc stem: 16-wide tail kernel, reflect, row runs
    [InlineData(6, 32, 3, 1, 29, 41, 1, 0f, false, 1)]   // dr stem: one block pair
    [InlineData(16, 48, 3, 1, 11, 13, 0, 0.25f, true, 1)] // odd block count, pad value, residual
    [InlineData(20, 20, 3, 2, 17, 19, 0, 0f, true, 2)]   // stride 2, oc lane tail, sigmoid
    [InlineData(32, 64, 3, 2, 12, 31, 1, 0f, false, 3)]  // stride 2, reflect, gelu
    [InlineData(24, 3, 3, 1, 9, 10, 0, 0f, false, 0)]    // oc=3 (single partial lane block)
    [InlineData(40, 36, 1, 1, 7, 33, 0, 0f, true, 1)]    // 1x1 conv
    [InlineData(8, 32, 5, 1, 14, 9, 1, 0f, false, 1)]    // 5x5 (direct path), narrow rows
    public void Conv_Direct_MatchesAvx2(int ic, int oc, int k, int stride, int h, int w, int mode,
        float padValue, bool residual, int act)
    {
        var rng = new Random(ic * 131 + oc * 7 + k);
        int pad = k / 2;
        int oh = (h + 2 * pad - k) / stride + 1, ow = (w + 2 * pad - k) / stride + 1;
        float[] x = rng.Array(h * w * ic), wt = rng.Array(oc * ic * k * k), b = rng.Array(oc);
        float[] res = residual ? rng.Array(oh * ow * oc) : [];
        var p = new ConvP
        {
            Pads = [pad, pad, pad, pad], Strides = [stride, stride], Dilations = [1, 1], Kernel = [k, k],
            PadMode = (PadMode)mode, PadValue = padValue, Act = (ActivationKind)act,
        };
        NhwcConvPlan kp = Nhwc.Pack(wt, [oc, ic, k, k], b, 1, ic);   // no Winograd: direct kernels
        Both(dst => Nhwc.Conv(x, h, w, ic, dst, oh, ow, oc, p, kp, res), oh * ow * oc);
    }

    [SkippableTheory]
    [InlineData(16, 16, 9, 11, 0, 0f, false, 0)]
    [InlineData(24, 20, 13, 6, 1, 0f, true, 1)]   // ic % 16 != 0: 8-lane input transform
    [InlineData(32, 48, 22, 18, 0, 0.5f, false, 3)]
    [InlineData(64, 16, 17, 23, 1, 0f, true, 0)]
    [InlineData(128, 64, 30, 41, 1, 0f, false, 1)] // several tile chunks per image
    public void Conv_Winograd_MatchesAvx2(int ic, int oc, int h, int w, int mode, float padValue, bool residual, int act)
    {
        var rng = new Random(ic * 17 + oc);
        float[] x = rng.Array(h * w * ic), wt = rng.Array(oc * ic * 9), b = rng.Array(oc);
        float[] res = residual ? rng.Array(h * w * oc) : [];
        var p = new ConvP
        {
            Pads = [1, 1, 1, 1], Strides = [1, 1], Dilations = [1, 1], Kernel = [3, 3],
            PadMode = (PadMode)mode, PadValue = padValue, Act = (ActivationKind)act,
        };
        Skip.IfNot(Nhwc.WinogradEligible(p, [oc, ic, 3, 3], ic), "Winograd disabled");
        NhwcConvPlan kp = Nhwc.Pack(wt, [oc, ic, 3, 3], b, 1, ic);
        kp.WinoU = Nhwc.PackWinograd(wt, oc, ic);
        Both(dst => Nhwc.Conv(x, h, w, ic, dst, h, w, oc, p, kp, res), h * w * oc);
    }

    [SkippableTheory]
    [InlineData(57, 128, 128, false, 0)]
    [InlineData(23, 160, 48, true, 3)]   // odd block count, transB, gelu
    [InlineData(5, 64, 20, false, 1)]    // fewer rows than a tile, lane tail
    public void Gemm_MatchesAvx2(int m, int k, int n, bool transB, int act)
    {
        var rng = new Random(m + k + n);
        float[] a = rng.Array(m * k), bm = rng.Array(k * n), bias = rng.Array(n);
        NhwcConvPlan kp = Nhwc.PackGemm(bm, k, n, transB, bias);
        Both(dst => Nhwc.Gemm(a, m, k, dst, n, kp, (ActivationKind)act), m * n);
    }

    [SkippableTheory]
    [InlineData(32, 2, 2, 0, false)]
    [InlineData(16, 3, 2, 1, true)]    // padded taps, fused relu
    public void MaxPool_MatchesAvx2(int c, int k, int s, int pad, bool relu)
    {
        var rng = new Random(c + k);
        int h = 19, w = 26, oh = (h + 2 * pad - k) / s + 1, ow = (w + 2 * pad - k) / s + 1;
        float[] x = rng.Array(h * w * c);
        var p = new PoolP { Kernel = [k, k], Strides = [s, s], Pads = [pad, pad, pad, pad], FuseRelu = relu };
        Both(dst => Nhwc.MaxPool(x, h, w, c, dst, oh, ow, p), oh * ow * c);
    }

    [SkippableTheory]
    [InlineData(32, true, true, true)]
    [InlineData(16, false, false, false)]
    [InlineData(24, true, false, true)]   // c % 16 != 0: generic resize
    public void Resize_MatchesAvx2(int c, bool relu, bool addend, bool addRelu)
    {
        var rng = new Random(c);
        int h = 13, w = 9, oh = 26, ow = 18;
        float[] x = rng.Array(h * w * c), add = addend ? rng.Array(oh * ow * c) : [];
        Both(dst => Nhwc.ResizeBilinear(x, h, w, c, dst, oh, ow, 2, 2,
            relu ? ActivationKind.Relu : ActivationKind.None, add, addRelu), oh * ow * c);
    }

    [SkippableTheory]
    [InlineData(32, 3, true, true, 2)]
    [InlineData(16, 3, false, true, 0)]
    [InlineData(24, 5, true, false, 1)]   // c % 16 != 0, oc 5
    public void UpHead_MatchesAvx2(int c, int oc, bool resizeRelu, bool skipRelu, int act)
    {
        var rng = new Random(c * oc);
        int h = 11, w = 15, oh = 22, ow = 30;   // ow % 4 != 0: partial pixel group
        float[] x = rng.Array(h * w * c), skip = rng.Array(oh * ow * c), wt = rng.Array(oc * c), b = rng.Array(oc);
        UpHeadPlan kp = Nhwc.PackUpHead(wt, oc, c, b);
        var p = new UpHeadP { Scales = [1, 1, 2, 2], ResizeRelu = resizeRelu, SkipRelu = skipRelu, Act = (ActivationKind)act };
        Both(dst => Nhwc.UpHead(x, h, w, c, skip, dst, oh, ow, 2, 2, p, kp), oh * ow * oc);
    }

    [SkippableTheory]
    [InlineData(3, 16, false, 1, 0, 1)]    // gc stem: 16-wide tail kernel, reflect, row runs
    [InlineData(6, 32, false, 1, 0, 1)]    // dr stem: one block pair
    [InlineData(16, 48, false, 0, 0.25f, 0)] // odd block count, pad value
    [InlineData(32, 64, true, 1, 0, 1)]    // Winograd bands
    [InlineData(24, 20, true, 0, 0, 3)]    // Winograd, ic % 16 != 0, lane tail, gelu
    public void ConvPool_MatchesAvx2(int ic, int oc, bool wino, int mode, float padValue, int act)
    {
        var rng = new Random(ic * 13 + oc);
        int h = 22, w = 30;   // pooled 11 x 15: odd tile counts
        float[] x = rng.Array(h * w * ic), wt = rng.Array(oc * ic * 9), b = rng.Array(oc);
        var p = new ConvP
        {
            Pads = [1, 1, 1, 1], Strides = [1, 1], Dilations = [1, 1], Kernel = [3, 3],
            PadMode = (PadMode)mode, PadValue = padValue, Act = (ActivationKind)act,
            Pool = new PoolP { Kernel = [2, 2], Strides = [2, 2], Pads = [0, 0, 0, 0], FuseRelu = act == 0 },
        };
        NhwcConvPlan kp = Nhwc.Pack(wt, [oc, ic, 3, 3], b, 1, ic);
        if (wino)
        {
            Skip.IfNot(Nhwc.WinogradEligible(p, [oc, ic, 3, 3], ic), "Winograd disabled");
            kp.WinoU = Nhwc.PackWinograd(wt, oc, ic);
        }
        float[] fused = Both(dst => Nhwc.ConvPool(x, h, w, ic, dst, h / 2, w / 2, oc, p, kp), h / 2 * (w / 2) * oc);

        // ... and equal to the separate conv -> maxpool on the AVX-512 tier
        var conv = new ConvP
        {
            Pads = p.Pads, Strides = p.Strides, Dilations = p.Dilations, Kernel = p.Kernel,
            PadMode = p.PadMode, PadValue = p.PadValue, Act = p.Act,
        };
        float[] full = new float[h * w * oc], pooled = new float[h / 2 * (w / 2) * oc];
        Nhwc.Conv(x, h, w, ic, full, h, w, oc, conv, kp, default);
        Nhwc.MaxPool(full, h, w, oc, pooled, h / 2, w / 2, p.Pool);
        Assert.True(pooled.AsSpan().SequenceEqual(fused), "fused conv+pool differs from conv -> maxpool");
    }

    [SkippableTheory]
    [InlineData(3, 16, 1, 2)]   // gc tail: skip = relu(conv3x3(3 -> 16)) recomputed, sigmoid
    [InlineData(6, 32, 1, 2)]   // dr tail
    [InlineData(8, 24, 0, 0)]   // c % 16 != 0
    public void UpHead_SkipConv_MatchesAvx2(int skipC, int c, int skipAct, int act)
    {
        var rng = new Random(skipC * 31 + c);
        int h = 11, w = 15, oh = 22, ow = 30, oc = 3;
        float[] x = rng.Array(h * w * c), sx = rng.Array(oh * ow * skipC);
        float[] cw = rng.Array(c * skipC * 9), cb = rng.Array(c), wt = rng.Array(oc * c), b = rng.Array(oc);
        var sc = new ConvP
        {
            Pads = [1, 1, 1, 1], Strides = [1, 1], Dilations = [1, 1], Kernel = [3, 3],
            PadMode = PadMode.Reflect, Act = (ActivationKind)skipAct,
            KernelPlan = Nhwc.Pack(cw, [c, skipC, 3, 3], cb, 1, skipC),
        };
        UpHeadPlan kp = Nhwc.PackUpHead(wt, oc, c, b);
        var p = new UpHeadP { Scales = [1, 1, 2, 2], ResizeRelu = true, SkipRelu = true, Act = (ActivationKind)act, SkipConv = sc };
        float[] fused = Both(dst => Nhwc.UpHead(x, h, w, c, sx, dst, oh, ow, 2, 2, p, kp, oh, ow, skipC), oh * ow * oc);

        // ... and equal to UpHead over the stored skip tensor on the AVX-512 tier
        float[] skip = new float[oh * ow * c], want = new float[oh * ow * oc];
        Nhwc.Conv(sx, oh, ow, skipC, skip, oh, ow, c, sc, (NhwcConvPlan)sc.KernelPlan!, default);
        var plain = new UpHeadP { Scales = p.Scales, ResizeRelu = true, SkipRelu = true, Act = p.Act };
        Nhwc.UpHead(x, h, w, c, skip, want, oh, ow, 2, 2, plain, kp);
        Assert.True(want.AsSpan().SequenceEqual(fused), "skip recompute differs from the stored skip");
    }

    [SkippableTheory]
    [InlineData("gc", 3)]
    [InlineData("dr", 6)]
    public void MemFusions_Avx512_MatchUnfusedAvx2(string net, int cin)
    {
        // whole net: memory-fused plan on the AVX-512 tier vs unfused plan on the AVX2 tier, bit for bit
        Skip.IfNot(s_tier && Avx2.IsSupported && Fma.IsSupported, "AVX-512 tier not active");
        var m = Sdcb.GcdrNet.Onnx.OnnxModel.Load(Path.Combine(AppContext.BaseDirectory, "testdata", "models",
            net == "gc" ? "gcdrnet_gcnet_sim.onnx" : "gcdrnet_drnet_sim.onnx"));
        int[] shape = [1, cin, 96, 128];
        var rnd = new Random(7);
        float[] input = Enumerable.Range(0, cin * 96 * 128).Select(_ => (float)rnd.NextDouble()).ToArray();
        float[] Run(bool fuse, bool avx512)
        {
            ExecPlan p = GraphCompiler.Compile(m, shape, new CompileOptions(Nhwc: true, PhysicalIo: true, MemFuse: fuse));
            try
            {
                Nhwc.UseAvx512 = avx512;
                using var s = new InferenceSession(p);
                input.CopyTo(s.Input); s.Run();
                return s.Output.ToArray();
            }
            finally { Nhwc.UseAvx512 = s_tier; }
        }
        float[] fused512 = Run(true, true), unfused2 = Run(false, false);
        Assert.True(fused512.AsSpan().SequenceEqual(unfused2), $"{net}: fused AVX-512 differs from unfused AVX2");
    }
}
