using System.Runtime.Intrinsics.Arm;
using Sdcb.SimdGcdrNet.Engine;
using Sdcb.SimdGcdrNet.Kernels;

namespace Sdcb.SimdGcdrNet.Tests;

/// <summary>
/// NEON tier vs the generic Vector&lt;float&gt; kernels in-process (<see cref="Nhwc.UseNeon"/> toggled).
/// The NEON kernels fuse multiply-adds (and reassociate the bilinear lerps), so results are compared
/// with a small tolerance; the memory fusions must stay bit-identical within the NEON tier.
/// Skipped off AArch64 (or with DOTNET_EnableHWIntrinsic=0).
/// </summary>
[Collection(TierToggle.Name)]
public class NeonKernelTests
{
    private static readonly bool s_tier = Nhwc.UseNeon;

    private static void SkipUnlessNeon() =>
        Skip.IfNot(s_tier && AdvSimd.Arm64.IsSupported, "NEON tier not active");

    private static float[] Both(Action<float[]> run, int n, float tol = 2e-5f)
    {
        SkipUnlessNeon();
        float[] vec = new float[n], neon = new float[n];
        try
        {
            Nhwc.UseNeon = false;
            run(vec);
            Nhwc.UseNeon = true;
            run(neon);
        }
        finally { Nhwc.UseNeon = s_tier; }
        // relative to the output's RMS: activations near zero still carry the pre-activation rounding
        double ss = 0;
        foreach (float v in vec) ss += (double)v * v;
        float scale = MathF.Max(1f, (float)Math.Sqrt(ss / Math.Max(1, n)));
        for (int i = 0; i < n; i++)
            if (!(MathF.Abs(vec[i] - neon[i]) <= tol * MathF.Max(scale, MathF.Abs(vec[i]))))
                Assert.Fail($"element {i}: vector {vec[i]:R} vs neon {neon[i]:R}");
        return neon;
    }

    private static ConvP Conv3(int k, int stride, int mode, float padValue, int act) => new()
    {
        Pads = [k / 2, k / 2, k / 2, k / 2], Strides = [stride, stride], Dilations = [1, 1], Kernel = [k, k],
        PadMode = (PadMode)mode, PadValue = padValue, Act = (ActivationKind)act,
    };

    [SkippableTheory]
    [InlineData(3, 16, 3, 1, 37, 23, 1, 0f, false, 0)]   // gc stem: run of 9 (2 by-element groups + 1 broadcast)
    [InlineData(6, 32, 3, 1, 29, 41, 1, 0f, false, 1)]   // dr stem: run of 18
    [InlineData(16, 48, 3, 1, 11, 13, 0, 0.25f, true, 1)] // odd block count, pad value, residual
    [InlineData(20, 20, 3, 2, 17, 19, 0, 0f, true, 2)]   // stride 2, oc lane tail, sigmoid
    [InlineData(32, 64, 3, 2, 12, 31, 1, 0f, false, 3)]  // stride 2, reflect, gelu
    [InlineData(24, 3, 3, 1, 9, 10, 0, 0f, false, 0)]    // oc=3 (single partial lane block)
    [InlineData(40, 36, 1, 1, 7, 33, 0, 0f, true, 1)]    // 1x1 conv
    [InlineData(5, 32, 5, 1, 14, 9, 1, 0f, false, 3)]    // 5x5, reflect: border runs gathered, gelu
    [InlineData(7, 16, 3, 1, 4, 3, 0, -0.5f, true, 2)]   // tiny image: every run crosses a border
    public void Conv_Direct_MatchesVector(int ic, int oc, int k, int stride, int h, int w, int mode,
        float padValue, bool residual, int act)
    {
        var rng = new Random(ic * 131 + oc * 7 + k);
        int pad = k / 2;
        int oh = (h + 2 * pad - k) / stride + 1, ow = (w + 2 * pad - k) / stride + 1;
        float[] x = rng.Array(h * w * ic), wt = rng.Array(oc * ic * k * k), b = rng.Array(oc);
        float[] res = residual ? rng.Array(oh * ow * oc) : [];
        ConvP p = Conv3(k, stride, mode, padValue, act);
        NhwcConvPlan kp = Nhwc.Pack(wt, [oc, ic, k, k], b, 1, ic);   // no Winograd: direct kernels
        Both(dst => Nhwc.Conv(x, h, w, ic, dst, oh, ow, oc, p, kp, res), oh * ow * oc);
    }

    [SkippableTheory]
    [InlineData(16, 16, 9, 11, 0, 0f, false, 0)]
    [InlineData(24, 20, 13, 6, 1, 0f, true, 1)]
    [InlineData(32, 48, 22, 18, 0, 0.5f, false, 3)]
    [InlineData(64, 16, 17, 23, 1, 0f, true, 2)]
    [InlineData(128, 64, 30, 41, 1, 0f, false, 1)] // several tile chunks per image
    public void Conv_Winograd_MatchesVector(int ic, int oc, int h, int w, int mode, float padValue, bool residual, int act)
    {
        var rng = new Random(ic * 17 + oc);
        float[] x = rng.Array(h * w * ic), wt = rng.Array(oc * ic * 9), b = rng.Array(oc);
        float[] res = residual ? rng.Array(h * w * oc) : [];
        ConvP p = Conv3(3, 1, mode, padValue, act);
        Skip.IfNot(Nhwc.WinogradEligible(p, [oc, ic, 3, 3], ic), "Winograd disabled");
        NhwcConvPlan kp = Nhwc.Pack(wt, [oc, ic, 3, 3], b, 1, ic);
        kp.WinoU = Nhwc.PackWinograd(wt, oc, ic);
        Both(dst => Nhwc.Conv(x, h, w, ic, dst, h, w, oc, p, kp, res), h * w * oc, 2e-4f);
    }

    [SkippableTheory]
    [InlineData(16, 0, 0f, false, 0)]
    [InlineData(20, 1, 0f, true, 3)]    // c % 16 != 0, residual, gelu
    [InlineData(22, 0, 0.5f, false, 2)] // c % 4 != 0: scalar channel tail, sigmoid
    public void Depthwise_MatchesVector(int c, int mode, float padValue, bool residual, int act)
    {
        var rng = new Random(c * 5 + act);
        int h = 13, w = 17;
        float[] x = rng.Array(h * w * c), wt = rng.Array(c * 9), b = rng.Array(c);
        float[] res = residual ? rng.Array(h * w * c) : [];
        ConvP p = Conv3(3, 1, mode, padValue, act);
        p.Group = c;
        NhwcConvPlan kp = Nhwc.Pack(wt, [c, 1, 3, 3], b, c, c);
        Both(dst => Nhwc.Conv(x, h, w, c, dst, h, w, c, p, kp, res), h * w * c);
    }

    [SkippableTheory]
    [InlineData(57, 128, 128, false, 0)]
    [InlineData(23, 160, 48, true, 3)]   // odd block count, transB, gelu
    [InlineData(5, 66, 20, false, 1)]    // fewer rows than a tile, k % 4 != 0, lane tail
    public void Gemm_MatchesVector(int m, int k, int n, bool transB, int act)
    {
        var rng = new Random(m + k + n);
        float[] a = rng.Array(m * k), bm = rng.Array(k * n), bias = rng.Array(n);
        NhwcConvPlan kp = Nhwc.PackGemm(bm, k, n, transB, bias);
        Both(dst => Nhwc.Gemm(a, m, k, dst, n, kp, (ActivationKind)act), m * n);
    }

    [SkippableTheory]
    [InlineData(32, 3, true, true, 2)]
    [InlineData(16, 3, false, true, 0)]
    [InlineData(24, 5, true, false, 1)]   // oc 5: generic path
    [InlineData(20, 4, false, false, 3)]  // oc 4 fills the accumulator, gelu
    public void UpHead_MatchesVector(int c, int oc, bool resizeRelu, bool skipRelu, int act)
    {
        var rng = new Random(c * oc);
        int h = 11, w = 15, oh = 22, ow = 30;
        float[] x = rng.Array(h * w * c), skip = rng.Array(oh * ow * c), wt = rng.Array(oc * c), b = rng.Array(oc);
        UpHeadPlan kp = Nhwc.PackUpHead(wt, oc, c, b);
        var p = new UpHeadP { Scales = [1, 1, 2, 2], ResizeRelu = resizeRelu, SkipRelu = skipRelu, Act = (ActivationKind)act };
        Both(dst => Nhwc.UpHead(x, h, w, c, skip, dst, oh, ow, 2, 2, p, kp), oh * ow * oc);
    }

    [SkippableTheory]
    [InlineData(128, 64)]
    [InlineData(37, 18)]   // c % 4 != 0
    public void LayerNorm_MatchesVector(int c, int rows)
    {
        var rng = new Random(c + rows);
        float[] x = rng.Array(rows * c), wt = rng.Array(c), b = rng.Array(c);
        Both(dst => Ops.LayerNorm(x, [rows, c], wt, b, dst, 1e-5f), rows * c);
    }

    [SkippableTheory]
    [InlineData(3, 16, false, 1, 0f, 1)]    // gc stem
    [InlineData(6, 32, false, 1, 0f, 1)]    // dr stem
    [InlineData(16, 48, false, 0, 0.25f, 0)] // odd block count, pad value
    [InlineData(32, 64, true, 1, 0f, 1)]    // Winograd bands
    [InlineData(24, 20, true, 0, 0f, 3)]    // Winograd, lane tail, gelu
    public void ConvPool_MatchesVector_AndUnfused(int ic, int oc, bool wino, int mode, float padValue, int act)
    {
        var rng = new Random(ic * 13 + oc);
        int h = 22, w = 30;
        float[] x = rng.Array(h * w * ic), wt = rng.Array(oc * ic * 9), b = rng.Array(oc);
        ConvP p = Conv3(3, 1, mode, padValue, act);
        p.Pool = new PoolP { Kernel = [2, 2], Strides = [2, 2], Pads = [0, 0, 0, 0], FuseRelu = act == 0 };
        NhwcConvPlan kp = Nhwc.Pack(wt, [oc, ic, 3, 3], b, 1, ic);
        if (wino)
        {
            Skip.IfNot(Nhwc.WinogradEligible(p, [oc, ic, 3, 3], ic), "Winograd disabled");
            kp.WinoU = Nhwc.PackWinograd(wt, oc, ic);
        }
        float[] fused = Both(dst => Nhwc.ConvPool(x, h, w, ic, dst, h / 2, w / 2, oc, p, kp), h / 2 * (w / 2) * oc, wino ? 2e-4f : 2e-5f);

        // ... and bit-identical to the separate conv -> maxpool on the NEON tier
        ConvP conv = Conv3(3, 1, mode, padValue, act);
        float[] full = new float[h * w * oc], pooled = new float[h / 2 * (w / 2) * oc];
        Nhwc.Conv(x, h, w, ic, full, h, w, oc, conv, kp, default);
        Nhwc.MaxPool(full, h, w, oc, pooled, h / 2, w / 2, p.Pool);
        Assert.True(pooled.AsSpan().SequenceEqual(fused), "fused conv+pool differs from conv -> maxpool");
    }

    [SkippableTheory]
    [InlineData(3, 16, 1, 2)]   // gc tail
    [InlineData(6, 32, 1, 2)]   // dr tail
    [InlineData(8, 24, 0, 0)]
    public void UpHead_SkipConv_MatchesVector_AndStoredSkip(int skipC, int c, int skipAct, int act)
    {
        var rng = new Random(skipC * 31 + c);
        int h = 11, w = 15, oh = 22, ow = 30, oc = 3;
        float[] x = rng.Array(h * w * c), sx = rng.Array(oh * ow * skipC);
        float[] cw = rng.Array(c * skipC * 9), cb = rng.Array(c), wt = rng.Array(oc * c), b = rng.Array(oc);
        ConvP sc = Conv3(3, 1, (int)PadMode.Reflect, 0f, skipAct);
        sc.KernelPlan = Nhwc.Pack(cw, [c, skipC, 3, 3], cb, 1, skipC);
        UpHeadPlan kp = Nhwc.PackUpHead(wt, oc, c, b);
        var p = new UpHeadP { Scales = [1, 1, 2, 2], ResizeRelu = true, SkipRelu = true, Act = (ActivationKind)act, SkipConv = sc };
        float[] fused = Both(dst => Nhwc.UpHead(x, h, w, c, sx, dst, oh, ow, 2, 2, p, kp, oh, ow, skipC), oh * ow * oc);

        // ... and bit-identical to UpHead over the stored skip tensor on the NEON tier
        float[] skip = new float[oh * ow * c], want = new float[oh * ow * oc];
        Nhwc.Conv(sx, oh, ow, skipC, skip, oh, ow, c, sc, (NhwcConvPlan)sc.KernelPlan!, default);
        var plain = new UpHeadP { Scales = p.Scales, ResizeRelu = true, SkipRelu = true, Act = p.Act };
        Nhwc.UpHead(x, h, w, c, skip, want, oh, ow, 2, 2, plain, kp);
        Assert.True(want.AsSpan().SequenceEqual(fused), "skip recompute differs from the stored skip");
    }

    [SkippableTheory]
    [InlineData("gc", 3)]
    [InlineData("dr", 6)]
    public void MemFusions_Neon_MatchUnfused(string net, int cin)
    {
        // whole net on the NEON tier: memory-fused plan vs unfused plan, bit for bit
        SkipUnlessNeon();
        var m = Sdcb.SimdGcdrNet.Onnx.OnnxModel.Load(Path.Combine(AppContext.BaseDirectory, "testdata", "models",
            net == "gc" ? "gcdrnet_gcnet_sim.onnx" : "gcdrnet_drnet_sim.onnx"));
        int[] shape = [1, cin, 96, 128];
        var rnd = new Random(7);
        float[] input = Enumerable.Range(0, cin * 96 * 128).Select(_ => (float)rnd.NextDouble()).ToArray();
        float[] Run(bool fuse)
        {
            ExecPlan p = GraphCompiler.Compile(m, shape, new CompileOptions(Nhwc: true, PhysicalIo: true, MemFuse: fuse));
            using var s = new InferenceSession(p);
            input.CopyTo(s.Input); s.Run();
            return s.Output.ToArray();
        }
        Assert.True(Run(true).AsSpan().SequenceEqual(Run(false)), $"{net}: fused NEON differs from unfused NEON");
    }
}

/// <summary>Tests that flip a process-wide kernel tier switch must not overlap any other test.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class TierToggle
{
    public const string Name = "TierToggle";
}
