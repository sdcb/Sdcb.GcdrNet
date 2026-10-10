using System.Runtime.Intrinsics.X86;
using Sdcb.GcdrNet.Engine;
using Sdcb.GcdrNet.Kernels;

namespace Sdcb.GcdrNet.Tests;

/// <summary>
/// AVX-512 tier vs the AVX2 tier in-process (<see cref="Nhwc.UseAvx512"/> toggled): the wider
/// kernels keep every output element's reduction order and epilogue math, so results must be
/// bit-identical. Skipped on machines (or JIT settings) without AVX-512.
/// </summary>
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
}
