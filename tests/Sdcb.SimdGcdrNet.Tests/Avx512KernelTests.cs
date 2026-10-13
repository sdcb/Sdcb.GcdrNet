using System.Runtime.Intrinsics.X86;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdGcdrNet.Extensions;
using Sdcb.SimdOnnx.Kernels;
using Sdcb.SimdGcdrNet.Kernels;

namespace Sdcb.SimdGcdrNet.Tests;

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
    [InlineData(32, 3, true, true, 2)]
    [InlineData(16, 3, false, true, 0)]
    [InlineData(24, 5, true, false, 1)]   // c % 16 != 0, oc 5
    public void UpHead_MatchesAvx2(int c, int oc, bool resizeRelu, bool skipRelu, int act)
    {
        var rng = new Random(c * oc);
        int h = 11, w = 15, oh = 22, ow = 30;   // ow % 4 != 0: partial pixel group
        float[] x = rng.Array(h * w * c), skip = rng.Array(oh * ow * c), wt = rng.Array(oc * c), b = rng.Array(oc);
        UpHeadPlan kp = GcdrKernels.PackUpHead(wt, oc, c, b);
        var p = new UpHeadP { Scales = [1, 1, 2, 2], ResizeRelu = resizeRelu, SkipRelu = skipRelu, Act = (ActivationKind)act };
        Both(dst => GcdrKernels.UpHead(x, h, w, c, skip, dst, oh, ow, 2, 2, p, kp), oh * ow * oc);
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
        UpHeadPlan kp = GcdrKernels.PackUpHead(wt, oc, c, b);
        var p = new UpHeadP { Scales = [1, 1, 2, 2], ResizeRelu = true, SkipRelu = true, Act = (ActivationKind)act, SkipConv = sc };
        float[] fused = Both(dst => GcdrKernels.UpHead(x, h, w, c, sx, dst, oh, ow, 2, 2, p, kp, oh, ow, skipC), oh * ow * oc);

        // ... and equal to UpHead over the stored skip tensor on the AVX-512 tier
        float[] skip = new float[oh * ow * c], want = new float[oh * ow * oc];
        Nhwc.Conv(sx, oh, ow, skipC, skip, oh, ow, c, sc, (NhwcConvPlan)sc.KernelPlan!, default);
        var plain = new UpHeadP { Scales = p.Scales, ResizeRelu = true, SkipRelu = true, Act = p.Act };
        GcdrKernels.UpHead(x, h, w, c, skip, want, oh, ow, 2, 2, plain, kp);
        Assert.True(want.AsSpan().SequenceEqual(fused), "skip recompute differs from the stored skip");
    }

}
