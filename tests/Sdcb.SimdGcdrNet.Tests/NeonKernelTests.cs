using System.Runtime.Intrinsics.Arm;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdGcdrNet.Extensions;
using Sdcb.SimdOnnx.Kernels;
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
    [InlineData(32, 3, true, true, 2)]
    [InlineData(16, 3, false, true, 0)]
    [InlineData(24, 5, true, false, 1)]   // oc 5: generic path
    [InlineData(20, 4, false, false, 3)]  // oc 4 fills the accumulator, gelu
    public void UpHead_MatchesVector(int c, int oc, bool resizeRelu, bool skipRelu, int act)
    {
        var rng = new Random(c * oc);
        int h = 11, w = 15, oh = 22, ow = 30;
        float[] x = rng.Array(h * w * c), skip = rng.Array(oh * ow * c), wt = rng.Array(oc * c), b = rng.Array(oc);
        UpHeadPlan kp = GcdrKernels.PackUpHead(wt, oc, c, b);
        var p = new UpHeadP { Scales = [1, 1, 2, 2], ResizeRelu = resizeRelu, SkipRelu = skipRelu, Act = (ActivationKind)act };
        Both(dst => GcdrKernels.UpHead(x, h, w, c, skip, dst, oh, ow, 2, 2, p, kp), oh * ow * oc);
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
        UpHeadPlan kp = GcdrKernels.PackUpHead(wt, oc, c, b);
        var p = new UpHeadP { Scales = [1, 1, 2, 2], ResizeRelu = true, SkipRelu = true, Act = (ActivationKind)act, SkipConv = sc };
        float[] fused = Both(dst => GcdrKernels.UpHead(x, h, w, c, sx, dst, oh, ow, 2, 2, p, kp, oh, ow, skipC), oh * ow * oc);

        // ... and bit-identical to UpHead over the stored skip tensor on the NEON tier
        float[] skip = new float[oh * ow * c], want = new float[oh * ow * oc];
        Nhwc.Conv(sx, oh, ow, skipC, skip, oh, ow, c, sc, (NhwcConvPlan)sc.KernelPlan!, default);
        var plain = new UpHeadP { Scales = p.Scales, ResizeRelu = true, SkipRelu = true, Act = p.Act };
        GcdrKernels.UpHead(x, h, w, c, skip, want, oh, ow, 2, 2, plain, kp);
        Assert.True(want.AsSpan().SequenceEqual(fused), "skip recompute differs from the stored skip");
    }

}

/// <summary>Tests that flip a process-wide kernel tier switch must not overlap any other test.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class TierToggle
{
    public const string Name = "TierToggle";
}
