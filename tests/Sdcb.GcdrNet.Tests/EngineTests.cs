using Sdcb.GcdrNet.Engine;
using Sdcb.GcdrNet.Kernels;
using Sdcb.GcdrNet.Onnx;
using Xunit;

namespace Sdcb.GcdrNet.Tests;

/// <summary>Plan-level fusion assertions, NOFUSE cross-check, CPU pipeline paths.</summary>
public class EngineTests
{
    internal static string GcPath => Path.Combine(Golden.ModelDir, "gcdrnet_gcnet_sim.onnx");
    internal static string DrPath => Path.Combine(Golden.ModelDir, "gcdrnet_drnet_sim.onnx");

    [Fact]
    public void Fusion_ChannelShift_And_ResidualAdd()
    {
        // fused plan must contain the ChannelShift nodes and fused conv residual adds —
        // if a model export change breaks the FuseShift pattern, this fails loudly
        ExecPlan dr = GraphCompiler.Compile(OnnxModel.Load(DrPath), [1, 6, 96, 128]);
        Assert.Equal(8, dr.Nodes.Count(n => n.Op == ExecOp.ChannelShift));

        ExecPlan gc = GraphCompiler.Compile(OnnxModel.Load(GcPath), [1, 3, 96, 128]);
        Assert.True(gc.Nodes.Count(n => n.Op == ExecOp.ChannelShift) >= 4,
            "gc plan lost its ChannelShift fusion");
    }

    [Fact]
    public void Fusion_Conv_Gelu_ThroughAlias()
    {
        // the gelu decomposition commutes before its producer transpose, then
        // fuses into the 1x1 conv via the reshape Buf-alias — the plan must
        // contain no Gelu node and the conv must carry the epilogue
        ExecPlan gc = GraphCompiler.Compile(OnnxModel.Load(GcPath), [1, 3, 96, 128]);
        Assert.DoesNotContain(gc.Nodes, n => n.Op == ExecOp.Gelu);
        Assert.True(gc.Nodes.Count(n => n.Op == ExecOp.Conv &&
                ((ConvP)n.P!).Act == ActivationKind.Gelu) >= 2,
            "gc plan lost the Conv+Gelu fusion");
    }

    [Theory]
    [InlineData(false, 2)]   // logical NCHW I/O: one boundary relayout in, one out
    [InlineData(true, 0)]    // physical I/O (fused pipelines): no transposes at all
    public void Nhwc_TransposesDissolve(bool physicalIo, int expected)
    {
        // all 32 model transposes per net ([0,2,1], [0,3,1,2]) must dissolve into
        // zero-copy layout views; acceptance bound is <= 8
        foreach (var (path, c) in new[] { (GcPath, 3), (DrPath, 6) })
        {
            ExecPlan p = GraphCompiler.Compile(OnnxModel.Load(path), [1, c, 96, 128],
                new CompileOptions(Nhwc: true, PhysicalIo: physicalIo));
            int n = p.Nodes.Count(m => m.Op == ExecOp.Transpose);
            Assert.True(n <= 8, $"{Path.GetFileName(path)}: {n} transposes");
            Assert.Equal(expected, n);
        }
    }

    [Fact]
    public void Nhwc_MatchesNchw()
    {
        // layout-planned CPU plan vs the legacy all-NCHW plan, same input
        foreach (var (path, tag, name) in new[] { (GcPath, "pad", "input"), (DrPath, "pad", "dr_in") })
        {
            float[] input = Golden.F32(tag, name);
            int[] shape = Golden.ShapeOf(tag, name);
            OnnxModel m = OnnxModel.Load(path);
            float[] Run(bool nhwc)
            {
                using var s = new InferenceSession(GraphCompiler.Compile(m, shape, new CompileOptions(nhwc)));
                input.CopyTo(s.Input); s.Run();
                return s.Output.ToArray();
            }
            var (max, rmse, _) = Golden.Diff(Run(true), Run(false));
            Assert.True(max < 1e-5f, $"{Path.GetFileName(path)} nhwc vs nchw max diff {max} rmse {rmse}");
        }
    }

    [Fact]
    public void NoFuse_MatchesFused()
    {
        float[] input = Golden.F32("small", "input");
        int[] shape = Golden.ShapeOf("small", "input");
        OnnxModel m = OnnxModel.Load(GcPath);

        float[] Fused()
        {
            Environment.SetEnvironmentVariable("GCDRNET_NOFUSE", null);
            using var s = new InferenceSession(GraphCompiler.Compile(m, shape));
            input.CopyTo(s.Input); s.Run();
            return s.Output.ToArray();
        }
        float[] Unfused()
        {
            Environment.SetEnvironmentVariable("GCDRNET_NOFUSE", "1");
            try
            {
                using var s = new InferenceSession(GraphCompiler.Compile(m, shape));
                input.CopyTo(s.Input); s.Run();
                return s.Output.ToArray();
            }
            finally { Environment.SetEnvironmentVariable("GCDRNET_NOFUSE", null); }
        }

        var (max, rmse, _) = Golden.Diff(Fused(), Unfused());
        Assert.True(max < 1e-5f, $"fused vs unfused max diff {max} rmse {rmse}");
    }

    [Theory]
    [InlineData("small")]
    [InlineData("pad")]
    [InlineData("mid")]
    public void DivConcat_MatchesOracle_Float(string tag)
    {
        // stage-2 math verified at float precision, not just through u8 output:
        // our GC session on the oracle input -> our own div+concat -> oracle dr_in
        float[] x = Golden.F32(tag, "input");
        int[] shape = Golden.ShapeOf(tag, "input");
        float[] expect = Golden.F32(tag, "dr_in");

        OnnxModel m = OnnxModel.Load(GcPath);
        CompiledModel cm = CompiledModel.Compile(m, shape);
        using var sess = cm.CreateSession();
        x.CopyTo(sess.Input);
        sess.Run();
        float[] shadow = sess.Output.ToArray();

        int hw = shape[2] * shape[3];
        float[] drIn = new float[6 * hw];
        Array.Copy(x, drIn, 3 * hw);
        Kernels.PixelOps.DivClampInto(x, shadow, drIn.AsSpan(3 * hw));

        var (max, rmse, _) = Golden.Diff(drIn, expect);
        Assert.True(max < 1e-3f, $"dr_in[{tag}] max diff {max} rmse {rmse}");
    }

    [Fact]
    public void Transpose_NonIdentityLeadingPerm_Correct()
    {
        // perm [1,0,3,2] must NOT take the last-two-dim fast path (would be wrong);
        // it should fall through to the generic permute and still be correct
        var rng = new Random(5);
        int d0 = 2, d1 = 3, d2 = 4, d3 = 5;
        float[] src = rng.Array(d0 * d1 * d2 * d3);
        float[] got = new float[src.Length];
        Ops.Transpose(src, [d0, d1, d2, d3], got, [d1, d0, d3, d2], [1, 0, 3, 2]);
        for (int i0 = 0; i0 < d0; i0++)
            for (int i1 = 0; i1 < d1; i1++)
                for (int i2 = 0; i2 < d2; i2++)
                    for (int i3 = 0; i3 < d3; i3++)
                        Assert.Equal(
                            src[((i0 * d1 + i1) * d2 + i2) * d3 + i3],
                            got[((i1 * d0 + i0) * d3 + i3) * d2 + i2]);
    }

    [Fact]
    public void ChannelShift_OffsetBeyondWidth_Zeros()
    {
        // |off| >= w must produce a fully zeroed group, not a negative Slice
        float[] src = Enumerable.Repeat(1f, 2 * 4 * 5).ToArray();   // c=2,h=4,w=5
        float[] dst = Enumerable.Repeat(-1f, src.Length).ToArray();
        var p = new ShiftP { Axis = 3, Offsets = [5], GroupEnds = [2] };
        Ops.ChannelShift(src, [1, 2, 4, 5], dst, [1, 2, 4, 5], p);
        Assert.All(dst, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Conv_GroupedNonDepthwise_MatchesReference()
    {
        // group=2 with ic=4 (not depthwise) routes to the scalar path — check vs manual
        var rng = new Random(11);
        int ic = 4, oc = 4, ih = 5, iw = 6, g = 2;
        float[] x = rng.Array(ic * ih * iw);
        float[] w = rng.Array(oc * (ic / g) * 9);
        float[] dst = new float[oc * ih * iw];
        var p = new ConvP { Pads = [1, 1, 1, 1], Strides = [1, 1], Dilations = [1, 1], Kernel = [3, 3], Group = g };
        Ops.Conv(x, [1, ic, ih, iw], w, [oc, ic / g, 3, 3], ReadOnlySpan<float>.Empty,
            dst, [1, oc, ih, iw], p);

        for (int o = 0; o < oc; o++)
            for (int y = 0; y < ih; y++)
                for (int xx = 0; xx < iw; xx++)
                {
                    float acc = 0;
                    int gi = o / (oc / g) * (ic / g);
                    for (int c = 0; c < ic / g; c++)
                        for (int ky = 0; ky < 3; ky++)
                            for (int kx = 0; kx < 3; kx++)
                            {
                                int sy = y + ky - 1, sx = xx + kx - 1;
                                if ((uint)sy < (uint)ih && (uint)sx < (uint)iw)
                                    acc += x[(gi + c) * ih * iw + sy * iw + sx] *
                                           w[((o * (ic / g) + c) * 3 + ky) * 3 + kx];
                            }
                    Assert.Equal(acc, dst[o * ih * iw + y * iw + xx], 5);
                }
    }

    [Fact]
    public void Conv_ResidualAdd_MatchesManual()
    {
        // unit-level check of the Conv+Add epilogue (the fused path has no model coverage)
        var rng = new Random(7);
        int ic = 4, oc = 8, ih = 9, iw = 11;
        float[] x = rng.Array(ic * ih * iw);
        float[] w = rng.Array(oc * ic * 9);
        float[] b = rng.Array(oc);
        float[] res = rng.Array(oc * ih * iw);
        float[] want = new float[oc * ih * iw], got = new float[oc * ih * iw];

        var p = new ConvP { Pads = [1, 1, 1, 1], Strides = [1, 1], Dilations = [1, 1], Kernel = [3, 3] };
        Ops.Conv(x, [1, ic, ih, iw], w, [oc, ic, 3, 3], b, want, [1, oc, ih, iw], p);
        for (int i = 0; i < want.Length; i++) want[i] += res[i];

        Ops.Conv(x, [1, ic, ih, iw], w, [oc, ic, 3, 3], b, got, [1, oc, ih, iw], p, res);
        var (max, _, _) = Golden.Diff(want, got);
        Assert.True(max < 1e-4f, $"conv+add max diff {max}");
    }

    [SkippableTheory]
    [InlineData(16, 16, 9, 11, 0, 0f, false, 0)]   // constant pad
    [InlineData(24, 20, 13, 6, 1, 0f, true, 1)]    // reflect, residual, relu, oc lane tail, odd tiles
    [InlineData(32, 48, 4, 4, 0, 0.5f, false, 3)]  // nonzero pad value, gelu
    [InlineData(64, 16, 17, 23, 1, 0f, true, 0)]
    public void Conv_Winograd_MatchesDirect(int ic, int oc, int h, int w, int mode, float padValue,
        bool residual, int act)
    {
        var rng = new Random(ic * 31 + oc);
        float[] x = rng.Array(h * w * ic);           // NHWC
        float[] wt = rng.Array(oc * ic * 9);
        float[] b = rng.Array(oc);
        float[] res = residual ? rng.Array(h * w * oc) : [];
        var p = new ConvP
        {
            Pads = [1, 1, 1, 1], Strides = [1, 1], Dilations = [1, 1], Kernel = [3, 3],
            PadMode = (PadMode)mode, PadValue = padValue, Act = (ActivationKind)act,
        };
        Skip.IfNot(Nhwc.WinogradEligible(p, [oc, ic, 3, 3], ic), "Winograd disabled or ISA tier without it");
        NhwcConvPlan direct = Nhwc.Pack(wt, [oc, ic, 3, 3], b, 1, ic);
        NhwcConvPlan wino = Nhwc.Pack(wt, [oc, ic, 3, 3], b, 1, ic);
        wino.WinoU = Nhwc.PackWinograd(wt, oc, ic);
        float[] want = new float[h * w * oc], got = new float[h * w * oc];
        Nhwc.Conv(x, h, w, ic, want, h, w, oc, p, direct, res);
        Nhwc.Conv(x, h, w, ic, got, h, w, oc, p, wino, res);
        var (max, _, _) = Golden.Diff(want, got);
        float scale = want.Max(MathF.Abs);
        Assert.True(max < 1e-4f * scale, $"winograd max diff {max} (|out| max {scale})");
    }

    [Fact]
    public void Pipeline_SizeSwitching_MatchesFresh_Cpu()
    {
        // one size-agnostic worker reuses (dirty, larger) arenas across sizes and the
        // per-size plans share packed weights: every size must match a fresh enhancer
        string[] tags = ["mid", "small", "pad", "mid", "small"];
        var fresh = new Dictionary<string, byte[]>();
        foreach (string tag in tags.Distinct())
        {
            var (_, _, _, _, oh, ow) = Golden.Dims(tag);
            using var e = GcdrEnhancer.FromFiles(GcPath, DrPath);
            fresh[tag] = e.Enhance(Golden.Bgr(tag), ow, oh);
        }

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath);
        foreach (string tag in tags)
        {
            var (_, _, _, _, oh, ow) = Golden.Dims(tag);
            byte[] dst = enh.Enhance(Golden.Bgr(tag), ow, oh);
            Assert.True(dst.AsSpan().SequenceEqual(fresh[tag]), $"size-switch[{tag}] differs from a fresh worker");
        }
        Assert.Equal(1, enh.PoolStats.workers);
    }

    [Theory]
    [InlineData("small", 13, 17)]
    [InlineData("pad", 8, 5)]
    public void Pipeline_Strided_Cpu(string tag, int srcPad, int dstPad)
    {
        var (_, _, _, _, oh, ow) = Golden.Dims(tag);
        byte[] tight = Golden.Bgr(tag);
        byte[] expectU8 = Golden.U8(tag);

        int ss = ow * 3 + srcPad, ds = ow * 3 + dstPad;
        byte[] src = new byte[ss * oh], dst = new byte[ds * oh];
        for (int y = 0; y < oh; y++)
            Array.Copy(tight, y * ow * 3, src, y * ss, ow * 3);

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Cpu);
        enh.Enhance(src, ow, oh, ss, dst, ds);

        int maxd = 0;
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow * 3; x++)
                maxd = Math.Max(maxd, Math.Abs(dst[y * ds + x] - expectU8[y * ow * 3 + x]));
        Assert.True(maxd <= 2, $"pipeline-cpu-strided[{tag}] max u8 diff {maxd}");
    }

    [Fact]
    public void Pipeline_Mid_U8()
    {
        var (_, _, _, _, oh, ow) = Golden.Dims("mid");
        byte[] src = Golden.Bgr("mid");
        byte[] expectU8 = Golden.U8("mid");

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Cpu);
        byte[] dst = enh.Enhance(src, ow, oh);

        int maxd = 0;
        for (int i = 0; i < dst.Length; i++) maxd = Math.Max(maxd, Math.Abs(dst[i] - expectU8[i]));
        Assert.True(maxd <= 2, $"pipeline-mid max u8 diff {maxd}");
    }

    [Fact]
    public void Concurrent_Enhance_SameSize()
    {
        // the per-size session pool must return *distinct* sessions to concurrent callers
        var (_, _, _, _, oh, ow) = Golden.Dims("small");
        byte[] src = Golden.Bgr("small");
        byte[] expectU8 = Golden.U8("small");

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Cpu);
        int[] maxDiffs = new int[8];
        Parallel.For(0, 8, t =>
        {
            byte[] dst = enh.Enhance(src, ow, oh);
            int maxd = 0;
            for (int i = 0; i < dst.Length; i++) maxd = Math.Max(maxd, Math.Abs(dst[i] - expectU8[i]));
            maxDiffs[t] = maxd;
        });
        Assert.All(maxDiffs, d => Assert.True(d <= 2, $"concurrent enhance diff {d}"));
    }

    [Fact]
    public void Pipeline_MinSize_32x32()
    {
        int w = 32, h = 32;
        byte[] src = new byte[w * h * 3];
        var rng = new Random(3);
        rng.NextBytes(src);

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Cpu);
        byte[] dst = enh.Enhance(src, w, h);
        Assert.Equal(w * h * 3, dst.Length);
        Assert.All(dst, v => Assert.InRange(v, (byte)0, (byte)255));
    }
}

internal static class RngExt
{
    public static float[] Array(this Random r, int n)
        => Enumerable.Range(0, n).Select(_ => (float)r.NextDouble() * 2f - 1f).ToArray();
}
