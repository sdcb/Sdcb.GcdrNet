using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdGcdrNet.Extensions;
using Sdcb.SimdOnnx.Kernels;
using Sdcb.SimdGcdrNet.Kernels;
using Sdcb.SimdOnnx.Onnx;
using Xunit;

namespace Sdcb.SimdGcdrNet.Tests;

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
        ExecPlan dr = GraphCompiler.Compile(OnnxModel.Load(DrPath), [1, 6, 96, 128], CompileOptions.Default with { Extensions = GcdrExtensions.Instance });
        Assert.Equal(8, dr.Nodes.Count(n => n.Ext == ChannelShiftOp.Instance));

        ExecPlan gc = GraphCompiler.Compile(OnnxModel.Load(GcPath), [1, 3, 96, 128], CompileOptions.Default with { Extensions = GcdrExtensions.Instance });
        Assert.True(gc.Nodes.Count(n => n.Ext == ChannelShiftOp.Instance) >= 4,
            "gc plan lost its ChannelShift fusion");
    }

    [Fact]
    public void Fusion_Conv_Gelu_ThroughAlias()
    {
        // the gelu decomposition commutes before its producer transpose, then
        // fuses into the 1x1 conv via the reshape Buf-alias — the plan must
        // contain no Gelu node and the conv must carry the epilogue
        ExecPlan gc = GraphCompiler.Compile(OnnxModel.Load(GcPath), [1, 3, 96, 128], CompileOptions.Default with { Extensions = GcdrExtensions.Instance });
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
                new CompileOptions(Nhwc: true, PhysicalIo: physicalIo, Extensions: GcdrExtensions.Instance));
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
                using var s = new InferenceSession(GraphCompiler.Compile(m, shape, new CompileOptions(nhwc, Extensions: GcdrExtensions.Instance)));
                input.CopyTo(s.Input); s.Run();
                return s.Output.ToArray();
            }
            var (max, rmse, _) = Golden.Diff(Run(true), Run(false));
            Assert.True(max < 1e-5f, $"{Path.GetFileName(path)} nhwc vs nchw max diff {max} rmse {rmse}");
        }
    }

    [Fact]
    public void MemFusions_BitIdentical_AndShrinkArena()
    {
        // conv+maxpool and UpHead skip recompute must be present in CPU NHWC plans, must
        // not change a single bit of the output, and must shrink the arena
        foreach (var (path, c) in new[] { (GcPath, 3), (DrPath, 6) })
        {
            OnnxModel m = OnnxModel.Load(path);
            int[] shape = [1, c, 96, 128];
            var rnd = new Random(7);
            float[] input = Enumerable.Range(0, c * 96 * 128).Select(_ => (float)rnd.NextDouble()).ToArray();
            (float[] y, ExecPlan p) Run(bool fuse)
            {
                ExecPlan p = GraphCompiler.Compile(m, shape, new CompileOptions(Nhwc: true, PhysicalIo: true, MemFuse: fuse, Extensions: GcdrExtensions.Instance));
                using var s = new InferenceSession(p);
                input.CopyTo(s.Input); s.Run();
                return (s.Output.ToArray(), p);
            }
            var (fy, fp) = Run(true);
            var (uy, up) = Run(false);
            string name = Path.GetFileName(path);
            Assert.Equal(3, fp.Nodes.Count(n => n.P is ConvP { Pool: not null }));
            Assert.DoesNotContain(fp.Nodes, n => n.Op == ExecOp.MaxPool);
            Assert.Contains(fp.Nodes, n => n.P is UpHeadP { SkipConv: not null });
            Assert.True(fp.ArenaFloats * 2 < up.ArenaFloats, $"{name}: arena {fp.ArenaFloats} vs unfused {up.ArenaFloats}");
            Assert.True(fy.AsSpan().SequenceEqual(uy), $"{name}: fused output differs");
        }
    }

    [Fact]
    public void MemFusions_GpuPlans_StemOnly()
    {
        // Vulkan plans get the stem conv+pool and the UpHead recompute (small-ic shaders) only
        foreach (var (path, c) in new[] { (GcPath, 3), (DrPath, 6) })
        {
            ExecPlan p = GraphCompiler.Compile(OnnxModel.Load(path), [1, c, 96, 128],
                new CompileOptions(Nhwc: true, PhysicalIo: true, Target: PlanTarget.Vulkan, Extensions: GcdrExtensions.Instance));
            Assert.Single(p.Nodes, n => n.P is ConvP { Pool: not null });
            Assert.Equal(2, p.Nodes.Count(n => n.Op == ExecOp.MaxPool));
            Assert.Contains(p.Nodes, n => n.P is UpHeadP { SkipConv: not null, SkipW: >= 0 });
        }
    }

    [Fact]
    public void NoFuse_MatchesFused()
    {
        float[] input = Golden.F32("small", "input");
        int[] shape = Golden.ShapeOf("small", "input");
        OnnxModel m = OnnxModel.Load(GcPath);

        // per-compile option, not GCDRNET_NOFUSE: the env var would also unfuse graphs that tests
        // running in parallel compile meanwhile (byte-exact Vulkan comparisons then fail)
        float[] Run(bool fuse)
        {
            using var s = new InferenceSession(GraphCompiler.Compile(m, shape,
                CompileOptions.Default with { Extensions = GcdrExtensions.Instance, Fuse = fuse }));
            input.CopyTo(s.Input); s.Run();
            return s.Output.ToArray();
        }

        var (max, rmse, _) = Golden.Diff(Run(fuse: true), Run(fuse: false));
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
    public void ChannelShift_OffsetBeyondWidth_Zeros()
    {
        // |off| >= w must produce a fully zeroed group, not a negative Slice
        float[] src = Enumerable.Repeat(1f, 2 * 4 * 5).ToArray();   // c=2,h=4,w=5
        float[] dst = Enumerable.Repeat(-1f, src.Length).ToArray();
        var p = new ShiftP { Axis = 3, Offsets = [5], GroupEnds = [2] };
        GcdrKernels.ChannelShiftNchw(src, [1, 2, 4, 5], dst, [1, 2, 4, 5], p);
        Assert.All(dst, v => Assert.Equal(0f, v));
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
