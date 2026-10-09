using Sdcb.GcdrNet.Engine;
using Sdcb.GcdrNet.Kernels;
using Sdcb.GcdrNet.Onnx;
using Xunit;

namespace Sdcb.GcdrNet.Tests;

/// <summary>Plan-level fusion assertions, NOFUSE cross-check, CPU pipeline paths.</summary>
public class EngineTests
{
    private static string GcPath => Path.Combine(Golden.ModelDir, "gcdrnet_gcnet_sim.onnx");
    private static string DrPath => Path.Combine(Golden.ModelDir, "gcdrnet_drnet_sim.onnx");

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
