using Sdcb.GcdrNet.Backends.Vulkan;
using Sdcb.GcdrNet.Engine;
using Sdcb.GcdrNet.Onnx;
using Xunit;

namespace Sdcb.GcdrNet.Tests;

public class VulkanTests
{
    private static string GcPath => Path.Combine(Golden.Dir, "models", "gcdrnet_gcnet_sim.onnx");
    private static string DrPath => Path.Combine(Golden.Dir, "models", "gcdrnet_drnet_sim.onnx");

    [SkippableTheory]
    [InlineData("small")]
    [InlineData("pad")]
    [InlineData("mid")]
    public void GcNet_MatchesGolden_Vulkan(string tag)
    {
        Skip.IfNot(VkAvailable());
        float[] input = Golden.F32(tag, "input");
        int[] shape = Golden.ShapeOf(tag, "input");
        float[] expect = Golden.F32(tag, "shadow");

        OnnxModel m = OnnxModel.Load(GcPath);
        using var sess = VkBackend.Default.NewSession(m, shape);
        input.CopyTo(sess.Input);
        sess.Run();
        var (max, rmse, _) = Golden.Diff(sess.Output, expect);
        Assert.True(max < 2e-3f, $"gcnet-vk[{tag}] max diff {max} rmse {rmse}");
    }

    [SkippableTheory]
    [InlineData("small")]
    [InlineData("pad")]
    [InlineData("mid")]
    public void DrNet_MatchesGolden_Vulkan(string tag)
    {
        Skip.IfNot(VkAvailable());
        float[] input = Golden.F32(tag, "dr_in");
        int[] shape = Golden.ShapeOf(tag, "dr_in");
        float[] expect = Golden.F32(tag, "output");

        OnnxModel m = OnnxModel.Load(DrPath);
        using var sess = VkBackend.Default.NewSession(m, shape);
        input.CopyTo(sess.Input);
        sess.Run();
        var (max, rmse, _) = Golden.Diff(sess.Output, expect);
        Assert.True(max < 2e-3f, $"drnet-vk[{tag}] max diff {max} rmse {rmse}");
    }

    [SkippableTheory]
    [InlineData("small")]
    [InlineData("pad")]
    public void Pipeline_MatchesGoldenU8_Vulkan(string tag)
    {
        Skip.IfNot(VkAvailable());
        var (_, _, _, _, oh, ow) = Golden.Dims(tag);
        byte[] src = Golden.Bgr(tag);
        byte[] expectU8 = File.ReadAllBytes(Path.Combine(Golden.Dir, $"{tag}_final.u8"));

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Vulkan);
        byte[] dst = new byte[ow * oh * 3];
        enh.Enhance(src, ow, oh, ow * 3, dst, ow * 3);

        int diffs = 0, maxd = 0;
        for (int i = 0; i < dst.Length; i++)
        {
            int d = Math.Abs(dst[i] - expectU8[i]);
            if (d > 0) diffs++;
            if (d > maxd) maxd = d;
        }
        Assert.True(maxd <= 2, $"pipeline-vk[{tag}] max u8 diff {maxd}, changed px {diffs}/{dst.Length}");
    }

    [SkippableTheory]
    [InlineData("small", 13, 17)]
    [InlineData("pad", 8, 5)]
    public void Pipeline_Strided_Vulkan(string tag, int srcPad, int dstPad)
    {
        Skip.IfNot(VkAvailable());
        var (_, _, _, _, oh, ow) = Golden.Dims(tag);
        byte[] tight = Golden.Bgr(tag);
        byte[] expectU8 = File.ReadAllBytes(Path.Combine(Golden.Dir, $"{tag}_final.u8"));

        int ss = ow * 3 + srcPad, ds = ow * 3 + dstPad;
        byte[] src = new byte[ss * oh];
        byte[] dst = new byte[ds * oh];
        for (int y = 0; y < oh; y++)
            Array.Copy(tight, y * ow * 3, src, y * ss, ow * 3);

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Vulkan);
        enh.Enhance(src, ow, oh, ss, dst, ds);

        int maxd = 0;
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow * 3; x++)
                maxd = Math.Max(maxd, Math.Abs(dst[y * ds + x] - expectU8[y * ow * 3 + x]));
        Assert.True(maxd <= 2, $"pipeline-vk-strided[{tag}] max u8 diff {maxd}");
    }

    private static bool VkAvailable()
    {
        try { _ = VkBackend.Default.Dev.DeviceName; return true; }
        catch { return false; }
    }
}
