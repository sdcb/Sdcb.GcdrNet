using Sdcb.SimdOnnx.Backends.Vulkan;

namespace Sdcb.SimdGcdrNet.Tests;

/// <summary>
/// Marks the process-wide Vulkan device lost, which disturbs every other Vulkan test running in the
/// process: opt-in (SIMDONNX_TEST_DEVICELOSS=1), run it on its own (--filter DeviceLoss).
/// </summary>
public class VulkanDeviceLossTests
{
    private static string GcPath => Path.Combine(Golden.Dir, "models", "gcdrnet_gcnet_sim.onnx");
    private static string DrPath => Path.Combine(Golden.Dir, "models", "gcdrnet_drnet_sim.onnx");

    [SkippableFact]
    public void Enhance_RecoversFromDeviceLoss_Vulkan()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("SIMDONNX_TEST_DEVICELOSS") == "1", "opt-in: SIMDONNX_TEST_DEVICELOSS=1");
        VkBackend before;
        try { before = VkBackend.Default; }
        catch { throw new SkipException("no Vulkan device"); }
        var (_, _, _, _, oh, ow) = Golden.Dims("small");
        byte[] src = Golden.Bgr("small");

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Vulkan);
        byte[] first = new byte[ow * oh * 3], second = new byte[ow * oh * 3];
        enh.Enhance(src, ow, oh, ow * 3, first, ow * 3);

        before.Dev.Lost = true;
        enh.Enhance(src, ow, oh, ow * 3, second, ow * 3);

        Assert.NotSame(before, VkBackend.Default);
        Assert.Equal(first, second);
    }
}
