using Sdcb.SimdGcdrNet.Models;

namespace Sdcb.SimdGcdrNet.Tests;

public class ModelsPackageTests
{
    [Fact]
    public void EmbeddedModels_MatchTestdataFiles()
    {
        foreach (var (open, file) in new (Func<Stream>, string)[]
        {
            (GcdrModels.OpenGcNet, "gcdrnet_gcnet_sim.onnx"),
            (GcdrModels.OpenDrNet, "gcdrnet_drnet_sim.onnx"),
        })
        {
            using Stream s = open();
            using var ms = new MemoryStream();
            s.CopyTo(ms);
            Assert.Equal(File.ReadAllBytes(Path.Combine(Golden.Dir, "models", file)), ms.ToArray());
        }
    }

    [Fact]
    public void CreateEnhancer_MatchesFromFiles()
    {
        var (_, _, _, _, oh, ow) = Golden.Dims("small");
        byte[] src = Golden.Bgr("small");
        using var fromFiles = GcdrEnhancer.FromFiles(
            Path.Combine(Golden.Dir, "models", "gcdrnet_gcnet_sim.onnx"),
            Path.Combine(Golden.Dir, "models", "gcdrnet_drnet_sim.onnx"));
        using var embedded = GcdrModels.CreateEnhancer();
        Assert.Equal(fromFiles.Enhance(src, ow, oh), embedded.Enhance(src, ow, oh));
    }
}
