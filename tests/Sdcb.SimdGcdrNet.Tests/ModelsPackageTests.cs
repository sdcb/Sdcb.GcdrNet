using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sdcb.SimdGcdrNet.Models;

namespace Sdcb.SimdGcdrNet.Tests;

public class ModelsPackageTests
{
    [Fact]
    public void EmbeddedModels_MatchTestdataFiles()
    {
        Assert.True(GcdrModels.GcNet.SequenceEqual(File.ReadAllBytes(Path.Combine(Golden.Dir, "models", "gcdrnet_gcnet_sim.onnx"))));
        Assert.True(GcdrModels.DrNet.SequenceEqual(File.ReadAllBytes(Path.Combine(Golden.Dir, "models", "gcdrnet_drnet_sim.onnx"))));
    }

    [Fact]
    public void EmbeddedModels_AreStableViews()
    {
        // same memory on every access: a view of the assembly image, not a fresh copy
        Assert.True(Unsafe.AreSame(
            ref MemoryMarshal.GetReference(GcdrModels.GcNet), ref MemoryMarshal.GetReference(GcdrModels.GcNet)));
    }

    [Fact]
    public void FromBytes_MatchesFromFiles()
    {
        var (_, _, _, _, oh, ow) = Golden.Dims("small");
        byte[] src = Golden.Bgr("small");
        using var fromFiles = GcdrEnhancer.FromFiles(
            Path.Combine(Golden.Dir, "models", "gcdrnet_gcnet_sim.onnx"),
            Path.Combine(Golden.Dir, "models", "gcdrnet_drnet_sim.onnx"));
        using var embedded = GcdrEnhancer.FromBytes(GcdrModels.GcNet, GcdrModels.DrNet);
        Assert.Equal(fromFiles.Enhance(src, ow, oh), embedded.Enhance(src, ow, oh));
    }
}
