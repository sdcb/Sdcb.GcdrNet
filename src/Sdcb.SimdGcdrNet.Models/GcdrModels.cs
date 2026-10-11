using System.Resources;

namespace Sdcb.SimdGcdrNet.Models;

/// <summary>
/// The GCDRNet models (simplified GC-Net + DR-Net ONNX, ~21 MB) embedded in this assembly.
/// This package has no dependencies, so it is versioned independently of the engine:
/// <code>
/// using GcdrEnhancer enh = GcdrEnhancer.FromStreams(GcdrModels.OpenGcNet, GcdrModels.OpenDrNet, GcdrBackend.Auto);
/// </code>
/// </summary>
public static class GcdrModels
{
    private const string GcNetResource = "gcnet.onnx", DrNetResource = "drnet.onnx";

    /// <summary>Opens the embedded GC-Net (shading estimation) ONNX model. Caller disposes.</summary>
    public static Stream OpenGcNet() => Open(GcNetResource);

    /// <summary>Opens the embedded DR-Net (detail restoration) ONNX model. Caller disposes.</summary>
    public static Stream OpenDrNet() => Open(DrNetResource);

    private static Stream Open(string name) =>
        typeof(GcdrModels).Assembly.GetManifestResourceStream(name)
        ?? throw new MissingManifestResourceException(
            $"Embedded model '{name}' not found in {typeof(GcdrModels).Assembly.GetName().Name}.");
}
