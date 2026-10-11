using System.Resources;

namespace Sdcb.GcdrNet.Models;

/// <summary>
/// The GCDRNet models (simplified GC-Net + DR-Net ONNX, ~21 MB) embedded in this assembly.
/// </summary>
public static class GcdrModels
{
    private const string GcNetResource = "gcnet.onnx", DrNetResource = "drnet.onnx";

    /// <summary>Creates an enhancer from the embedded models, streamed straight from the
    /// assembly image (no intermediate copy).</summary>
    public static GcdrEnhancer CreateEnhancer(GcdrBackend backend = GcdrBackend.Cpu)
    {
        using Stream gc = OpenGcNet();
        using Stream dr = OpenDrNet();
        return GcdrEnhancer.FromStreams(gc, dr, backend);
    }

    /// <summary>Opens the embedded GC-Net (shading estimation) ONNX model. Caller disposes.</summary>
    public static Stream OpenGcNet() => Open(GcNetResource);

    /// <summary>Opens the embedded DR-Net (detail restoration) ONNX model. Caller disposes.</summary>
    public static Stream OpenDrNet() => Open(DrNetResource);

    private static Stream Open(string name) =>
        typeof(GcdrModels).Assembly.GetManifestResourceStream(name)
        ?? throw new MissingManifestResourceException(
            $"Embedded model '{name}' not found in {typeof(GcdrModels).Assembly.GetName().Name}.");
}
