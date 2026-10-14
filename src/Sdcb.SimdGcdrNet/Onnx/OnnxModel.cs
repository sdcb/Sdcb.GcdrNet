namespace Sdcb.SimdGcdrNet.Onnx;

/// <summary>Validated ONNX model: graph nodes, initializers, IO metadata.</summary>
public sealed class OnnxModel
{
    private OnnxModel(int opset, OnnxGraphData graph, Dictionary<string, OnnxTensorData> initializers)
    {
        Opset = opset;
        Graph = graph;
        Initializers = initializers;
    }

    /// <summary>The model's default-domain (<c>ai.onnx</c>) opset version.</summary>
    public int Opset { get; }
    internal OnnxGraphData Graph { get; }
    internal Dictionary<string, OnnxTensorData> Initializers { get; }

    /// <summary>Loads an ONNX model from the file at <paramref name="path"/>.</summary>
    public static OnnxModel Load(string path)
    {
        using FileStream fs = File.OpenRead(path);
        return Load(fs);
    }

    /// <summary>Loads an ONNX model by reading <paramref name="source"/> to its end.</summary>
    public static OnnxModel Load(Stream source) => Build(OnnxProtoReader.Parse(source));

    /// <summary>Parses the model straight from <paramref name="source"/> (pinned for the duration
    /// of the call, no intermediate copy); weights are decoded into model-owned arrays, so the
    /// memory may be released afterwards.</summary>
    public static unsafe OnnxModel Load(ReadOnlySpan<byte> source)
    {
        fixed (byte* p = source)
        {
            using var ums = new UnmanagedMemoryStream(p, source.Length);
            return Load(ums);
        }
    }

    private static OnnxModel Build(OnnxModelData parsed)
    {
        OnnxGraphData g = parsed.Graph!;
        int opset = checked((int)(parsed.Opsets.FirstOrDefault(
            static x => x.Domain.Length == 0 || x.Domain == "ai.onnx")?.Version ?? 0));
        Dictionary<string, OnnxTensorData> init = new(g.Initializers.Count);
        foreach (OnnxTensorData t in g.Initializers) init[t.Name] = t;
        return new OnnxModel(opset, g, init);
    }
}
