namespace Sdcb.GcdrNet.Onnx;

/// <summary>Validated ONNX model: graph nodes, initializers, IO metadata.</summary>
public sealed class OnnxModel
{
    private OnnxModel(int opset, OnnxGraphData graph, Dictionary<string, OnnxTensorData> initializers)
    {
        Opset = opset;
        Graph = graph;
        Initializers = initializers;
    }

    public int Opset { get; }
    internal OnnxGraphData Graph { get; }
    internal Dictionary<string, OnnxTensorData> Initializers { get; }

    public static OnnxModel Load(string path)
    {
        using FileStream fs = File.OpenRead(path);
        return Load(fs);
    }

    public static OnnxModel Load(Stream source) => Build(OnnxProtoReader.Parse(source));

    public static OnnxModel Load(ReadOnlyMemory<byte> source)
    {
        using var ms = new MemoryStream(source.ToArray(), writable: false);
        return Load(ms);
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
