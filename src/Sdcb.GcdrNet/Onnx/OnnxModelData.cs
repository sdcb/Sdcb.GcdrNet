namespace Sdcb.GcdrNet.Onnx;

/// <summary>Raw parsed ONNX structures (protobuf level, pre-validation).</summary>
internal sealed class OnnxModelData
{
    public long IrVersion;
    public List<OnnxOpsetData> Opsets { get; } = [];
    public OnnxGraphData? Graph;
}

internal sealed class OnnxOpsetData
{
    public string Domain = "";
    public long Version;
}

internal sealed class OnnxGraphData
{
    public List<OnnxNodeData> Nodes { get; } = [];
    public List<OnnxTensorData> Initializers { get; } = [];
    public List<OnnxValueInfoData> Inputs { get; } = [];
    public List<OnnxValueInfoData> Outputs { get; } = [];
    public List<OnnxValueInfoData> ValueInfo { get; } = [];
}

internal sealed class OnnxNodeData
{
    public List<string> Inputs { get; } = [];
    public List<string> Outputs { get; } = [];
    public string Name = "";
    public string OpType = "";
    public string Domain = "";
    public List<OnnxAttributeData> Attributes { get; } = [];

    public OnnxAttributeData? Attr(string name)
    {
        foreach (OnnxAttributeData a in Attributes)
            if (a.Name == name) return a;
        return null;
    }

    public long IntAttr(string name, long fallback = 0) => Attr(name)?.Int ?? fallback;
    public float FloatAttr(string name, float fallback = 0f) => Attr(name)?.Float ?? fallback;
    public string StringAttr(string name, string fallback = "") =>
        Attr(name)?.String is byte[] b ? System.Text.Encoding.UTF8.GetString(b) : fallback;
    public long[] IntsAttr(string name) => Attr(name)?.Ints.ToArray() ?? [];
}

internal sealed class OnnxAttributeData
{
    public string Name = "";
    public int Type;
    public float Float;
    public long Int;
    public byte[]? String;
    public OnnxTensorData? Tensor;
    public List<float> Floats { get; } = [];
    public List<long> Ints { get; } = [];
    public List<byte[]> Strings { get; } = [];
}

internal sealed class OnnxTensorData
{
    public List<long> Dims { get; } = [];
    public int DataType;
    public List<float> FloatData { get; } = [];
    public List<int> Int32Data { get; } = [];
    public List<byte[]> StringData { get; } = [];
    public List<long> Int64Data { get; } = [];
    public byte[]? RawData;
    public string Name = "";
    public List<double> DoubleData { get; } = [];
    public List<ulong> UInt64Data { get; } = [];

    public const int TypeFloat = 1, TypeUInt8 = 2, TypeInt8 = 3, TypeUInt16 = 4, TypeInt16 = 5,
        TypeInt32 = 6, TypeInt64 = 7, TypeString = 8, TypeBool = 9, TypeFloat16 = 10,
        TypeDouble = 11, TypeUInt32 = 12, TypeUInt64 = 13;

    public int[] DimsArray() => Dims.Select(d => checked((int)d)).ToArray();

    /// <summary>Decoded element count (product of dims; scalar = 1).</summary>
    public long ElementCount()
    {
        long n = 1;
        foreach (long d in Dims) n *= d;
        return n;
    }

    /// <summary>Decodes this tensor to a float array (float/float16/int sources).</summary>
    public float[] AsFloats()
    {
        int count = checked((int)ElementCount());
        if (RawData is byte[] raw)
        {
            switch (DataType)
            {
                case TypeFloat:
                {
                    float[] f = new float[checked(raw.Length / 4)];
                    Buffer.BlockCopy(raw, 0, f, 0, raw.Length);
                    return f;
                }
                case TypeDouble:
                {
                    double[] d = new double[checked(raw.Length / 8)];
                    Buffer.BlockCopy(raw, 0, d, 0, raw.Length);
                    return d.Select(v => (float)v).ToArray();
                }
                case TypeInt32:
                {
                    int[] v = new int[checked(raw.Length / 4)];
                    Buffer.BlockCopy(raw, 0, v, 0, raw.Length);
                    return v.Select(x => (float)x).ToArray();
                }
                case TypeInt64:
                {
                    long[] v = new long[checked(raw.Length / 8)];
                    Buffer.BlockCopy(raw, 0, v, 0, raw.Length);
                    return v.Select(x => (float)x).ToArray();
                }
                default:
                    throw new NotSupportedException($"Tensor '{Name}': raw dtype {DataType} cannot be read as float.");
            }
        }
        if (FloatData.Count > 0) return FloatData.ToArray();
        if (Int32Data.Count > 0) return Int32Data.Select(v => (float)v).ToArray();
        if (Int64Data.Count > 0) return Int64Data.Select(v => (float)v).ToArray();
        if (DoubleData.Count > 0) return DoubleData.Select(v => (float)v).ToArray();
        if (count == 0) return [];
        throw new InvalidDataException($"Tensor '{Name}' has no float-readable data.");
    }

    /// <summary>Decodes this tensor to an int64 array.</summary>
    public long[] AsInt64s()
    {
        if (RawData is byte[] raw)
        {
            switch (DataType)
            {
                case TypeInt64:
                {
                    long[] v = new long[checked(raw.Length / 8)];
                    Buffer.BlockCopy(raw, 0, v, 0, raw.Length);
                    return v;
                }
                case TypeInt32:
                {
                    int[] v = new int[checked(raw.Length / 4)];
                    Buffer.BlockCopy(raw, 0, v, 0, raw.Length);
                    return v.Select(x => (long)x).ToArray();
                }
                case TypeFloat:
                {
                    float[] f = new float[checked(raw.Length / 4)];
                    Buffer.BlockCopy(raw, 0, f, 0, raw.Length);
                    return f.Select(x => (long)x).ToArray();
                }
                default:
                    throw new NotSupportedException($"Tensor '{Name}': raw dtype {DataType} cannot be read as int64.");
            }
        }
        if (Int64Data.Count > 0) return Int64Data.ToArray();
        if (Int32Data.Count > 0) return Int32Data.Select(v => (long)v).ToArray();
        if (FloatData.Count > 0) return FloatData.Select(v => (long)v).ToArray();
        if (ElementCount() == 0) return [];
        throw new InvalidDataException($"Tensor '{Name}' has no int-readable data.");
    }
}

internal sealed class OnnxValueInfoData
{
    public string Name = "";
    public int ElementType;
    public int[]? Shape; // -1 marks a dynamic dimension
}
