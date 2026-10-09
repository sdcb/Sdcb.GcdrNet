namespace Sdcb.GcdrNet.Engine;

/// <summary>Executable op set — the lowered form the session runs.</summary>
internal enum ExecOp
{
    Conv,            // NCHW conv, optional bias + fused activation
    MaxPool,         // NCHW pooling
    Elementwise,     // Add/Sub/Mul/Div with broadcast (incl. scalar const)
    Relu,
    Sigmoid,
    Erf,             // elementwise erf (normally fused into Gelu)
    Gelu,            // exact-erf GELU, elementwise
    LayerNorm,       // normalize over last dim
    Gemm,            // [.., M, K] x [K, N] + bias[N]
    ResizeBilinear,  // NCHW, half_pixel, per-axis scales
    Pad,             // constant / reflect on trailing dims
    Transpose,       // arbitrary permute
    Slice,           // strided subrange copy
    Concat,          // axis join
    ChannelShift,    // fused shift-MLP: per channel group, shift along H or W (zero-fill)
    Copy,            // plain copy (for Squeeze/Unsqueeze/identity leftovers)
}

internal enum ActivationKind { None, Relu, Sigmoid }

internal enum PadMode { Constant, Reflect }

internal enum EwOp { Add, Sub, Mul, Div }

// ---- per-op parameter records (resolved at compile time) ----

internal sealed class ConvP
{
    public required int[] Pads;      // [top, left, bottom, right] spatial only
    public required int[] Strides;   // [sh, sw]
    public required int[] Dilations; // [dh, dw]
    public required int[] Kernel;    // [kh, kw]
    public int Group = 1;
    public ActivationKind Act;
    public PadMode PadMode = PadMode.Constant;
    public float PadValue;           // constant-mode fill
    public int BiasValue = -1;       // value id of per-channel bias, -1 none
    public int AddValue = -1;        // value id of same-shape residual addend, -1 none
    public object? KernelPlan;       // packed weights / kernel choice, filled at plan build
}

internal sealed class PoolP
{
    public required int[] Kernel;
    public required int[] Strides;
    public required int[] Pads;
    public bool FuseRelu;
}

internal sealed class EwP
{
    public required EwOp Op;
    public ActivationKind Act;
}

internal sealed class LnP
{
    public float Eps;
}

internal sealed class GemmP
{
    public bool TransB;             // B stored [N,K] when true (onnx transB=1)
    public int BiasValue = -1;
}

internal sealed class ResizeP
{
    public required float[] Scales; // per-dim, only [.., sh, sw] used
    public bool FuseRelu;
}

internal sealed class PadP
{
    public required int[] Pads;     // [2*rank] begin..end
    public PadMode Mode = PadMode.Constant;
    public float Value;
}

internal sealed class TransposeP
{
    public required int[] Perm;
}

internal sealed class SliceP
{
    public required int[] Starts;
    public required int[] Ends;
    public required int[] Axes;
    public required int[] Steps;
}

internal sealed class ConcatP
{
    public int Axis;
}

internal sealed class ShiftP
{
    public int Axis;                 // spatial axis index (2 = H, 3 = W)
    public required int[] GroupEnds; // cumulative channel-group end indices (len 5)
    public required int[] Offsets;   // per-group shift (len 5)
}

/// <summary>A resolved tensor value: compile-time constant or activation.</summary>
internal sealed class CValue
{
    public int[] Shape = [];
    public int DType;            // onnx dtype: 1 float, 7 int64
    public long[]? Ints;         // compile-time int content
    public float[]? Floats;      // compile-time float content (weights or small consts)
    public int Buf = -1;         // activation buffer id; >= 0 → runtime tensor
    public string Name = "";
    public bool IsInt => Ints != null;
    public bool IsConst => Ints != null || Floats != null;
    public bool IsActivation => Buf >= 0;
    public long ElementCount => Shape.Aggregate(1L, (a, b) => a * b);
}

/// <summary>A buffer reference into either the activation arena or a const array.</summary>
internal readonly struct FBuf
{
    public readonly float[] Data;
    public readonly int Offset;
    public readonly int Length;
    public FBuf(float[] data, int offset, int length) { Data = data; Offset = offset; Length = length; }
    public Span<float> Span => Data.AsSpan(Offset, Length);
}

/// <summary>One executable step: op + value ids.</summary>
internal sealed class ExecNode
{
    public required ExecOp Op;
    public required int[] In;    // value ids
    public required int[] Out;
    public object? P;
    public int SrcNode = -1;     // onnx node index (diagnostics)
    public string Name = "";
}

/// <summary>Compiled graph ready for a session: exec nodes + buffer plan.</summary>
internal sealed class ExecPlan
{
    public required ExecNode[] Nodes;
    public required CValue[] Values;
    public required int[] BufSize;      // floats per activation buffer
    public required int[] BufOffset;    // arena offset per buffer
    public required int ArenaFloats;
    public required int InputValue;
    public required int OutputValue;
}
