namespace Sdcb.SimdGcdrNet.Engine;

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
    Transpose,       // arbitrary permute (physical perm in layout-planned plans)
    Slice,           // strided subrange copy
    Concat,          // axis join
    ChannelShift,    // fused shift-MLP: per channel group, shift along H or W (zero-fill)
    Copy,            // plain copy (for Squeeze/Unsqueeze/identity leftovers)
    UpHead,          // fused NHWC tail: act(W·(resize(x) + [relu](skip)) + b), pointwise W with few outputs
}

internal enum ActivationKind { None, Relu, Sigmoid, Gelu }

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
    // NHWC plans: fused 2x2/2 max pool epilogue — Out[0] is the pooled tensor
    public PoolP? Pool;
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
    public ActivationKind Act;
    public object? KernelPlan;      // backend-private packed B, filled at session build
}

internal sealed class ResizeP
{
    public required float[] Scales; // per-dim, only [.., sh, sw] used
    public bool FuseRelu;           // relu on the resized value (before the optional add)
    // optional same-shape addend in In[1] (NHWC plans): out = [relu](resize(x)) + [relu](In[1])
    public bool AddRelu;            // relu on the addend
}

/// <summary>
/// Fused NHWC network tail (In = [x, skip, w, bias?]):
/// out[o] = act(bias[o] + Σ_c w[o,c]·s[c]) with s = [relu](resize(x)) + [relu](skip).
/// Replaces Resize → (Relu) → Add → 1x1 Conv → (act) so the full-resolution
/// intermediates are never stored.
/// </summary>
internal sealed class UpHeadP
{
    public required float[] Scales;
    public bool ResizeRelu;
    public bool SkipRelu;
    public ActivationKind Act;
    public object? KernelPlan;      // backend-private packed weights
    // NHWC plans: In[1] is the input of this (still scheduled) conv and the skip is
    // recomputed, so the full-resolution skip tensor need not stay live
    public ConvP? SkipConv;
    public int SkipW = -1, SkipB = -1;   // the skip conv's weight / bias value ids
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
    public int Axis;                 // logical spatial axis index (2 = H, 3 = W)
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
    /// <summary>Physical dim order: physical dim i holds logical dim Layout[i]. null = identity.</summary>
    public int[]? Layout;
    /// <summary>Zero-copy view source (reshape/squeeze/unsqueeze/identity); -1 none.</summary>
    public int AliasOf = -1;
    public bool IsInt => Ints != null;
    public bool IsConst => Ints != null || Floats != null;
    public bool IsActivation => Buf >= 0;
    public long ElementCount => Shape.Aggregate(1L, (a, b) => a * b);
    /// <summary>Shape in storage order — what kernels address by.</summary>
    public int[] PhysShape => Layout is null ? Shape : Engine.Layout.Phys(Shape, Layout);
}

/// <summary>A buffer reference into either the activation arena or a const array.</summary>
internal readonly unsafe struct FBuf
{
    public readonly float[]? Data;
    public readonly float* Ptr;
    public readonly int Offset;
    public readonly int Length;
    public FBuf(float[] data, int offset, int length) { Data = data; Offset = offset; Length = length; }
    public FBuf(float* ptr, int length) { Ptr = ptr; Length = length; }
    public Span<float> Span => Data != null ? Data.AsSpan(Offset, Length) : new Span<float>(Ptr, Length);
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
    /// <summary>Rank-4 activations stored channels-last (layout pass ran).</summary>
    public bool Nhwc;
    /// <summary>Input/output stay in plan layout (no boundary relayout); else logical NCHW.</summary>
    public bool PhysicalIo;
}

/// <summary>Plan compilation options.</summary>
/// <param name="Nhwc">run the layout pass: rank-4 activations stored channels-last.</param>
/// <param name="PhysicalIo">keep input/output in plan layout (no boundary relayout); else logical NCHW.</param>
/// <param name="MemFuse">memory fusions (conv+pool, UpHead skip recompute; NHWC only).</param>
/// <param name="Gpu">plan for the Vulkan backend (restricts fusions to what its shaders implement).</param>
internal sealed record CompileOptions(bool Nhwc, bool PhysicalIo = false, bool MemFuse = true, bool Gpu = false)
{
    /// <summary>GCDRNET_NHWC=0 forces the legacy all-NCHW plan (A/B oracle).</summary>
    public static bool DefaultNhwc { get; } = Environment.GetEnvironmentVariable("GCDRNET_NHWC") != "0";
    public static CompileOptions Default => new(DefaultNhwc);
    /// <summary>GCDRNET_NOFUSE_MEM=1 disables the CPU memory fusions (conv+pool, skip recompute).</summary>
    public static bool MemFusions { get; } = Environment.GetEnvironmentVariable("GCDRNET_NOFUSE_MEM") != "1";
}
