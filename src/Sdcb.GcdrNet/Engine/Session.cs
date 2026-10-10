using System.Runtime.InteropServices;
using Sdcb.GcdrNet.Kernels;
using Sdcb.GcdrNet.Onnx;

namespace Sdcb.GcdrNet.Engine;

/// <summary>
/// A reusable inference session over a compiled <see cref="ExecPlan"/>.
/// Not thread-safe; create one per thread or serialize externally.
/// </summary>
public sealed class InferenceSession : IDisposable
{
    private readonly ExecPlan _plan;
    private readonly float[] _arena;
    private readonly int[] _offsets;
    private readonly long[] _nodeTicks;
    private readonly int _inputValue, _outputValue;

    public InferenceSession(CompiledModel model) : this(model.Plan) { }

    internal InferenceSession(ExecPlan plan)
    {
        _plan = plan;
        _arena = new float[plan.ArenaFloats];
        _offsets = plan.BufOffset;
        _inputValue = plan.InputValue;
        _outputValue = plan.OutputValue;
        _nodeTicks = new long[plan.Nodes.Length];
        if (plan.Nhwc)
            foreach (ExecNode n in plan.Nodes)
                if (n.P is GemmP gp && gp.KernelPlan is null && plan.Values[n.In[1]].Floats is { } bw)
                {
                    int[] bs = plan.Values[n.In[1]].Shape;
                    int kk = gp.TransB ? bs[1] : bs[0], nn = gp.TransB ? bs[0] : bs[1];
                    ReadOnlySpan<float> gb = gp.BiasValue >= 0 ? plan.Values[gp.BiasValue].Floats : default;
                    Interlocked.CompareExchange(ref gp.KernelPlan, Nhwc.PackGemm(bw, kk, nn, gp.TransB, gb), null);
                }
                else if (n.P is ConvP cp && cp.KernelPlan is null)
                {
                    // CPU-private packed weights; plans are shared, packing is idempotent
                    CValue w = plan.Values[n.In[1]];
                    ReadOnlySpan<float> b = cp.BiasValue >= 0 ? plan.Values[cp.BiasValue].Floats : default;
                    Interlocked.CompareExchange(ref cp.KernelPlan,
                        Nhwc.Pack(w.Floats!, w.Shape, b, cp.Group, plan.Values[n.In[0]].Shape[1]), null);
                }
    }

    /// <summary>Input tensor shape (e.g. [1,3,H,W]).</summary>
    public int[] InputShape => _plan.Values[_inputValue].Shape;
    public int[] OutputShape => _plan.Values[_outputValue].Shape;

    /// <summary>The writable input window inside the arena (NCHW float).</summary>
    public Span<float> Input => GetSpan(_inputValue);
    /// <summary>The output tensor after <see cref="Run"/>.</summary>
    public ReadOnlySpan<float> Output => GetSpan(_outputValue);

    private Span<float> GetSpan(int valueId)
    {
        CValue v = _plan.Values[valueId];
        if (v.Buf >= 0)
            return _arena.AsSpan(_offsets[v.Buf], checked((int)v.ElementCount));
        if (v.Floats != null) return v.Floats;
        throw new InvalidOperationException($"Value '{v.Name}' has no float storage.");
    }

    private ReadOnlySpan<float> GetIn(int valueId)
    {
        if (valueId < 0) return ReadOnlySpan<float>.Empty;
        return GetSpan(valueId);
    }

    /// <summary>Runs the whole graph. Input must already be written to <see cref="Input"/>.</summary>
    public void Run()
    {
        ExecNode[] nodes = _plan.Nodes;
        CValue[] values = _plan.Values;
        for (int i = 0; i < nodes.Length; i++)
        {
            ExecNode n = nodes[i];
            long t0 = s_profile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            try { Exec(n); }
            catch (Exception ex)
            {
                string In(int v) => v < 0 ? "-" : $"{values[v].Name}:{string.Join('x', values[v].Shape)}";
                throw new InvalidOperationException(
                    $"node[{i}] {n.Op} '{n.Name}' in=[{string.Join(", ", n.In.Select(In))}] out=[{string.Join(", ", n.Out.Select(In))}]", ex);
            }
            if (s_profile) _nodeTicks[i] += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
        }
    }

    private static readonly bool s_profile = Environment.GetEnvironmentVariable("GCDRNET_PROF") == "1";

    /// <summary>Per-op-type profile (ticks), populated when GCDRNET_PROF=1.</summary>
    internal Dictionary<ExecOp, TimeSpan> Profile()
    {
        var d = new Dictionary<ExecOp, TimeSpan>();
        for (int i = 0; i < _plan.Nodes.Length; i++)
        {
            ExecOp op = _plan.Nodes[i].Op;
            d.TryGetValue(op, out var cur);
            d[op] = cur + TimeSpan.FromSeconds((double)_nodeTicks[i] / System.Diagnostics.Stopwatch.Frequency);
        }
        return d;
    }

    /// <summary>Per-node profile (node index, op, name, ticks→time), populated when GCDRNET_PROF=1.</summary>
    internal IEnumerable<(int i, ExecNode n, TimeSpan t)> NodeProfile() =>
        _plan.Nodes.Select((n, i) => (i, n, TimeSpan.FromSeconds((double)_nodeTicks[i] / System.Diagnostics.Stopwatch.Frequency)));

    /// <summary>Debug/test: run a single node by index. Arena must be pre-initialized.</summary>
    internal void ExecNode(int i) => Exec(_plan.Nodes[i]);

    internal int[] ShapeOf(int valueId) => _plan.Values[valueId].Shape;

    /// <summary>Debug/test: read an activation value's arena region.</summary>
    internal ReadOnlySpan<float> ValueSpan(int valueId) => GetSpan(valueId);

    private void Exec(ExecNode n)
    {
        CValue[] values = _plan.Values;
        if (_plan.Nhwc && ExecNhwc(n)) return;
        switch (n.Op)
        {
            case ExecOp.Conv:
            {
                var p = (ConvP)n.P!;
                CValue x = values[n.In[0]], w = values[n.In[1]], o = values[n.Out[0]];
                ReadOnlySpan<float> bias = p.BiasValue >= 0 ? GetIn(p.BiasValue) : ReadOnlySpan<float>.Empty;
                ReadOnlySpan<float> add = p.AddValue >= 0 ? GetIn(p.AddValue) : ReadOnlySpan<float>.Empty;
                Ops.Conv(GetIn(n.In[0]), x.Shape, GetIn(n.In[1]), w.Shape, bias,
                    GetSpan(n.Out[0]), o.Shape, p, add);
                break;
            }
            case ExecOp.MaxPool:
            {
                var p = (PoolP)n.P!;
                Ops.MaxPool(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p);
                break;
            }
            case ExecOp.Elementwise:
            {
                var p = (EwP)n.P!;
                Ops.Elementwise(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetIn(n.In[1]), values[n.In[1]].PhysShape,
                    GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p.Op, p.Act);
                break;
            }
            case ExecOp.Relu: Ops.Relu(GetIn(n.In[0]), GetSpan(n.Out[0])); break;
            case ExecOp.Sigmoid: Ops.Sigmoid(GetIn(n.In[0]), GetSpan(n.Out[0])); break;
            case ExecOp.Erf: Ops.Erf(GetIn(n.In[0]), GetSpan(n.Out[0])); break;
            case ExecOp.Gelu: Ops.Gelu(GetIn(n.In[0]), GetSpan(n.Out[0])); break;
            case ExecOp.LayerNorm:
            {
                var p = (LnP)n.P!;
                Ops.LayerNorm(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetIn(n.In[1]), GetIn(n.In[2]), GetSpan(n.Out[0]), p.Eps);
                break;
            }
            case ExecOp.Gemm:
            {
                var p = (GemmP)n.P!;
                ReadOnlySpan<float> bias = p.BiasValue >= 0 ? GetIn(p.BiasValue) : ReadOnlySpan<float>.Empty;
                Ops.Gemm(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetIn(n.In[1]), values[n.In[1]].PhysShape, bias,
                    GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p);
                break;
            }
            case ExecOp.ResizeBilinear:
            {
                var p = (ResizeP)n.P!;
                Ops.ResizeBilinear(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p.Scales,
                    p.FuseRelu ? ActivationKind.Relu : ActivationKind.None);
                break;
            }
            case ExecOp.Pad:
            {
                var p = (PadP)n.P!;
                Ops.Pad(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p);
                break;
            }
            case ExecOp.Transpose:
            {
                var p = (TransposeP)n.P!;
                Ops.Transpose(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p.Perm);
                break;
            }
            case ExecOp.Slice:
            {
                var p = (SliceP)n.P!;
                Ops.Slice(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p);
                break;
            }
            case ExecOp.Concat:
            {
                var p = (ConcatP)n.P!;
                var ins = new FBuf[n.In.Length];
                var shapes = new int[n.In.Length][];
                for (int k = 0; k < n.In.Length; k++)
                {
                    CValue cv = values[n.In[k]];
                    ins[k] = cv.Buf >= 0
                        ? new FBuf(_arena, _offsets[cv.Buf], checked((int)cv.ElementCount))
                        : new FBuf(cv.Floats!, 0, checked((int)cv.ElementCount));
                    shapes[k] = cv.PhysShape;
                }
                Ops.Concat(ins, shapes, GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p.Axis);
                break;
            }
            case ExecOp.ChannelShift:
            {
                var p = (ShiftP)n.P!;
                Ops.ChannelShift(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p);
                break;
            }
            case ExecOp.Copy:
                GetIn(n.In[0]).CopyTo(GetSpan(n.Out[0]));
                break;
            default:
                throw new NotSupportedException($"ExecOp {n.Op}");
        }
    }

    /// <summary>Channels-last kernels for the layout-sensitive ops; false = generic (storage-shape) path.</summary>
    private bool ExecNhwc(ExecNode n)
    {
        CValue[] values = _plan.Values;
        switch (n.Op)
        {
            case ExecOp.Conv:
            {
                var p = (ConvP)n.P!;
                int[] i = values[n.In[0]].PhysShape, o = values[n.Out[0]].PhysShape;
                ReadOnlySpan<float> add = p.AddValue >= 0 ? GetIn(p.AddValue) : ReadOnlySpan<float>.Empty;
                Nhwc.Conv(GetIn(n.In[0]), i[1], i[2], i[3], GetSpan(n.Out[0]), o[1], o[2], o[3],
                    p, (NhwcConvPlan)p.KernelPlan!, add);
                return true;
            }
            case ExecOp.MaxPool:
            {
                int[] i = values[n.In[0]].PhysShape, o = values[n.Out[0]].PhysShape;
                Nhwc.MaxPool(GetIn(n.In[0]), i[1], i[2], i[3], GetSpan(n.Out[0]), o[1], o[2], (PoolP)n.P!);
                return true;
            }
            case ExecOp.ResizeBilinear:
            {
                var p = (ResizeP)n.P!;
                int[] i = values[n.In[0]].PhysShape, o = values[n.Out[0]].PhysShape;
                Nhwc.ResizeBilinear(GetIn(n.In[0]), i[1], i[2], i[3], GetSpan(n.Out[0]), o[1], o[2],
                    p.Scales[2], p.Scales[3], p.FuseRelu ? ActivationKind.Relu : ActivationKind.None);
                return true;
            }
            case ExecOp.Pad:
            {
                int[] i = values[n.In[0]].PhysShape, o = values[n.Out[0]].PhysShape;
                Nhwc.Pad(GetIn(n.In[0]), i[1], i[2], i[3], GetSpan(n.Out[0]), o[1], o[2], (PadP)n.P!);
                return true;
            }
            case ExecOp.Gemm when ((GemmP)n.P!).KernelPlan is NhwcConvPlan gk:
            {
                var p = (GemmP)n.P!;
                int[] a = values[n.In[0]].PhysShape;
                int k = a[^1], m = (int)(values[n.In[0]].ElementCount / k);
                Nhwc.Gemm(GetIn(n.In[0]), m, k, GetSpan(n.Out[0]), values[n.Out[0]].Shape[^1], gk, p.Act);
                return true;
            }
            case ExecOp.ChannelShift:
            {
                int[] i = values[n.In[0]].PhysShape;
                Nhwc.ChannelShift(GetIn(n.In[0]), i[1], i[2], i[3], GetSpan(n.Out[0]), (ShiftP)n.P!);
                return true;
            }
            default:
                return false;
        }
    }

    public void Dispose() { }
}
