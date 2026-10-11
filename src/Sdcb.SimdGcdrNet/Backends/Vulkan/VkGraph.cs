using System.Runtime.InteropServices;
using Sdcb.SimdGcdrNet.Engine;

namespace Sdcb.SimdGcdrNet.Backends.Vulkan;

/// <summary>Binding slots resolved per-session: which underlying buffer.</summary>
internal enum VkSlot : int { Arena = 0, Consts = 1, Dummy = 2, StageIn = 3, StageOut = 4, Direct = 5 }

internal readonly struct VkBind
{
    public readonly VkSlot Slot;
    public readonly VkBuffer? Buf;   // when Slot == Direct
    public readonly ulong Offset, Size;
    public VkBind(VkSlot s, ulong off, ulong size) { Slot = s; Offset = off; Size = size; }
    public VkBind(VkBuffer b, ulong off, ulong size) { Slot = VkSlot.Direct; Buf = b; Offset = off; Size = size; }
}

/// <summary>One recorded compute dispatch.</summary>
internal sealed class VkDispatch
{
    public required VkPipeline Pipe;
    public required VkBind[] Binds;
    public required byte[] Pc;
    public uint Gx, Gy = 1, Gz = 1;
    public string Name = "";
}

/// <summary>
/// Per-(device, ExecPlan) half of the Vulkan backend: backend-private constant
/// packing/upload and the node list compiled to <see cref="VkDispatch"/> records.
/// Sessions own the activation arena and resolve slot ids to live buffers.
///
/// Plans are layout-planned (NHWC, see <see cref="LayoutPlanner"/>); every
/// addressing decision reads <see cref="CValue.PhysShape"/>. With <see cref="F16"/>
/// activations and weights are stored as float16 (16-bit storage only — all
/// shader math, reductions included, is fp32); conv bias and LayerNorm params
/// stay fp32. Const bytes therefore differ per backend and per precision.
/// </summary>
internal sealed class VkGraphModel : IDisposable
{
    private readonly VkDevice _dev;
    private readonly ExecPlan _plan;
    private int _refs = 1;  // the graph cache holds the first reference
    private readonly VkConstStore _store;
    internal readonly VkBuffer Consts;       // all const tensors, backend-packed (shared store)
    private readonly Dictionary<(int v, CK k), ulong> _constOff = new();
    internal readonly VkBuffer Dummy;        // 64B zero buffer for unused bindings
    internal readonly VkDispatch[] Dispatches;
    internal readonly ulong ConstBytes;
    /// <summary>Activations + weights stored as float16 (fp32 math in every shader).</summary>
    internal readonly bool F16;
    /// <summary>Bytes per stored activation element.</summary>
    internal int Es => F16 ? 2 : 4;

    /// <summary>How a const is laid out on the device (decided by its consumer).</summary>
    private enum CK { Raw, F32, ConvW, DwW, ConvWT, GemmNK }

    /// <param name="constKey">Sharing key for the const store (the source model): graphs of the
    /// same model at other sizes reuse its uploaded weights. Null = private store.</param>
    internal VkGraphModel(VkDevice dev, ExecPlan plan, bool f16, object? constKey = null)
    {
        _dev = dev;
        _plan = plan;
        F16 = f16;
        if (!plan.Nhwc) throw new NotSupportedException("Vulkan backend requires a layout-planned (NHWC) plan.");

        // ---- consts: per-consumer packing ----
        var need = new List<(int v, CK k)>();
        void Need(int v, CK k)
        {
            if (v < 0 || plan.Values[v].Floats is null) return;
            if (!need.Contains((v, k))) need.Add((v, k));
        }
        foreach (ExecNode n in plan.Nodes)
        {
            switch (n.P)
            {
                case ConvP cp:
                    Need(n.In[0], CK.Raw);
                    Need(n.In[1], cp.Group > 1 ? CK.DwW : UseCoopConv(dev, f16, plan, n, cp) ? CK.ConvWT : CK.ConvW);
                    Need(cp.BiasValue, CK.F32);
                    Need(cp.AddValue, CK.Raw);
                    break;
                case GemmP gp:
                    Need(n.In[0], CK.Raw); Need(n.In[1], UseCoopGemm(dev, f16, plan, n, gp) ? CK.GemmNK : CK.Raw);
                    Need(gp.BiasValue, CK.F32);
                    break;
                case LnP:
                    Need(n.In[0], CK.Raw); Need(n.In[1], CK.F32); Need(n.In[2], CK.F32);
                    break;
                case UpHeadP up:
                    Need(n.In[2], CK.Raw);
                    if (n.In.Length > 3) Need(n.In[3], CK.F32);
                    if (up.SkipConv is not null) { Need(up.SkipW, CK.ConvW); Need(up.SkipB, CK.F32); }
                    break;
                default:
                    foreach (int iv in n.In) Need(iv, CK.Raw);
                    break;
            }
        }
        foreach (var (v, k) in need)
        {
            CValue cv = plan.Values[v];
            if (k == CK.Raw && cv.Layout is { } l && !Engine.Layout.IsIdentity(l))
                throw new NotSupportedException($"Vulkan: const '{cv.Name}' with non-identity layout");
        }
        float[][] srcs = need.Select(x => plan.Values[x.v].Floats!).ToArray();
        int[] kinds = need.Select(x => (int)x.k).ToArray();
        _store = VkConstStore.Acquire(dev, constKey, f16, srcs, kinds, () =>
        {
            ulong total = 0;
            var offs = new ulong[need.Count];
            var items = new List<(ulong off, float[] data, bool half)>();
            for (int i = 0; i < need.Count; i++)
            {
                var (v, k) = need[i];
                CValue cv = plan.Values[v];
                float[] data = k switch
                {
                    CK.ConvW => PackConv(cv),
                CK.ConvWT => PackConvT(cv),
                CK.GemmNK => PackGemmNK(cv, plan.Nodes.First(m => m.P is GemmP && m.In[1] == v)),
                    CK.DwW => PackDw(cv),
                    _ => cv.Floats!,
                };
                bool half = f16 && k != CK.F32;
                total = (total + 63) & ~63UL;
                offs[i] = total;
                items.Add((total, data, half));
                total += (ulong)data.Length * (half ? 2UL : 4UL);
            }
            byte[] host = new byte[Math.Max((total + 63) & ~63UL, 64)];
            foreach (var (off, data, half) in items)
            {
                if (half)
                {
                    Span<Half> dst = MemoryMarshal.Cast<byte, Half>(host.AsSpan(checked((int)off), data.Length * 2));
                    for (int i = 0; i < data.Length; i++) dst[i] = (Half)data[i];
                }
                else
                    MemoryMarshal.AsBytes(data.AsSpan()).CopyTo(host.AsSpan(checked((int)off)));
            }
            return (offs, host);
        });
        for (int i = 0; i < need.Count; i++) _constOff[need[i]] = _store.Offsets[i];
        Consts = _store.Consts;
        Dummy = _store.Dummy;
        ConstBytes = _store.Bytes;

        var disps = new List<VkDispatch>();
        NodeRanges = new (int start, int count)[plan.Nodes.Length];
        for (int i = 0; i < plan.Nodes.Length; i++)
        {
            int start = disps.Count;
            disps.AddRange(Emit(plan.Nodes[i]));
            NodeRanges[i] = (start, disps.Count - start);
        }
        Dispatches = disps.ToArray();
    }

    // ONNX [OC][IC][KH][KW] -> [K][OC], k = (ky*KW + kx)*IC + ic (matches conv_tile.comp)
    private static float[] PackConv(CValue w)
    {
        int oc = w.Shape[0], ic = w.Shape[1], taps = w.Shape[2] * w.Shape[3];
        float[] src = w.Floats!, dst = new float[src.Length];
        for (int o = 0; o < oc; o++)
            for (int c = 0; c < ic; c++)
                for (int t = 0; t < taps; t++)
                    dst[(t * ic + c) * oc + o] = src[(o * ic + c) * taps + t];
        return dst;
    }

    // ONNX [OC][IC][KH][KW] -> [OC][Kpad], k = (ky*KW + kx)*IC + ic, Kpad = K rounded up to 8
    // (zero-filled; matches conv_cm.comp)
    private static float[] PackConvT(CValue w)
    {
        int oc = w.Shape[0], ic = w.Shape[1], taps = w.Shape[2] * w.Shape[3], k = (taps * ic + 7) & ~7;
        float[] src = w.Floats!, dst = new float[oc * k];
        for (int o = 0; o < oc; o++)
            for (int c = 0; c < ic; c++)
                for (int t = 0; t < taps; t++)
                    dst[o * k + t * ic + c] = src[(o * ic + c) * taps + t];
        return dst;
    }

    /// <summary>Tensor-core conv eligibility (conv_cm.comp): fp16 storage on a coopmat device,
    /// dense, channel counts in 16 B quads, an epilogue it implements, M within the grid limit.</summary>
    private static bool UseCoopConv(VkDevice dev, bool f16, ExecPlan plan, ExecNode n, ConvP cp)
    {
        if (!dev.CoopMat16 || dev.ConvDot || !f16 || cp.Group != 1 || cp.Pool is not null || cp.Dilations.Any(d => d != 1)) return false;
        if (cp.Act is not (ActivationKind.None or ActivationKind.Relu or ActivationKind.Gelu)) return false;
        if (cp.PadMode == PadMode.Constant && cp.PadValue != 0f) return false;
        int[] si = plan.Values[n.In[0]].PhysShape, so = plan.Values[n.Out[0]].PhysShape;   // [1,h,w,c]
        long m = (long)so[1] * so[2];
        return so[3] % 8 == 0 && (m + 127) / 128 <= 65535;   // cin % 8 != 0 -> scalar-gather stem variant
    }

    // Gemm B -> [N][K] (k fastest), the conv_cm weight layout
    private static float[] PackGemmNK(CValue b, ExecNode n)
    {
        var gp = (GemmP)n.P!;
        if (gp.TransB) return b.Floats!;   // already [N][K]
        int k = b.Shape[0], nn = b.Shape[1];
        float[] src = b.Floats!, dst = new float[src.Length];
        for (int i = 0; i < k; i++)
            for (int j = 0; j < nn; j++) dst[j * k + i] = src[i * nn + j];
        return dst;
    }

    /// <summary>Tensor-core Gemm (a 1x1 conv over M "pixels" through conv_cm.comp): fp16 storage
    /// on a coopmat device, const B, K/N in 16 B quads, relu/gelu/none epilogue.</summary>
    private static bool UseCoopGemm(VkDevice dev, bool f16, ExecPlan plan, ExecNode n, GemmP gp)
    {
        if (!dev.CoopMat16 || dev.ConvDot || !f16 || plan.Values[n.In[1]].Floats is null || plan.Values[n.In[1]].Shape.Length != 2) return false;
        // B packed per value: only when no other node reads it (its transB could differ)
        if (plan.Nodes.Count(m => m.In.Contains(n.In[1])) != 1) return false;
        if (gp.Act is not (ActivationKind.None or ActivationKind.Relu or ActivationKind.Gelu)) return false;
        CValue a = plan.Values[n.In[0]];
        int k = a.Shape[^1], nn = plan.Values[n.Out[0]].Shape[^1];
        if (a.PhysShape[^1] != k) return false;
        long m = a.ElementCount / k;
        return k % 8 == 0 && nn % 8 == 0 && (m + 127) / 128 <= 65535;
    }

    // ONNX [C][1][KH][KW] -> [tap][C]
    private static float[] PackDw(CValue w)
    {
        int c = w.Shape[0], taps = w.Shape[2] * w.Shape[3];
        float[] src = w.Floats!, dst = new float[src.Length];
        for (int ch = 0; ch < c; ch++)
            for (int t = 0; t < taps; t++) dst[t * c + ch] = src[ch * taps + t];
        return dst;
    }

    /// <summary>Dispatch index range per source ExecNode (for per-node debug runs).</summary>
    internal readonly (int start, int count)[] NodeRanges;

    // ---- value → bind resolution (byte offsets/sizes: all long math) ----
    private VkBind Bind(int valueId, CK kind = CK.Raw)
    {
        CValue v = _plan.Values[valueId];
        if (v.Buf >= 0)
            return new VkBind(VkSlot.Arena, (ulong)_plan.BufOffset[v.Buf] * (ulong)Es,
                (ulong)Math.Max(v.ElementCount, 1) * (ulong)Es);
        if (_constOff.TryGetValue((valueId, kind), out ulong off))
        {
            ulong es = F16 && kind != CK.F32 ? 2UL : 4UL;
            // ConvWT rows are K-padded to 8: bind the packed length, not the logical one
            long elems = kind == CK.ConvWT
                ? (long)v.Shape[0] * ((v.Shape[1] * v.Shape[2] * v.Shape[3] + 7) & ~7)
                : Math.Max(v.ElementCount, 1);
            return new VkBind(VkSlot.Consts, off, (ulong)elems * es);
        }
        if (v.Floats != null)
            throw new InvalidOperationException($"const value '{v.Name}' not packed as {kind}");
        throw new InvalidOperationException($"value '{v.Name}' has no GPU storage (ints must be folded)");
    }

    private static byte[] PcOf(params uint[] ints)
    {
        byte[] b = new byte[ints.Length * 4];
        Buffer.BlockCopy(ints, 0, b, 0, b.Length);
        return b;
    }

    private static byte[] PcOfF(uint[] ints, params float[] floats)
    {
        byte[] b = new byte[(ints.Length + floats.Length) * 4];
        Buffer.BlockCopy(ints, 0, b, 0, ints.Length * 4);
        Buffer.BlockCopy(floats, 0, b, ints.Length * 4, floats.Length * 4);
        return b;
    }

    private static uint[] U(params int[] v) { uint[] u = new uint[v.Length]; for (int i = 0; i < v.Length; i++) u[i] = (uint)v[i]; return u; }

    private static uint G(uint n) => (n + 255) / 256;

    /// <summary>
    /// Broadcast strides (padded rank-4, storage order of the output) of operand
    /// <paramref name="a"/> onto output <paramref name="o"/>. Output physical dim i
    /// holds logical dim o.Layout[i]; the operand's logical dim is that minus the
    /// numpy right-alignment offset.
    /// </summary>
    private static int[] PhysBroadcastStrides(CValue a, CValue o)
    {
        int rank = o.Shape.Length, off = rank - a.Shape.Length;
        if (rank > 4) throw new NotSupportedException("Vulkan elementwise: rank > 4");
        int[] lo = o.Layout ?? Engine.Layout.Identity(rank);
        int[] la = a.Layout ?? Engine.Layout.Identity(a.Shape.Length);
        int[] ap = a.PhysShape;
        int[] aStr = new int[ap.Length];
        for (int i = ap.Length - 1, acc = 1; i >= 0; i--) { aStr[i] = acc; acc *= ap[i]; }
        int[] r4 = new int[4];
        for (int i = 0; i < rank; i++)
        {
            int d = lo[i] - off;
            r4[4 - rank + i] = d < 0 || a.Shape[d] == 1 ? 0 : aStr[Array.IndexOf(la, d)];
        }
        return r4;
    }

    private string Sh(string name) => F16 ? name + "_h" : name;

    private IEnumerable<VkDispatch> Emit(ExecNode n)
    {
        CValue[] V = _plan.Values;
        // all addressing on storage-order shapes
        int[] shapeIn(int k) => V[n.In[k]].PhysShape;
        int[] shapeOut() => V[n.Out[0]].PhysShape;
        int nOut() => checked((int)V[n.Out[0]].ElementCount);

        switch (n.Op)
        {
            case ExecOp.Copy:
                yield return ND("unary", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf((uint)nOut(), 4), G((uint)nOut()));
                break;

            case ExecOp.Relu or ExecOp.Sigmoid or ExecOp.Erf or ExecOp.Gelu:
            {
                uint mode = n.Op switch
                {
                    ExecOp.Relu => 0, ExecOp.Sigmoid => 1, ExecOp.Erf => 2, _ => 3,
                };
                yield return ND("unary", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf((uint)nOut(), mode), G((uint)nOut()));
                break;
            }

            case ExecOp.Elementwise:
            {
                var p = (EwP)n.P!;
                CValue a = V[n.In[0]], bv = V[n.In[1]], o = V[n.Out[0]];
                int[] os = Pad4(shapeOut());
                int[] ostr = Strides(os);
                int[] astr = PhysBroadcastStrides(a, o);
                int[] bstr = PhysBroadcastStrides(bv, o);
                uint act = p.Act switch { ActivationKind.Relu => 1u, ActivationKind.Sigmoid => 2u, _ => 0u };
                // same-storage fast path: vec4 (n % 4 guard — no OOB tail writes)
                bool dense = astr.SequenceEqual(ostr) && bstr.SequenceEqual(ostr);
                if (nOut() % 4 == 0 && dense)
                {
                    yield return ND("ew2v", [Bind(n.In[0]), Bind(n.In[1]), Bind(n.Out[0])],
                        PcOf((uint)(nOut() / 4), (uint)p.Op, act), G((uint)(nOut() / 4)));
                    break;
                }
                yield return ND("ew2", [Bind(n.In[0]), Bind(n.In[1]), Bind(n.Out[0])],
                    PcOf(U(nOut(), ostr[0], ostr[1], ostr[2],
                        astr[0], astr[1], astr[2], astr[3],
                        bstr[0], bstr[1], bstr[2], bstr[3], (int)p.Op, (int)act)),
                    G((uint)nOut()));
                break;
            }

            case ExecOp.Conv:
            {
                var p = (ConvP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                if (p.Pool is { } pool)
                {
                    // small-ic stem conv + 2x2/2 max pool (GraphCompiler.GpuSmallConv)
                    int[] ci = shapeIn(0), co = shapeOut(), cw = V[n.In[1]].Shape;
                    if (!GraphCompiler.GpuSmallConv(p, cw, ci[3]) || p.AddValue >= 0 || ci[1] != co[1] * 2 || ci[2] != co[2] * 2)
                        throw new NotSupportedException("Vulkan convpool: unsupported conv");
                    uint pf = (p.BiasValue >= 0 ? 1u : 0u) | (p.PadMode == PadMode.Reflect ? 2u : 0u)
                        | (p.Act == ActivationKind.Relu ? 4u : 0u) | (pool.FuseRelu ? 8u : 0u);
                    bool k3 = _dev.ConvPoolK3 && p.Kernel[0] == 3 && p.Kernel[1] == 3;
                    string cpName = k3 ? "convpool_k3" : _dev.ConvDot && F16 && cw[0] % 8 == 0 ? "convpool_pi" : "convpool";
                    yield return ND(cpName,
                        [Bind(n.In[0]), Bind(n.In[1], CK.ConvW), B(p.BiasValue, CK.F32), Bind(n.Out[0])],
                        PcOfF(U(ci[3], ci[1], ci[2], cw[0], co[1], co[2], p.Kernel[0], p.Kernel[1], p.Pads[0], p.Pads[1], (int)pf),
                            p.PadMode == PadMode.Constant ? p.PadValue : 0f),
                        G((uint)co[1] * (uint)co[2] * (uint)(cw[0] / 8)), name: "convpool");
                    break;
                }
                if (p.AddValue >= 0) RequireNhwc(p.AddValue);
                int[] si = shapeIn(0), sw_ = V[n.In[1]].Shape, so = shapeOut();   // [1,h,w,c]
                uint ih = (uint)si[1], iw = (uint)si[2], ic = (uint)si[3];
                uint oc = (uint)sw_[0], oh = (uint)so[1], ow = (uint)so[2];
                if (p.PadMode == PadMode.Constant && p.PadValue != 0f)
                    throw new NotSupportedException("Vulkan conv: nonzero constant pad");
                if (p.Dilations.Any(d => d != 1))
                    throw new NotSupportedException("Vulkan conv: dilation");
                if (p.Act == ActivationKind.Sigmoid)
                    throw new NotSupportedException("Vulkan conv: sigmoid epilogue");
                uint flags = (p.BiasValue >= 0 ? 1u : 0u)
                    | (p.PadMode == PadMode.Reflect ? 2u : 0u)
                    | (p.Act == ActivationKind.Relu ? 4u : 0u)
                    | (p.AddValue >= 0 ? 8u : 0u)
                    | (p.Act == ActivationKind.Gelu ? 16u : 0u);
                if (p.Group > 1)
                {
                    if (p.Group != ic || oc != ic) throw new NotSupportedException("grouped non-depthwise conv");
                    if (p.AddValue >= 0) throw new NotSupportedException("depthwise conv + residual");
                    bool v4 = _dev.Vec4Eltwise && ic % 4 == 0;
                    yield return ND(v4 ? "depthwise4" : "depthwise",
                        [Bind(n.In[0]), Bind(n.In[1], CK.DwW), B(p.BiasValue, CK.F32), Bind(n.Out[0])],
                        PcOfF(U((int)ic, (int)ih, (int)iw, (int)oh, (int)ow, p.Kernel[0], p.Kernel[1],
                            p.Strides[0], p.Strides[1], p.Pads[0], p.Pads[1], (int)(flags & ~8u)), p.PadValue),
                        G(ic * oh * ow / (v4 ? 4u : 1u)), name: "depthwise");
                    break;
                }
                if (UseCoopConv(_dev, F16, _plan, n, p))
                {
                    // BM = 128 rows (pixels) in every variant; BN by oc divisibility
                    (string cm, uint bn) = ic % 8 != 0 ? ("conv_cm_n32s", 32u)
                        : oc % 128 == 0 ? ("conv_cm", 128u) : oc % 64 == 0 ? ("conv_cm_n64", 64u) : ("conv_cm_n32", 32u);
                    uint m = oh * ow, kr = (uint)(p.Kernel[0] * p.Kernel[1]) * ic, kd = (kr + 7u) & ~7u;
                    uint cflags = (p.BiasValue >= 0 ? 1u : 0u) | (p.AddValue >= 0 ? 2u : 0u)
                        | (p.PadMode == PadMode.Reflect ? 4u : 0u)
                        | ((p.Act == ActivationKind.Relu ? 1u : p.Act == ActivationKind.Gelu ? 2u : 0u) << 4);
                    yield return ND(cm,
                        [Bind(n.In[0]), Bind(n.In[1], CK.ConvWT), B(p.BiasValue, CK.F32), B(p.AddValue), Bind(n.Out[0])],
                        PcOf(m, oc, kd, cflags, ow, iw, ih, ic, (uint)p.Kernel[1],
                            (uint)p.Strides[0], (uint)p.Strides[1], (uint)p.Pads[0], (uint)p.Pads[1], kr),
                        (m + 127) / 128, (oc + bn - 1) / bn, name: "conv_cm");
                    break;
                }
                string convShader =
                    p.Kernel[0] == 3 && p.Kernel[1] == 3 && p.Strides[0] == 1 && p.Strides[1] == 1 ? "conv_k33s1" :
                    p.Kernel[0] == 3 && p.Kernel[1] == 3 && p.Strides[0] == 2 && p.Strides[1] == 2 ? "conv_k33s2" :
                    p.Kernel[0] == 1 && p.Kernel[1] == 1 && p.Strides[0] == 1 && p.Strides[1] == 1 ? "conv_k11s1" :
                    "conv_gen";
                if (_dev.ConvDot && F16 && convShader is "conv_k33s1" or "conv_k33s2" && ic % 4 == 0 && oc % 4 == 0)
                {
                    // Adreno 750 sweep (ms, 1600x1152 gc+dr): stride 1 at 4 px x 8 oc, stride 2 at 2 px x 8 oc
                    uint tm = p.Strides[1] == 1 ? 4u : 2u, tn = oc % 8 == 0 ? 2u : 1u;
                    uint threads = oc / (4 * tn) * ((ow + tm - 1) / tm) * oh;
                    uint wgs = (threads + 63) / 64, gx = Math.Min(wgs, 32768u);
                    yield return ND($"conv_dot_s{p.Strides[1]}_{tm}x{tn}",
                        [Bind(n.In[0]), Bind(n.In[1], CK.ConvW), B(p.BiasValue, CK.F32), B(p.AddValue), Bind(n.Out[0])],
                        PcOf(ic, ih, iw, oc, oh, ow, (uint)p.Kernel[0], (uint)p.Kernel[1],
                            (uint)p.Strides[0], (uint)p.Strides[1], (uint)p.Pads[0], (uint)p.Pads[1],
                            flags, 0),
                        gx, (wgs + gx - 1) / gx, name: "conv");
                    break;
                }
                yield return ND(convShader,
                    [Bind(n.In[0]), Bind(n.In[1], CK.ConvW), B(p.BiasValue, CK.F32), B(p.AddValue), Bind(n.Out[0])],
                    PcOf(ic, ih, iw, oc, oh, ow, (uint)p.Kernel[0], (uint)p.Kernel[1],
                        (uint)p.Strides[0], (uint)p.Strides[1], (uint)p.Pads[0], (uint)p.Pads[1],
                        flags, 0),
                    (ow + 15) / 16, (oh + 7) / 8, (oc + 63) / 64, name: "conv");
                break;
            }

            case ExecOp.MaxPool:
            {
                var p = (PoolP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                int[] si = shapeIn(0), so = shapeOut();
                bool v4 = _dev.Vec4Eltwise && si[3] % 4 == 0;
                yield return ND(v4 ? "maxpool4" : "maxpool", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf((uint)si[3], (uint)si[1], (uint)si[2], (uint)so[1], (uint)so[2],
                        (uint)p.Kernel[0], (uint)p.Kernel[1], (uint)p.Strides[0], (uint)p.Strides[1],
                        (uint)p.Pads[0], (uint)p.Pads[1], p.FuseRelu ? 1u : 0u),
                    G((uint)so[1] * (uint)so[2] * (uint)so[3] / (v4 ? 4u : 1u)), name: "maxpool");
                break;
            }

            case ExecOp.LayerNorm:
            {
                var p = (LnP)n.P!;
                uint len = (uint)V[n.In[0]].Shape[^1];
                if ((uint)shapeIn(0)[^1] != len) throw new NotSupportedException("Vulkan LayerNorm: last dim not innermost");
                uint rows = (uint)(V[n.In[0]].ElementCount / len);
                yield return ND("layernorm",
                    [Bind(n.In[0]), Bind(n.In[1], CK.F32), Bind(n.In[2], CK.F32), Bind(n.Out[0])],
                    PcOfF([len], p.Eps), rows, name: "layernorm");
                break;
            }

            case ExecOp.Gemm:
            {
                var p = (GemmP)n.P!;
                uint k = (uint)V[n.In[0]].Shape[^1];
                if ((uint)shapeIn(0)[^1] != k) throw new NotSupportedException("Vulkan Gemm: K not innermost");
                uint m = (uint)(V[n.In[0]].ElementCount / k);
                uint nn = (uint)V[n.Out[0]].Shape[^1];
                if (UseCoopGemm(_dev, F16, _plan, n, p))
                {
                    (string cm, uint bn) = nn % 128 == 0 ? ("conv_cm", 128u) : nn % 64 == 0 ? ("conv_cm_n64", 64u) : ("conv_cm_n32", 32u);
                    uint cflags = (p.BiasValue >= 0 ? 1u : 0u)
                        | ((p.Act == ActivationKind.Relu ? 1u : p.Act == ActivationKind.Gelu ? 2u : 0u) << 4);
                    // [M,K] x [N,K]^T as a 1x1 conv over a 1 x M image
                    yield return ND(cm,
                        [Bind(n.In[0]), Bind(n.In[1], CK.GemmNK), B(p.BiasValue, CK.F32), B(-1), Bind(n.Out[0])],
                        PcOf(m, nn, k, cflags, m, m, 1u, k, 1u, 1u, 1u, 0u, 0u, k),
                        (m + 127) / 128, (nn + bn - 1) / bn, name: "gemm_cm");
                    break;
                }
                uint gflags = (p.TransB ? 1u : 0u) | (p.BiasValue >= 0 ? 2u : 0u)
                    | (p.Act == ActivationKind.Relu ? 4u : 0u)
                    | (p.Act == ActivationKind.Gelu ? 8u : 0u);
                if (_dev.ConvDot && F16 && !p.TransB && k % 4 == 0 && nn % 8 == 0)
                {
                    // 4 rows x 8 columns per invocation
                    uint gth = nn / 8 * ((m + 3) / 4);
                    uint gwg = (gth + 63) / 64, ggx = Math.Min(gwg, 32768u);
                    yield return ND("gemm_dot",
                        [Bind(n.In[0]), Bind(n.In[1]), B(p.BiasValue, CK.F32), Bind(n.Out[0])],
                        PcOf(m, nn, k, gflags), ggx, (gwg + ggx - 1) / ggx, name: "gemm");
                    break;
                }
                yield return ND("gemm",
                    [Bind(n.In[0]), Bind(n.In[1]), B(p.BiasValue, CK.F32), Bind(n.Out[0])],
                    PcOf(m, nn, k, gflags),
                    (nn + 63) / 64, (m + 63) / 64, name: "gemm");
                break;
            }

            case ExecOp.ResizeBilinear:
            {
                var p = (ResizeP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                bool hasAdd = n.In.Length > 1;
                if (hasAdd) RequireNhwc(n.In[1]);
                int[] si = shapeIn(0), so = shapeOut();
                uint flags = (p.FuseRelu ? 1u : 0u) | (hasAdd ? 2u : 0u) | (p.AddRelu ? 4u : 0u);
                bool v4 = _dev.Vec4Eltwise && si[3] % 4 == 0;
                yield return ND(v4 ? "bilinear4" : "bilinear", [Bind(n.In[0]), hasAdd ? Bind(n.In[1]) : B(-1), Bind(n.Out[0])],
                    PcOfF([(uint)si[3], (uint)si[1], (uint)si[2], (uint)so[1], (uint)so[2], flags],
                        p.Scales[2], p.Scales[3]),
                    G((uint)so[1] * (uint)so[2] * (uint)so[3] / (v4 ? 4u : 1u)), name: "bilinear");
                break;
            }

            case ExecOp.UpHead:
            {
                var p = (UpHeadP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.In[1]); RequireNhwc(n.Out[0]);
                int[] si = shapeIn(0), so = shapeOut();
                uint c = (uint)si[3], oc = (uint)so[3];
                if (oc > 8 || c > 128 || c % 4 != 0) throw new NotSupportedException($"Vulkan UpHead: c={c} oc={oc}");
                bool hasBias = n.In.Length > 3;
                uint act = p.Act switch { ActivationKind.Relu => 1u, ActivationKind.Sigmoid => 2u, ActivationKind.None => 0u,
                    _ => throw new NotSupportedException($"Vulkan UpHead: {p.Act}") };
                uint flags = (p.ResizeRelu ? 1u : 0u) | (p.SkipRelu ? 2u : 0u) | (hasBias ? 4u : 0u) | (act << 3);
                if (p.SkipConv is { } sc)
                {
                    // skip recomputed from the stem conv's input (GraphCompiler.GpuSmallConv, c <= 32)
                    int[] xi = shapeIn(1), cw = V[p.SkipW].Shape;
                    if (!GraphCompiler.GpuSmallConv(sc, cw, xi[3]) || cw[0] != c || c > 32 || xi[1] != so[1] || xi[2] != so[2])
                        throw new NotSupportedException("Vulkan UpHead: unsupported skip conv");
                    uint cf = (sc.BiasValue >= 0 ? 1u : 0u) | (sc.PadMode == PadMode.Reflect ? 2u : 0u)
                        | (sc.Act == ActivationKind.Relu ? 4u : 0u);
                    uint np = _dev.ConvDot && F16 ? 2u : 1u;
                    yield return ND(np > 1 ? $"uphead_sc{np}" : "uphead_sc",
                        [Bind(n.In[0]), Bind(n.In[1]), Bind(n.In[2]), hasBias ? Bind(n.In[3], CK.F32) : B(-1, CK.F32), Bind(n.Out[0]),
                         Bind(p.SkipW, CK.ConvW), B(p.SkipB, CK.F32)],
                        PcOfF([c, (uint)si[1], (uint)si[2], (uint)so[1], (uint)so[2], oc, flags],
                            p.Scales[2], p.Scales[3]).Concat(PcOfF(U(xi[3], sc.Kernel[0], sc.Kernel[1], sc.Pads[0], sc.Pads[1], (int)cf),
                            sc.PadMode == PadMode.Constant ? sc.PadValue : 0f)).ToArray(),
                        G((uint)so[1] * (((uint)so[2] + np - 1) / np)), name: "uphead_sc");
                    break;
                }
                yield return ND("uphead",
                    [Bind(n.In[0]), Bind(n.In[1]), Bind(n.In[2]), hasBias ? Bind(n.In[3], CK.F32) : B(-1, CK.F32), Bind(n.Out[0])],
                    PcOfF([c, (uint)si[1], (uint)si[2], (uint)so[1], (uint)so[2], oc, flags],
                        p.Scales[2], p.Scales[3]),
                    G((uint)so[1] * (uint)so[2]));
                break;
            }

            case ExecOp.Pad:
            {
                // generic rank-4 pad in storage order (pads permuted to physical dims)
                var p = (PadP)n.P!;
                CValue x = V[n.In[0]];
                if (x.Shape.Length != 4) throw new NotSupportedException("Vulkan pad: rank != 4");
                int[] l = x.Layout ?? Engine.Layout.Identity(4);
                int[] id4 = shapeIn(0), od4 = shapeOut();
                int[] pb4 = new int[4];
                for (int d = 0; d < 4; d++) pb4[d] = p.Pads[l[d]];
                yield return ND("pad", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOfF(U(id4[0], id4[1], id4[2], id4[3], od4[0], od4[1], od4[2], od4[3],
                        pb4[0], pb4[1], pb4[2], pb4[3], p.Mode == PadMode.Reflect ? 1 : 0),
                        p.Value),
                    G((uint)V[n.Out[0]].ElementCount));
                break;
            }

            case ExecOp.Transpose:
            {
                // only planner relayouts reach here; the perm is physical
                var p = (TransposeP)n.P!;
                int[] is4 = Pad4(shapeIn(0));
                int[] os4 = Pad4(shapeOut());
                int[] ostr = Strides(os4), istr = Strides(is4);
                int[] perm4 = PermuteTo4(p.Perm, shapeIn(0).Length);
                yield return ND("transpose", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf(U(nOut(), ostr[0], ostr[1], ostr[2],
                        istr[perm4[0]], istr[perm4[1]], istr[perm4[2]], istr[perm4[3]])),
                    G((uint)nOut()));
                break;
            }

            case ExecOp.Slice:
            {
                // SliceP.Axes are physical after the layout pass
                var p = (SliceP)n.P!;
                int[] is4 = Pad4(shapeIn(0)), os4 = Pad4(shapeOut());
                int[] ostr = Strides(os4), istr = Strides(is4);
                uint[] st = [0, 0, 0, 0], sp = [1, 1, 1, 1];
                int shift = 4 - shapeIn(0).Length;
                for (int i = 0; i < p.Axes.Length; i++)
                {
                    st[p.Axes[i] + shift] = (uint)p.Starts[i];
                    sp[p.Axes[i] + shift] = (uint)p.Steps[i];
                }
                yield return ND("slice", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf(U(nOut(), ostr[0], ostr[1], ostr[2],
                        (int)st[0], (int)st[1], (int)st[2], (int)st[3],
                        (int)sp[0], (int)sp[1], (int)sp[2], (int)sp[3],
                        istr[0], istr[1], istr[2], istr[3])),
                    G((uint)nOut()));
                break;
            }

            case ExecOp.Concat:
            {
                // ConcatP.Axis is physical after the layout pass
                var p = (ConcatP)n.P!;
                int[] os = shapeOut();
                int axis = p.Axis;
                int inner = 1, dstAxis = os[axis];
                for (int d = axis + 1; d < os.Length; d++) inner *= os[d];
                int axisOff = 0;
                foreach (int iv in n.In)
                {
                    int[] ishape = V[iv].PhysShape;
                    int cnt = checked((int)V[iv].ElementCount);
                    if (cnt > 0)
                    {
                        yield return ND("concat_ax", [Bind(iv), Bind(n.Out[0])],
                            PcOf((uint)cnt, (uint)inner, (uint)ishape[axis], (uint)dstAxis, (uint)axisOff),
                            G((uint)cnt));
                    }
                    axisOff += ishape[axis];
                }
                break;
            }

            case ExecOp.ChannelShift:
            {
                var p = (ShiftP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                int[] si = shapeIn(0);   // [1,h,w,c]
                var ge = p.GroupEnds; var of = p.Offsets;
                if (ge.Length > 5) throw new NotSupportedException("Vulkan shift: > 5 groups");
                var pc = new List<uint> { (uint)si[3], (uint)si[1], (uint)si[2], (uint)p.Axis };
                for (int i = 0; i < 5; i++) pc.Add((uint)(i < ge.Length ? ge[i] : si[3]));
                for (int i = 0; i < 5; i++) pc.Add((uint)(i < of.Length ? of[i] : 0));
                bool v4 = _dev.Vec4Eltwise && si[3] % 4 == 0 && ge.All(e => e % 4 == 0);
                yield return ND(v4 ? "shift4" : "shift", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf(pc.ToArray()), G((uint)nOut() / (v4 ? 4u : 1u)), name: "shift");
                break;
            }

            default:
                throw new NotSupportedException($"Vulkan: ExecOp {n.Op} not implemented");
        }

        VkDispatch ND(string shader, VkBind[] binds, byte[] pc, uint gx, uint gy = 1, uint gz = 1, string? name = null)
        {
            return new VkDispatch
            {
                Pipe = shader.StartsWith("conv_cm", StringComparison.Ordinal)
                    ? _dev.GetCoopPipeline(shader, binds.Length, pc.Length)
                    : _dev.GetPipeline(Sh(shader), binds.Length, pc.Length),
                Pc = pc, Binds = binds, Gx = gx, Gy = gy, Gz = gz,
                Name = $"{shader}({n.Name})",
            };
        }

        VkBind B(int valueId, CK kind = CK.Raw) => valueId >= 0 ? Bind(valueId, kind) : new VkBind(VkSlot.Dummy, 0, 64);
    }

    private void RequireNhwc(int v)
    {
        CValue cv = _plan.Values[v];
        if (cv.Shape.Length != 4 || !Engine.Layout.Equivalent(cv.Shape, cv.Layout ?? Engine.Layout.Identity(4), Engine.Layout.Nhwc))
            throw new NotSupportedException($"Vulkan: '{cv.Name}' expected NHWC storage, got {Engine.Layout.Str(cv.Layout)}");
    }

    // pad a shape to rank 4 with leading 1s
    private static int[] Pad4(int[] s)
    {
        if (s.Length == 4) return s;
        int[] r = [1, 1, 1, 1];
        s.CopyTo(r, 4 - s.Length);
        return r;
    }

    private static int[] Strides(int[] shape4)
    {
        int[] s = new int[4]; s[3] = 1;
        for (int i = 2; i >= 0; i--) s[i] = s[i + 1] * shape4[i + 1];
        return s;
    }

    // lift an original-rank perm into padded rank-4 (leading dims map to themselves)
    private static int[] PermuteTo4(int[] perm, int rank)
    {
        int shift = 4 - rank;
        int[] p4 = [0, 1, 2, 3];
        for (int i = 0; i < rank; i++) p4[i + shift] = perm[i] + shift;
        return p4;
    }

    /// <summary>Sessions hold a reference so cache eviction can't free live const/dummy buffers.</summary>
    public void Acquire()
    {
        if (Interlocked.Increment(ref _refs) <= 1)
            throw new ObjectDisposedException(nameof(VkGraphModel), "const buffers already freed");
    }
    /// <summary>Acquire unless already freed (no resurrection; for racing cache readers).</summary>
    public bool TryAcquire()
    {
        while (true)
        {
            int r = Volatile.Read(ref _refs);
            if (r <= 0) return false;
            if (Interlocked.CompareExchange(ref _refs, r + 1, r) == r) return true;
        }
    }
    public void Release()
    {
        if (Interlocked.Decrement(ref _refs) == 0) _store.Release();
    }

    /// <summary>The plan this graph was compiled from (arena offsets, I/O values).</summary>
    internal ExecPlan Plan => _plan;
    public void Dispose() => Release();
}
