using Sdcb.SimdOnnx.Backends.Metal;
using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Kernels;

namespace Sdcb.SimdGcdrNet.Extensions;

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
    // In[1] is the input of this (still scheduled) conv and the skip is recomputed,
    // so the full-resolution skip tensor need not stay live
    public ConvP? SkipConv;
    public int SkipW = -1, SkipB = -1;   // the skip conv's weight / bias value ids

    public override string ToString() => $"rrelu={ResizeRelu} srelu={SkipRelu} act={Act}"
        + (SkipConv is { } sc ? $" skip=conv{sc.Kernel[0]}x{sc.Kernel[1]}(In[1]) sact={sc.Act}" : "");
}

internal sealed class UpHeadOp : ExtOp
{
    public static readonly UpHeadOp Instance = new();
    public override string Name => "UpHead";

    public override IEnumerable<int> ParamRefs(ExecNode n)
    {
        var p = (UpHeadP)n.P!;
        if (p.SkipW >= 0) yield return p.SkipW;
        if (p.SkipB >= 0) yield return p.SkipB;
    }

    public override void PlanLayout(LayoutPlanner lp, ExecNode n)
    {
        // activations channels-last, weights stay logical
        if (lp.Values[n.In[0]].Shape.Length != 4 || lp.Values[n.In[1]].Shape.Length != 4)
            throw new NotSupportedException($"layout: UpHead '{n.Name}' is not rank-4.");
        lp.Need(n, 0, Layout.Nhwc);
        lp.Need(n, 1, Layout.Nhwc);
        lp.Values[n.Out[0]].Layout = Layout.Nhwc;
    }

    public override void PackCpu(ExecPlan plan, ExecNode n, CpuKernels.Cache? cache)
    {
        var up = (UpHeadP)n.P!;
        if (up.KernelPlan is not null) return;
        CValue w = plan.Values[n.In[2]];
        float[]? b = n.In.Length > 3 ? plan.Values[n.In[3]].Floats : null;
        object kp = CpuKernels.Get(cache, n, w.Floats!, b, () => GcdrKernels.PackUpHead(w.Floats!, w.Shape[0], w.Shape[1], b));
        Interlocked.CompareExchange(ref up.KernelPlan, kp, null);
    }

    public override void RunCpu(InferenceSession s, ExecNode n)
    {
        var p = (UpHeadP)n.P!;
        CValue[] v = s.Plan.Values;
        int[] i = v[n.In[0]].PhysShape, o = v[n.Out[0]].PhysShape, k = v[n.In[1]].PhysShape;
        GcdrKernels.UpHead(s.GetIn(n.In[0]), i[1], i[2], i[3], s.GetIn(n.In[1]), s.GetSpan(n.Out[0]), o[1], o[2],
            p.Scales[2], p.Scales[3], p, (UpHeadPlan)p.KernelPlan!, k[1], k[2], k[3]);
    }

    public override IEnumerable<(int value, ConstKind kind)> GpuConsts(ExecNode n)
    {
        var up = (UpHeadP)n.P!;
        yield return (n.In[2], ConstKind.Raw);
        if (n.In.Length > 3) yield return (n.In[3], ConstKind.F32);
        if (up.SkipConv is not null)
        {
            yield return (up.SkipW, ConstKind.ConvW);
            if (up.SkipB >= 0) yield return (up.SkipB, ConstKind.F32);
        }
    }

    private static (uint c, uint oc, uint flags) Common(CValue[] V, ExecNode n, string backend)
    {
        var p = (UpHeadP)n.P!;
        int[] si = V[n.In[0]].PhysShape, so = V[n.Out[0]].PhysShape;
        uint c = (uint)si[3], oc = (uint)so[3];
        if (oc > 8 || c > 128 || c % 4 != 0) throw new NotSupportedException($"{backend} UpHead: c={c} oc={oc}");
        uint act = p.Act switch
        {
            ActivationKind.Relu => 1u, ActivationKind.Sigmoid => 2u, ActivationKind.None => 0u,
            _ => throw new NotSupportedException($"{backend} UpHead: {p.Act}"),
        };
        uint flags = (p.ResizeRelu ? 1u : 0u) | (p.SkipRelu ? 2u : 0u) | (n.In.Length > 3 ? 4u : 0u) | (act << 3);
        return (c, oc, flags);
    }

    private static byte[] SkipPc(CValue[] V, ExecNode n, uint c, string backend)
    {
        var p = (UpHeadP)n.P!;
        var sc = p.SkipConv!;
        // skip recomputed from the stem conv's input (GraphCompiler.GpuSmallConv, c <= 32)
        int[] xi = V[n.In[1]].PhysShape, cw = V[p.SkipW].Shape, so = V[n.Out[0]].PhysShape;
        if (!GraphCompiler.GpuSmallConv(sc, cw, xi[3]) || cw[0] != c || c > 32 || xi[1] != so[1] || xi[2] != so[2])
            throw new NotSupportedException($"{backend} UpHead: unsupported skip conv");
        uint cf = (sc.BiasValue >= 0 ? 1u : 0u) | (sc.PadMode == PadMode.Reflect ? 2u : 0u)
            | (sc.Act == ActivationKind.Relu ? 4u : 0u);
        return VkGraphModel.PcOfF(VkGraphModel.U(xi[3], sc.Kernel[0], sc.Kernel[1], sc.Pads[0], sc.Pads[1], (int)cf),
            sc.PadMode == PadMode.Constant ? sc.PadValue : 0f);
    }

    public override IEnumerable<VkDispatch> EmitVulkan(VkGraphModel g, ExecNode n)
    {
        var p = (UpHeadP)n.P!;
        CValue[] V = g.Plan.Values;
        g.RequireNhwc(n.In[0]); g.RequireNhwc(n.In[1]); g.RequireNhwc(n.Out[0]);
        int[] si = V[n.In[0]].PhysShape, so = V[n.Out[0]].PhysShape;
        var (c, oc, flags) = Common(V, n, "Vulkan");
        byte[] head = VkGraphModel.PcOfF([c, (uint)si[1], (uint)si[2], (uint)so[1], (uint)so[2], oc, flags], p.Scales[2], p.Scales[3]);
        VkBind bias = n.In.Length > 3 ? g.Bind(n.In[3], ConstKind.F32) : g.BindOrDummy(-1, ConstKind.F32);
        if (p.SkipConv is not null)
        {
            uint np = g.Device.ConvDot && g.F16 ? 2u : 1u;
            yield return g.Dispatch(n, np > 1 ? $"uphead_sc{np}" : "uphead_sc",
                [g.Bind(n.In[0]), g.Bind(n.In[1]), g.Bind(n.In[2]), bias, g.Bind(n.Out[0]),
                 g.Bind(p.SkipW, ConstKind.ConvW), g.BindOrDummy(p.SkipB, ConstKind.F32)],
                [.. head, .. SkipPc(V, n, c, "Vulkan")],
                VkGraphModel.G((uint)so[1] * (((uint)so[2] + np - 1) / np)), owner: GcdrExtensions.Owner);
            yield break;
        }
        yield return g.Dispatch(n, "uphead",
            [g.Bind(n.In[0]), g.Bind(n.In[1]), g.Bind(n.In[2]), bias, g.Bind(n.Out[0])],
            head, VkGraphModel.G((uint)so[1] * (uint)so[2]), owner: GcdrExtensions.Owner);
    }

    public override IEnumerable<MtlDispatch> EmitMetal(MtlGraphModel g, ExecNode n)
    {
        var p = (UpHeadP)n.P!;
        CValue[] V = g.Plan.Values;
        g.RequireNhwc(n.In[0]); g.RequireNhwc(n.In[1]); g.RequireNhwc(n.Out[0]);
        int[] si = V[n.In[0]].PhysShape, so = V[n.Out[0]].PhysShape;
        var (c, oc, flags) = Common(V, n, "Metal");
        byte[] head = MtlGraphModel.PcOfF([c, (uint)si[1], (uint)si[2], (uint)so[1], (uint)so[2], oc, flags], p.Scales[2], p.Scales[3]);
        MtlBind bias = n.In.Length > 3 ? g.Bind(n.In[3], ConstKind.F32) : g.BindOrDummy(-1, ConstKind.F32);
        if (p.SkipConv is not null)
        {
            yield return g.Dispatch(n, "uphead_sc",
                [g.Bind(n.In[0]), g.Bind(n.In[1]), g.Bind(n.In[2]), bias, g.Bind(n.Out[0]),
                 g.Bind(p.SkipW, ConstKind.ConvW), g.BindOrDummy(p.SkipB, ConstKind.F32)],
                [.. head, .. SkipPc(V, n, c, "Metal")],
                MtlGraphModel.G((uint)so[1] * (uint)so[2]), owner: GcdrExtensions.Owner);
            yield break;
        }
        yield return g.Dispatch(n, "uphead",
            [g.Bind(n.In[0]), g.Bind(n.In[1]), g.Bind(n.In[2]), bias, g.Bind(n.Out[0])],
            head, MtlGraphModel.G((uint)so[1] * (uint)so[2]), owner: GcdrExtensions.Owner);
    }
}

/// <summary>
/// Resize → (Relu) → Add(·, [Relu](skip)) → 1x1 Conv (few outputs) → (Sigmoid|Relu) becomes one
/// <see cref="UpHeadOp"/> node in NHWC plans — the network tail, where every intermediate is a
/// full-resolution tensor that would otherwise round-trip memory. The fused node takes the
/// position of the last node it replaces, so every input is produced before it.
/// </summary>
internal sealed class FuseUpHeadPass : GraphPass
{
    public override PassStage Stage => PassStage.Logical;

    public override void Run(PassContext ctx)
    {
        if (!ctx.Options.Nhwc || Env.Flag("NOFUSE_UP") == "1") return;
        List<ExecNode> nodes = ctx.Nodes;
        List<CValue> values = ctx.Values;

        // resolve add(resize(x), other): (resize node, skip value, skip relu node or null)
        (ExecNode rs, int skip, ExecNode? relu)? MatchAdd(ExecNode add)
        {
            if (add.Op != ExecOp.Elementwise || add.In.Length != 2 || add.P is not EwP { Op: EwOp.Add, Act: ActivationKind.None })
                return null;
            for (int k = 0; k < 2; k++)
            {
                int u = add.In[k], o = add.In[1 - k];
                int ri = ctx.Producer(u);
                if (ri < 0 || nodes[ri].Op != ExecOp.ResizeBilinear || nodes[ri].In.Length != 1) continue;
                if (ctx.Uses(u) != 1 || !ctx.SameShape(u, add.Out[0]) || !ctx.SameShape(o, add.Out[0]) || values[o].Buf < 0) continue;
                int pi = ctx.Producer(o);
                if (pi >= 0 && nodes[pi].Op == ExecOp.Relu && nodes[pi].In.Length == 1 && ctx.Uses(o) == 1
                    && values[nodes[pi].In[0]].Buf >= 0 && ctx.SameShape(nodes[pi].In[0], o))
                    return (nodes[ri], nodes[pi].In[0], nodes[pi]);
                return (nodes[ri], o, null);
            }
            return null;
        }

        for (int ci = 0; ci < nodes.Count; ci++)
        {
            ExecNode c = nodes[ci];
            if (c.Op != ExecOp.Conv || c.P is not ConvP cp || cp.Group != 1 || cp.Act != ActivationKind.None
                || cp.AddValue >= 0 || cp.Kernel[0] != 1 || cp.Kernel[1] != 1 || cp.Strides.Any(s => s != 1)
                || cp.Pads.Any(p => p != 0) || cp.Dilations.Any(d => d != 1)) continue;
            CValue w = values[c.In[1]];
            // kernel limits (oc lanes, staged weights, vec4 channels)
            if (w.Floats is null || w.Shape.Length != 4 || w.Shape[0] > 8 || w.Shape[1] > 128 || w.Shape[1] % 4 != 0) continue;
            if (cp.BiasValue >= 0 && values[cp.BiasValue].Floats is null) continue;
            int s = c.In[0];
            int ai = ctx.Producer(s);
            if (ai < 0 || ctx.Uses(s) != 1 || values[s].Shape.Length != 4) continue;
            ExecNode addNode = nodes[ai];
            if (MatchAdd(addNode) is not { } mt) continue;
            var (rs, skip, relu) = mt;

            ExecNode? act = null;
            int y = c.Out[0];
            if (ctx.Uses(y) == 1)
                act = nodes.FirstOrDefault(m => m.In.Length == 1 && m.In[0] == y
                    && m.Op is ExecOp.Sigmoid or ExecOp.Relu && values[m.Out[0]].Buf >= 0);
            var head = new ExecNode
            {
                Op = ExecOp.Ext,
                Ext = UpHeadOp.Instance,
                In = cp.BiasValue >= 0 ? [rs.In[0], skip, c.In[1], cp.BiasValue] : [rs.In[0], skip, c.In[1]],
                Out = [act?.Out[0] ?? y],
                P = new UpHeadP
                {
                    Scales = ((ResizeP)rs.P!).Scales,
                    ResizeRelu = ((ResizeP)rs.P!).FuseRelu,
                    SkipRelu = relu is not null,
                    Act = act is null ? ActivationKind.None : act.Op == ExecOp.Sigmoid ? ActivationKind.Sigmoid : ActivationKind.Relu,
                },
                SrcNode = c.SrcNode,
                Name = c.Name + ".uphead",
            };
            nodes[ci] = head;
            nodes.Remove(rs); nodes.Remove(addNode);
            if (relu is not null) nodes.Remove(relu);
            if (act is not null) nodes.Remove(act);
            ci = -1;   // indices shifted; rescan (few nodes, cheap)
        }
    }
}

/// <summary>
/// UpHead skip recompute (after layout planning): when the skip is a cheap direct conv of a
/// smaller input (stem conv: 3/6 → 16/32 channels), UpHead reads the conv's input and recomputes
/// the skip rows, so the skip need not stay live across the network. GPU plans only get it for the
/// small-ic convs their shaders stage (<see cref="GraphCompiler.GpuSmallConv"/>).
/// </summary>
internal sealed class UpHeadSkipRecomputePass : GraphPass
{
    public override PassStage Stage => PassStage.Physical;

    public override void Run(PassContext ctx)
    {
        List<ExecNode> nodes = ctx.Nodes;
        List<CValue> values = ctx.Values;
        foreach (ExecNode u in nodes)
        {
            if (u.P is not UpHeadP up || up.SkipConv is not null) continue;
            int s = u.In[1], ci = ctx.Producer(s);
            if (ci < 0 || nodes[ci].P is not ConvP cp || cp.Group != 1 || cp.AddValue >= 0 || cp.Pool is not null
                || cp.Strides.Any(x => x != 1) || cp.Dilations.Any(x => x != 1)) continue;
            int xin = nodes[ci].In[0];
            CValue w = values[nodes[ci].In[1]];
            if (w.Floats is null || w.Shape.Length != 4 || !ctx.IsNhwc4(s) || !ctx.IsNhwc4(xin)) continue;
            int ic = values[xin].Shape[1], c = values[s].Shape[1];
            // cheap to recompute, much smaller to keep; the conv itself stays scheduled
            // for its other reader(s) (and keeps the shared ConvP packed)
            if (ic * 2 > c || Nhwc.WinogradEligible(cp, w.Shape, ic) || ctx.Uses(s) < 2) continue;
            if (cp.BiasValue >= 0 && values[cp.BiasValue].Floats is null) continue;
            bool sameHw = values[xin].Shape[2] == values[s].Shape[2] && values[xin].Shape[3] == values[s].Shape[3];
            if (ctx.Options.Gpu && (!GraphCompiler.GpuSmallConv(cp, w.Shape, ic) || c > 32 || !sameHw)) continue;
            u.In[1] = xin;
            up.SkipConv = cp;
            up.SkipW = nodes[ci].In[1];
            up.SkipB = cp.BiasValue;
        }
    }
}
