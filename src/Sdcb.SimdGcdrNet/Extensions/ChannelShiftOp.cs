using Sdcb.SimdOnnx.Backends.Metal;
using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Kernels;

namespace Sdcb.SimdGcdrNet.Extensions;

internal sealed class ShiftP
{
    public int Axis;                 // logical spatial axis index (2 = H, 3 = W)
    public required int[] GroupEnds; // cumulative channel-group end indices (len 5)
    public required int[] Offsets;   // per-group shift (len 5)
}

/// <summary>Fused shift-MLP: per channel group, shift along H or W (zero-fill). In = [x], P = <see cref="ShiftP"/>.</summary>
internal sealed class ChannelShiftOp : ExtOp
{
    public static readonly ChannelShiftOp Instance = new();
    public override string Name => "ChannelShift";

    public override void RunCpu(InferenceSession s, ExecNode n)
    {
        var p = (ShiftP)n.P!;
        CValue[] v = s.Plan.Values;
        if (s.Plan.Nhwc)
        {
            int[] i = v[n.In[0]].PhysShape;
            GcdrKernels.ChannelShift(s.GetIn(n.In[0]), i[1], i[2], i[3], s.GetSpan(n.Out[0]), p);
        }
        else
            GcdrKernels.ChannelShiftNchw(s.GetIn(n.In[0]), v[n.In[0]].PhysShape, s.GetSpan(n.Out[0]), v[n.Out[0]].PhysShape, p);
    }

    private static (uint[] pc, bool v4) Pc(int[] si, ShiftP p, bool vec4)
    {
        var ge = p.GroupEnds; var of = p.Offsets;
        if (ge.Length > 5) throw new NotSupportedException("GPU shift: > 5 groups");
        var pc = new List<uint> { (uint)si[3], (uint)si[1], (uint)si[2], (uint)p.Axis };
        for (int i = 0; i < 5; i++) pc.Add((uint)(i < ge.Length ? ge[i] : si[3]));
        for (int i = 0; i < 5; i++) pc.Add((uint)(i < of.Length ? of[i] : 0));
        return (pc.ToArray(), vec4 && si[3] % 4 == 0 && ge.All(e => e % 4 == 0));
    }

    public override IEnumerable<VkDispatch> EmitVulkan(VkGraphModel g, ExecNode n)
    {
        g.RequireNhwc(n.In[0]); g.RequireNhwc(n.Out[0]);
        int[] si = g.Plan.Values[n.In[0]].PhysShape;   // [1,h,w,c]
        var (pc, v4) = Pc(si, (ShiftP)n.P!, g.Device.Vec4Eltwise);
        uint count = (uint)g.Plan.Values[n.Out[0]].ElementCount / (v4 ? 4u : 1u);
        yield return g.Dispatch(n, v4 ? "shift4" : "shift", [g.Bind(n.In[0]), g.Bind(n.Out[0])],
            VkGraphModel.PcOf(pc), VkGraphModel.G(count), owner: GcdrExtensions.Owner);
    }

    public override IEnumerable<MtlDispatch> EmitMetal(MtlGraphModel g, ExecNode n)
    {
        g.RequireNhwc(n.In[0]); g.RequireNhwc(n.Out[0]);
        int[] si = g.Plan.Values[n.In[0]].PhysShape;
        var (pc, v4) = Pc(si, (ShiftP)n.P!, true);
        uint count = (uint)g.Plan.Values[n.Out[0]].ElementCount / (v4 ? 4u : 1u);
        yield return g.Dispatch(n, v4 ? "shift4" : "shift", [g.Bind(n.In[0]), g.Bind(n.Out[0])],
            MtlGraphModel.PcOf(pc), MtlGraphModel.G(count), owner: GcdrExtensions.Owner);
    }
}

/// <summary>
/// Recognizes the shift-MLP subgraph: Pad(const ±p on H or W) -> channel Split (5 groups via
/// Slice) -> per-group roll (Slice+Slice+Concat on the spatial axis) -> Concat(axis=1) ->
/// Slice(unpad). Replaced by a single <see cref="ChannelShiftOp"/> node writing the shifted [1,C,h,w].
/// </summary>
internal sealed class FuseShiftPass : GraphPass
{
    public override PassStage Stage => PassStage.Logical;

    public override void Run(PassContext ctx)
    {
        List<ExecNode> nodes = ctx.Nodes;
        List<CValue> values = ctx.Values;
        for (int i = 0; i < nodes.Count; i++)
        {
            ExecNode pad = nodes[i];
            if (pad.Op != ExecOp.Pad || pad.P is not PadP pp || pp.Mode != PadMode.Constant)
                continue;
            int inId = pad.In[0], padOut = pad.Out[0];
            int rank = values[padOut].Shape.Length;
            if (rank != 4) continue;
            int padH = pp.Pads[2], padW = pp.Pads[3];
            if (pp.Pads[2 + rank] != padH || pp.Pads[3 + rank] != padW) continue;

            // channel split pieces of the padded tensor
            var pieces = nodes.Where(m => m.Op == ExecOp.Slice && m.In.Length == 1 && m.In[0] == padOut).ToList();
            if (pieces.Count != 5) continue;
            int[] padShape = values[padOut].Shape;
            // SliceP.Starts/Ends are per-Axes-position arrays. For a slice emitted from a Split
            // over axis C, every axes entry is identity except axis 1 (the group range).
            bool RangeAt(SliceP s, int ax, int[] inShape, out int st, out int en)
            {
                st = 0; en = inShape[ax];
                for (int k = 0; k < s.Axes.Length; k++)
                    if (s.Axes[k] == ax) { st = s.Starts[k]; en = s.Ends[k]; break; }
                return s.Axes.Where(a => a != ax).All(a =>
                    s.Starts[Array.IndexOf(s.Axes, a)] == 0 &&
                    s.Ends[Array.IndexOf(s.Axes, a)] >= inShape[a]);
            }
            var pieceInfo = pieces
                .Select(m => (node: m, s: (SliceP)m.P!))
                .Where(t => RangeAt(t.s, 1, padShape, out _, out _))
                .OrderBy(t => { RangeAt(t.s, 1, padShape, out int st, out _); return st; })
                .ToArray();
            if (pieceInfo.Length != 5) continue;
            int[] groupEnds = pieceInfo.Select(t => { RangeAt(t.s, 1, padShape, out _, out int en); return en; }).ToArray();

            // each piece's roll: two Slice consumers + one Concat
            int[] offsets = new int[5];
            int shiftAxis = -1;
            int[] pieceOutIds = pieceInfo.Select(t => t.node.Out[0]).ToArray();
            var smallConcatOuts = new List<int>(5);
            bool fail = false;
            var toRemove = new List<ExecNode>(pieces);
            for (int gI = 0; gI < 5; gI++)
            {
                int pieceId = pieceOutIds[gI];
                var rollSlices = nodes.Where(m => m.Op == ExecOp.Slice && m.In.Length == 1 && m.In[0] == pieceId).ToList();
                if (rollSlices.Count != 2) { fail = true; break; }
                var concat = nodes.FirstOrDefault(m => m.Op == ExecOp.Concat && m.In.Length == 2 &&
                    m.In.Contains(rollSlices[0].Out[0]) && m.In.Contains(rollSlices[1].Out[0]));
                if (concat is null) { fail = true; break; }
                var sp = (SliceP)rollSlices[0].P!; var sp2 = (SliceP)rollSlices[1].P!;
                // both roll slices must slice a single spatial axis
                int SliceAxis(SliceP s, int[] pieceShape)
                {
                    int found = -1;
                    for (int k = 0; k < s.Axes.Length; k++)
                    {
                        int d = s.Axes[k];
                        bool sliced = s.Starts[k] != 0 || s.Ends[k] < pieceShape[d];
                        if (sliced && found != -1) return -1;
                        if (sliced) found = d;
                    }
                    return found;
                }
                int[] pieceShape = values[pieceId].Shape;
                int ax = SliceAxis(sp, pieceShape);
                int ax2 = SliceAxis(sp2, pieceShape);
                // one side may be unsliced (identity roll for the 0-offset group)
                if (ax < 0) ax = ax2;
                if (ax2 >= 0 && ax2 != ax) ax = -1;
                if (ax < 0 || (ax != 2 && ax != 3)) { fail = true; break; }
                if (shiftAxis == -1) shiftAxis = ax;
                else if (shiftAxis != ax) { fail = true; break; }
                int L = pieceShape[ax];
                // concat order = [B, A] where A = piece[0..cut] is the "head"
                var sA = rollSlices[0].Out[0] == concat.In[0] ? sp2 : sp;
                var sB = rollSlices[0].Out[0] == concat.In[0] ? sp : sp2;
                RangeAt(sA, ax, pieceShape, out int saSt, out int saEn);
                RangeAt(sB, ax, pieceShape, out int sbSt, out int sbEn);
                if (saSt != 0 || sbSt != saEn || sbEn < L) { fail = true; break; }
                int cut = saEn - saSt;
                // effective shift on the unpadded axis: out[y] = orig[y + s], zero-fill.
                // rolled[i] = padded[(i+cut) mod L]; sliced[y] = rolled[y+padAmt]
                int padAmt = shiftAxis == 2 ? padH : padW;
                offsets[gI] = ((padAmt + cut) % L) - padAmt;
                toRemove.AddRange(rollSlices);
                toRemove.Add(concat);
                smallConcatOuts.Add(concat.Out[0]);
            }
            ExecNode? bigConcat = fail ? null : nodes.FirstOrDefault(m =>
                m.Op == ExecOp.Concat && m.In.Length == 5 && smallConcatOuts.All(m.In.Contains));
            if (fail || bigConcat is null) continue;
            var unpad = nodes.FirstOrDefault(m => m.Op == ExecOp.Slice && m.In.Length == 1 && m.In[0] == bigConcat.Out[0]);
            if (unpad is null) continue;
            toRemove.Add(bigConcat);
            toRemove.Add(unpad);
            toRemove.Add(pad);

            var shift = new ExecNode
            {
                Op = ExecOp.Ext,
                Ext = ChannelShiftOp.Instance,
                In = [inId],
                Out = unpad.Out,
                P = new ShiftP { Axis = shiftAxis, GroupEnds = groupEnds, Offsets = offsets },
                SrcNode = pad.SrcNode,
                Name = pad.Name + ".shift",
            };
            foreach (ExecNode m in toRemove) nodes.Remove(m);
            nodes.Insert(Math.Min(i, nodes.Count), shift);
        }
    }
}
