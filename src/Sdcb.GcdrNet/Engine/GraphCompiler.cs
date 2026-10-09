using Sdcb.GcdrNet.Onnx;

namespace Sdcb.GcdrNet.Engine;

/// <summary>
/// Compiles an <see cref="OnnxModel"/> into an <see cref="ExecPlan"/> for a
/// concrete input shape. All ONNX dynamic-shape plumbing (Shape/Squeeze/Div/
/// Mod/Concat/Slice on int64 scalars) is evaluated at compile time; remaining
/// float ops are emitted with fully resolved shapes.
/// </summary>
internal static class GraphCompiler
{
    public static ExecPlan Compile(OnnxModel model, ReadOnlySpan<int> inputShape)
    {
        var c = new Compiler(model);
        return c.Run(inputShape);
    }

    private sealed class Compiler
    {
        private readonly OnnxModel _m;
        private readonly List<CValue> _values = [];
        private readonly Dictionary<string, int> _byName = new(StringComparer.Ordinal);
        private readonly List<ExecNode> _nodes = [];
        private readonly List<int> _bufSize = [];
        private int _inputValue = -1;

        public Compiler(OnnxModel m) => _m = m;

        private int AddValue(CValue v)
        {
            int id = _values.Count;
            _values.Add(v);
            if (v.Name.Length != 0) _byName[v.Name] = id;
            return id;
        }

        private int NewBuf(int size)
        {
            int id = _bufSize.Count;
            _bufSize.Add(size);
            return id;
        }

        private CValue V(int id) => _values[id];

        private int Input(int i, OnnxNodeData n)
        {
            string name = n.Inputs[i];
            if (name.Length == 0) return -1;
            if (!_byName.TryGetValue(name, out int id))
                throw new InvalidDataException($"Node '{n.Name}' ({n.OpType}): input '{name}' not produced.");
            return id;
        }

        private int[] Inputs(OnnxNodeData n, int skip = 0)
        {
            int[] ids = new int[n.Inputs.Count - skip];
            for (int i = skip; i < n.Inputs.Count; i++) ids[i - skip] = Input(i, n);
            return ids;
        }

        private int _curNi;   // onnx node index currently being emitted (for SrcNode)

        private int Emit(ExecOp op, int[] ins, CValue output, object? p, OnnxNodeData n)
        {
            int outId = AddValue(output);
            _nodes.Add(new ExecNode { Op = op, In = ins, Out = [outId], P = p, SrcNode = _curNi, Name = n.Name });
            return outId;
        }

        public ExecPlan Run(ReadOnlySpan<int> inputShape)
        {
            OnnxGraphData g = _m.Graph;
            foreach (OnnxTensorData t in g.Initializers)
            {
                int[] dims = t.DimsArray();
                if (t.DataType == OnnxTensorData.TypeFloat)
                    AddValue(new CValue { Name = t.Name, Shape = dims, DType = 1, Floats = t.AsFloats() });
                else
                    AddValue(new CValue { Name = t.Name, Shape = dims, DType = t.DataType, Ints = t.AsInt64s() });
            }
            if (g.Inputs.Count != 1) throw new NotSupportedException("Expected single graph input.");
            {
                string inName = g.Inputs[0].Name;
                int elems = 1;
                int[] shape = inputShape.ToArray();
                foreach (int d in shape) elems *= d;
                int buf = NewBuf(elems);
                _inputValue = AddValue(new CValue { Name = inName, Shape = shape, DType = 1, Buf = buf });
            }

            for (int ni = 0; ni < g.Nodes.Count; ni++)
            {
                OnnxNodeData n = g.Nodes[ni];
                _curNi = ni;
                EmitNode(n);
            }

            // graph outputs
            if (g.Outputs.Count != 1) throw new NotSupportedException("Expected single graph output.");
            int outputValue = _byName[g.Outputs[0].Name];

            return FuseAndPlan(_nodes, _values, _bufSize, _inputValue, outputValue, _m);
        }

        // ---------- fusion + buffer planning ----------
        private static ExecPlan FuseAndPlan(List<ExecNode> nodes, List<CValue> values,
            List<int> bufSize, int inputValue, int outputValue, OnnxModel model)
        {
            if (Environment.GetEnvironmentVariable("GCDRNET_NOFUSE") != "1")
                Fuse(nodes, values);

            PlanArena(nodes, values, bufSize, inputValue, outputValue,
                out int[] offsets, out int arenaFloats);
            if (Environment.GetEnvironmentVariable("GCDRNET_DUMP") == "1")
            {
                var hist = nodes.GroupBy(m => m.Op).OrderByDescending(g => g.Count());
                Console.Error.WriteLine($"[plan] {nodes.Count} nodes, arena {arenaFloats * 4.0 / 1e6:F1} MB");
                foreach (var h in hist) Console.Error.WriteLine($"  {h.Key,-14} x{h.Count()}");
                foreach (var m in nodes.Where(m => m.Op is ExecOp.Relu or ExecOp.Elementwise or ExecOp.Transpose))
                {
                    var prod = nodes.Where(q => q.Out.Intersect(m.In).Any())
                        .Select(q => q.Op.ToString());
                    var sz = m.Out.Select(v => values[v].Shape.Aggregate(1, (a, b) => a * b));
                    Console.Error.WriteLine($"  {m.Op} {m.Name} <- [{string.Join(",", prod)}] n={string.Join(",", sz)}" +
                        (m.Op == ExecOp.Transpose && m.P is TransposeP tp
                            ? $" perm={string.Join(",", tp.Perm)} in={string.Join("x", values[m.In[0]].Shape)} out={string.Join("x", values[m.Out[0]].Shape)}"
                            : m.Op == ExecOp.Elementwise
                                ? $" a={string.Join("x", values[m.In[0]].Shape)} b={string.Join("x", values[m.In[1]].Shape)}"
                                : ""));
                }
            }
            return new ExecPlan
            {
                Nodes = nodes.ToArray(),
                Values = values.ToArray(),
                BufSize = bufSize.ToArray(),
                BufOffset = offsets,
                ArenaFloats = arenaFloats,
                InputValue = inputValue,
                OutputValue = outputValue,
            };
        }

        /// <summary>Structural fusions over the exec node list (single consumer chains).</summary>
        private static void Fuse(List<ExecNode> nodes, List<CValue> values)
        {
            // pass 1: Pad -> Conv  (fold reflect/constant padding into the conv)
            // pass 2: Conv -> Relu (activation epilogue)
            // pass 3: Div(x,√2) Erf Add(x,1) Mul(x,0.5) Mul(x0) -> Gelu
            // pass 4: Pad(+2) -> 5x Slice -> 10x Slice + 5x Concat -> Concat -> Slice
            //         (shift-MLP) -> ChannelShift
            for (int i = 0; i < nodes.Count; i++)
            {
                ExecNode n = nodes[i];
                if (n.Op == ExecOp.Conv && n.P is ConvP cp && cp.Pads.All(p => p == 0))
                {
                    int src = n.In[0];
                    ExecNode? pad = nodes.FirstOrDefault(m => m.Out.Length == 1 && m.Out[0] == src && m.Op == ExecOp.Pad);
                    if (pad != null && SoleUse(nodes, src, i) && pad.P is PadP pp &&
                        PadsOnlySpatial(pp.Pads, values[src].Shape.Length))
                    {
                        cp.Pads = [pp.Pads[2], pp.Pads[3], pp.Pads[2 + values[src].Shape.Length], pp.Pads[3 + values[src].Shape.Length]];
                        cp.PadMode = pp.Mode;
                        cp.PadValue = pp.Value;
                        n.In[0] = pad.In[0];
                        nodes.RemoveAt(nodes.IndexOf(pad));
                        i--;
                        continue;
                    }
                }
            }
            // Relu/Sigmoid/elementwise-tail fusions
            for (int i = 0; i + 1 < nodes.Count; i++)
            {
                ExecNode a = nodes[i], b = nodes[i + 1];
                if (a.Op == ExecOp.Conv && b.Op == ExecOp.Relu && b.In[0] == a.Out[0] &&
                    SoleUse(nodes, a.Out[0], i + 1))
                {
                    ((ConvP)a.P!).Act = ActivationKind.Relu;
                    // rewrite consumers of b.Out to use a.Out
                    Redirect(nodes, values, b.Out[0], a.Out[0]);
                    nodes.RemoveAt(i + 1);
                    i--;
                }
                else if (a.Op == ExecOp.MaxPool && b.Op == ExecOp.Relu && b.In[0] == a.Out[0] &&
                    SoleUse(nodes, a.Out[0], i + 1))
                {
                    ((PoolP)a.P!).FuseRelu = true;
                    Redirect(nodes, values, b.Out[0], a.Out[0]);
                    nodes.RemoveAt(i + 1);
                    i--;
                }
                else if (a.Op == ExecOp.ResizeBilinear && b.Op == ExecOp.Relu && b.In[0] == a.Out[0] &&
                    SoleUse(nodes, a.Out[0], i + 1))
                {
                    ((ResizeP)a.P!).FuseRelu = true;
                    Redirect(nodes, values, b.Out[0], a.Out[0]);
                    nodes.RemoveAt(i + 1);
                    i--;
                }
                else if (a.Op == ExecOp.Elementwise && b.Op == ExecOp.Relu && b.In[0] == a.Out[0] &&
                    SoleUse(nodes, a.Out[0], i + 1))
                {
                    ((EwP)a.P!).Act = ActivationKind.Relu;
                    Redirect(nodes, values, b.Out[0], a.Out[0]);
                    nodes.RemoveAt(i + 1);
                    i--;
                }
            }
            // Conv -> Add residual (skip connections): fold same-shape addend into conv epilogue
            for (int i = 0; i + 1 < nodes.Count; i++)
            {
                ExecNode a = nodes[i], b = nodes[i + 1];
                if (a.Op != ExecOp.Conv || b.Op != ExecOp.Elementwise ||
                    ((EwP)b.P!).Op != EwOp.Add ||
                    ((EwP)b.P!).Act is not (ActivationKind.None or ActivationKind.Relu))
                    continue;
                var cp = (ConvP)a.P!;
                if (cp.Act != ActivationKind.None || cp.Group > 1) continue; // depthwise shader has no residual path
                int outV = a.Out[0];
                int other = b.In[0] == outV ? b.In[1] : b.In[1] == outV ? b.In[0] : -1;
                if (other < 0 || !SoleUse(nodes, outV, i + 1)) continue;
                if (!values[other].Shape.SequenceEqual(values[outV].Shape)) continue;
                cp.AddValue = other;
                cp.Act = ((EwP)b.P!).Act;   // epilogue: +residual then activation (matches conv shader)
                Redirect(nodes, values, b.Out[0], outV);
                nodes.RemoveAt(i + 1);
                i--;
            }
            FuseGelu(nodes, values);
            FuseShift(nodes, values);
        }

        private static void Redirect(List<ExecNode> nodes, List<CValue> values, int oldValue, int newValue)
        {
            foreach (ExecNode m in nodes)
                for (int i = 0; i < m.In.Length; i++)
                    if (m.In[i] == oldValue) m.In[i] = newValue;
            if (values[oldValue].Name.Length != 0 && values[newValue].Name.Length == 0)
                values[newValue].Name = values[oldValue].Name;
        }

        /// <summary>True when value `v`'s only consumer is node index `useIdx`, once.</summary>
        private static bool SoleUse(List<ExecNode> nodes, int v, int useIdx)
        {
            int count = 0;
            for (int i = 0; i < nodes.Count; i++)
                foreach (int inp in nodes[i].In)
                    if (inp == v)
                    {
                        if (i != useIdx) return false;
                        if (++count > 1) return false;
                    }
            return count == 1;
        }

        private static bool PadsOnlySpatial(int[] pads, int rank)
        {
            if (pads.Length != 2 * rank) return false;
            for (int i = 0; i < 2; i++) if (pads[i] != 0 || pads[i + rank] != 0) return false;
            return true;
        }

        private static void FuseGelu(List<ExecNode> nodes, List<CValue> values)
        {
            // Div(X, sqrt2) -> Erf -> Add(+1) -> Mul(*0.5) -> Mul(*X)
            for (int i = 0; i + 4 < nodes.Count; i++)
            {
                ExecNode d = nodes[i], erf = nodes[i + 1], a1 = nodes[i + 2], mh = nodes[i + 3], mx = nodes[i + 4];
                if (d.Op != ExecOp.Elementwise || ((EwP)d.P!).Op != EwOp.Div ||
                    erf.Op != ExecOp.Erf ||
                    a1.Op != ExecOp.Elementwise || ((EwP)a1.P!).Op != EwOp.Add ||
                    mh.Op != ExecOp.Elementwise || ((EwP)mh.P!).Op != EwOp.Mul ||
                    mx.Op != ExecOp.Elementwise || ((EwP)mx.P!).Op != EwOp.Mul)
                    continue;
                if (erf.In[0] != d.Out[0] || a1.In[0] != erf.Out[0] && a1.In[1] != erf.Out[0] ||
                    (mh.In[0] != a1.Out[0] && mh.In[1] != a1.Out[0]) ||
                    (mx.In[0] != mh.Out[0] && mx.In[1] != mh.Out[0]))
                    continue;
                if (!IsScalarConst(values, d.In[1], MathF.Sqrt(2f))) continue;
                int a1Other = a1.In[0] == erf.Out[0] ? a1.In[1] : a1.In[0];
                int mhOther = mh.In[0] == a1.Out[0] ? mh.In[1] : mh.In[0];
                int mxOther = mx.In[0] == mh.Out[0] ? mx.In[1] : mx.In[0];
                if (!IsScalarConst(values, a1Other, 1f) || !IsScalarConst(values, mhOther, 0.5f)) continue;
                if (mxOther != d.In[0]) continue;
                // emit single Gelu
                values[mx.Out[0]].Shape = values[d.In[0]].Shape;
                nodes[i] = new ExecNode { Op = ExecOp.Gelu, In = [d.In[0]], Out = mx.Out, P = null, SrcNode = d.SrcNode, Name = d.Name };
                nodes.RemoveRange(i + 1, 4);
            }
        }

        private static bool IsScalarConst(List<CValue> values, int v, float expected)
        {
            CValue cv = values[v];
            return cv.Floats is { Length: 1 } f && MathF.Abs(f[0] - expected) < 1e-6f;
        }

        /// <summary>
        /// Recognizes the shift-MLP subgraph: Pad(const ±p on H or W) -> channel
        /// Split (5 groups via Slice) -> per-group roll (Slice+Slice+Concat on the
        /// spatial axis) -> Concat(axis=1) -> Slice(unpad). Replaced by a single
        /// ChannelShift node writing the shifted [1,C,h,w] tensor.
        /// </summary>
        private static void FuseShift(List<ExecNode> nodes, List<CValue> values)
        {
            for (int i = 0; i < nodes.Count; i++)
            {
                ExecNode pad = nodes[i];
                if (pad.Op != ExecOp.Pad || pad.P is not PadP pp || pp.Mode != PadMode.Constant)
                    continue;
                int inId = pad.In[0], padOut = pad.Out[0];
                int rank = values[padOut].Shape.Length;
                if (rank != 4) continue;
                // spatial pad amount (must be uniform ±p on exactly one spatial axis pair? ours: both H,W ±2)
                int padH = pp.Pads[2], padW = pp.Pads[3];
                if (pp.Pads[2 + rank] != padH || pp.Pads[3 + rank] != padW) continue;

                // find all consumer slices of padOut (channel split pieces)
                var pieces = nodes.Where(m => m.Op == ExecOp.Slice && m.In.Length == 1 && m.In[0] == padOut).ToList();
                if (Environment.GetEnvironmentVariable("GCDRNET_DUMP") == "1")
                    Console.Error.WriteLine($"[shift?] pad {pad.Name} pieces={pieces.Count}");
                if (pieces.Count != 5) continue;
                int[] padShape = values[padOut].Shape;
                // SliceP.Starts/Ends are per-Axes-position arrays (len == Axes.Length).
                // For a slice emitted from a Split over axis C, every axes entry is
                // identity except the one at axis 1, which carries the group range.
                bool RangeAt(SliceP s, int ax, int[] inShape, out int st, out int en)
                {
                    st = 0; en = inShape[ax];
                    for (int k = 0; k < s.Axes.Length; k++)
                        if (s.Axes[k] == ax) { st = s.Starts[k]; en = s.Ends[k]; break; }
                    return s.Axes.Where((a, k) => a != ax).All(a =>
                        s.Starts[Array.IndexOf(s.Axes, a)] == 0 &&
                        s.Ends[Array.IndexOf(s.Axes, a)] >= inShape[a]);
                }
                var pieceInfo = pieces
                    .Select(m => (node: m, s: (SliceP)m.P!))
                    .Where(t => RangeAt(t.s, 1, padShape, out _, out _))
                    .OrderBy(t => { RangeAt(t.s, 1, padShape, out int st, out _); return st; })
                    .ToArray();
                if (pieceInfo.Length != 5)
                {
                    if (Environment.GetEnvironmentVariable("GCDRNET_DUMP") == "1")
                        Console.Error.WriteLine($"  pieceInfo={pieceInfo.Length} (OnlyAxis fail)");
                    continue;
                }
                int[] groupEnds = pieceInfo.Select(t =>
                {
                    RangeAt(t.s, 1, padShape, out _, out int en); return en;
                }).ToArray();

                // for each piece find its roll: two Slice consumers + one Concat
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
                    if (Environment.GetEnvironmentVariable("GCDRNET_DUMP") == "1")
                    {
                        var allCons = nodes.Where(m => m.In.Contains(pieceId))
                            .Select(m => m.Op.ToString());
                        Console.Error.WriteLine($"  g{gI} piece {pieceId} consumers=[{string.Join(",", allCons)}] rollSlices={rollSlices.Count}");
                    }
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
                    // allow one side to be unsliced (identity roll for the 0-offset group)
                    if (ax < 0) ax = ax2;
                    if (ax2 >= 0 && ax2 != ax) ax = -1;
                    if (ax < 0 || (ax != 2 && ax != 3))
                    {
                        if (Environment.GetEnvironmentVariable("GCDRNET_DUMP") == "1")
                            Console.Error.WriteLine($"    ax={ax} ax2={ax2} s0.Axes=[{string.Join(",", sp.Axes)}] s1.Axes=[{string.Join(",", sp2.Axes)}]");
                        fail = true; break;
                    }
                    if (shiftAxis == -1) shiftAxis = ax;
                    else if (shiftAxis != ax) { fail = true; break; }
                    int L = pieceShape[ax];
                    // concat order = [B, A] where A = piece[0..cut] is the "head"
                    var sA = rollSlices[0].Out[0] == concat.In[0] ? sp2 : sp;
                    var sB = rollSlices[0].Out[0] == concat.In[0] ? sp : sp2;
                    RangeAt(sA, ax, pieceShape, out int saSt, out int saEn);
                    RangeAt(sB, ax, pieceShape, out int sbSt, out int sbEn);
                    // sanity: A is the head [0..cut], B is the tail [cut..L]
                    if (saSt != 0 || sbSt != saEn || sbEn < L) { fail = true; break; }
                    int cut = saEn - saSt;
                    // effective shift on UNPADDED axis: out[y] = orig[y + s], zero-fill.
                    // rolled[i] = padded[(i+cut) mod L]; sliced[y] = rolled[y+padAmt]
                    //   = padded[(y+padAmt+cut) mod L]; orig coord = that - padAmt.
                    int dim = values[pad.In[0]].Shape[shiftAxis];
                    int padAmt = shiftAxis == 2 ? padH : padW;
                    offsets[gI] = ((padAmt + cut) % L) - padAmt;
                    toRemove.AddRange(rollSlices);
                    toRemove.Add(concat);
                    smallConcatOuts.Add(concat.Out[0]);
                }
                ExecNode? bigConcat = fail ? null : nodes.FirstOrDefault(m =>
                    m.Op == ExecOp.Concat && m.In.Length == 5 &&
                    smallConcatOuts.All(m.In.Contains));
                if (fail || bigConcat is null)
                {
                    if (Environment.GetEnvironmentVariable("GCDRNET_DUMP") == "1")
                    {
                        Console.Error.WriteLine($"  fail={fail} bigConcat={(bigConcat is null ? "null" : "ok")} (gI loop)");
                        foreach (var cc in nodes.Where(m => m.Op == ExecOp.Concat))
                            Console.Error.WriteLine($"    concat in[{string.Join("+", cc.In.Select(v => v.ToString()))}] outs=[{string.Join("+", cc.Out.Select(v => v.ToString()))}]");
                    }
                    continue;
                }
                // unpad slice consuming bigConcat output
                var unpad = nodes.FirstOrDefault(m => m.Op == ExecOp.Slice && m.In.Length == 1 && m.In[0] == bigConcat.Out[0]);
                if (unpad is null)
                {
                    if (Environment.GetEnvironmentVariable("GCDRNET_DUMP") == "1")
                        Console.Error.WriteLine($"  unpad missing for concat out {bigConcat.Out[0]}");
                    continue;
                }
                toRemove.Add(bigConcat);
                toRemove.Add(unpad);
                toRemove.Add(pad);

                var shift = new ExecNode
                {
                    Op = ExecOp.ChannelShift,
                    In = [inId],
                    Out = unpad.Out,
                    P = new ShiftP { Axis = shiftAxis, GroupEnds = groupEnds, Offsets = offsets },
                    SrcNode = pad.SrcNode,
                    Name = pad.Name + ".shift",
                };
                if (Environment.GetEnvironmentVariable("GCDRNET_DUMP") == "1")
                    Console.Error.WriteLine($"[shift!] {pad.Name} axis={shiftAxis} ends=[{string.Join(",", groupEnds)}] offs=[{string.Join(",", offsets)}]");
                foreach (ExecNode m in toRemove) nodes.Remove(m);
                nodes.Insert(Math.Min(i, nodes.Count), shift);
            }
        }

        /// <summary>Lifetime-based arena packing (def-order, free-list reuse).</summary>
        private static void PlanArena(List<ExecNode> nodes, List<CValue> values,
            List<int> bufSize, int inputValue, int outputValue,
            out int[] offsets, out int arenaFloats)
        {
            int bufCount = bufSize.Count;
            const int Align = 16; // floats (64B)
            int AlignUp(int v) => (v + Align - 1) & ~(Align - 1);

            int[] def = Enumerable.Repeat(int.MaxValue, bufCount).ToArray();
            int[] use = Enumerable.Repeat(-1, bufCount).ToArray();
            for (int ni = 0; ni < nodes.Count; ni++)
            {
                ExecNode n = nodes[ni];
                foreach (int iv in n.In)
                {
                    int b = values[iv].Buf;
                    if (b >= 0) { use[b] = Math.Max(use[b], ni); def[b] = Math.Min(def[b], ni); }
                }
                // value ids carried in params (conv bias/residual, gemm bias) count as uses
                int aux = n.P switch
                {
                    ConvP c => Math.Max(c.BiasValue, c.AddValue),
                    GemmP g => g.BiasValue,
                    _ => -1,
                };
                if (aux >= 0)
                {
                    // note: for ConvP, BiasValue and AddValue may differ — scan both
                    if (n.P is ConvP cp2)
                        foreach (int iv in new[] { cp2.BiasValue, cp2.AddValue })
                        {
                            if (iv < 0) continue;
                            int b = values[iv].Buf;
                            if (b >= 0) { use[b] = Math.Max(use[b], ni); def[b] = Math.Min(def[b], ni); }
                        }
                    else
                    {
                        int b = values[aux].Buf;
                        if (b >= 0) { use[b] = Math.Max(use[b], ni); def[b] = Math.Min(def[b], ni); }
                    }
                }
                foreach (int ov in n.Out)
                {
                    int b = values[ov].Buf;
                    if (b >= 0) def[b] = Math.Min(def[b], ni);
                }
            }
            int inputBuf = values[inputValue].Buf;
            int outputBuf = values[outputValue].Buf;
            def[inputBuf] = -1;
            // keep input + output live for the whole run: callers may read
            // Input (e.g. two-stage pipelines reusing it) after Run().
            use[inputBuf] = nodes.Count;
            use[outputBuf] = nodes.Count;

            var order = Enumerable.Range(0, bufCount)
                .Where(b => def[b] != int.MaxValue)
                .OrderBy(b => def[b]).ThenByDescending(b => bufSize[b]).ToList();
            var active = new List<int>();   // placed buf ids still live
            var free = new List<(int off, int size)>();
            offsets = new int[bufCount];
            Array.Fill(offsets, -1);
            int bump = 0;
            foreach (int b in order)
            {
                // release dead bufs before allocating this one
                for (int i = active.Count - 1; i >= 0; i--)
                {
                    int ob = active[i];
                    if (use[ob] < def[b])
                    {
                        active.RemoveAt(i);
                        free.Add((offsets[ob], AlignUp(bufSize[ob])));
                    }
                }
                int need = AlignUp(bufSize[b]);
                int bestIdx = -1;
                for (int fi = 0; fi < free.Count; fi++)
                    if (free[fi].size >= need && (bestIdx < 0 || free[fi].size < free[bestIdx].size))
                        bestIdx = fi;
                if (bestIdx >= 0)
                {
                    var chunk = free[bestIdx];
                    offsets[b] = chunk.off;
                    free.RemoveAt(bestIdx);
                    if (chunk.size > need) free.Add((chunk.off + need, chunk.size - need));
                }
                else
                {
                    offsets[b] = bump;
                    bump += need;
                }
                active.Add(b);
            }
            arenaFloats = bump;
        }

        private static readonly HashSet<string> IntOps = new(StringComparer.Ordinal)
            { "Squeeze", "Unsqueeze", "Div", "Mod", "Add", "Sub", "Mul", "Reshape",
              "Concat", "Slice", "Gather", "Cast", "Floor", "Ceil", "Neg", "Abs",
              "Min", "Max", "Equal", "Less", "Where", "ConstantOfShape", "Expand",
              "Identity", "Tile" };

        private void EmitNode(OnnxNodeData n)
        {
            string op = n.OpType;
            // Shape → int tensor of input dims
            if (op == "Shape")
            {
                CValue x = V(Input(0, n));
                if (x.Shape is null) throw new InvalidDataException("Shape input has no known shape.");
                int start = (int)n.IntAttr("start", 0);
                int end = (int)n.IntAttr("end", x.Shape.Length);
                int rank = x.Shape.Length;
                if (start < 0) start += rank;
                if (end < 0) end += rank;
                start = Math.Clamp(start, 0, rank);
                end = Math.Clamp(end, 0, rank);
                long[] dims = x.Shape[start..end].Select(d => (long)d).ToArray();
                AddValue(new CValue { Name = n.Outputs[0], Shape = [dims.Length], DType = 7, Ints = dims });
                return;
            }
            if (op == "Size")
            {
                CValue x = V(Input(0, n));
                AddValue(new CValue { Name = n.Outputs[0], Shape = [], DType = 7, Ints = [x.ElementCount] });
                return;
            }
            if (op == "Constant")
            {
                OnnxAttributeData? a = n.Attr("value");
                if (a?.Tensor is OnnxTensorData t)
                {
                    if (t.DataType == OnnxTensorData.TypeFloat)
                        AddValue(new CValue { Name = n.Outputs[0], Shape = t.DimsArray(), DType = 1, Floats = t.AsFloats() });
                    else
                        AddValue(new CValue { Name = n.Outputs[0], Shape = t.DimsArray(), DType = t.DataType, Ints = t.AsInt64s() });
                    return;
                }
                throw new NotSupportedException("Constant without tensor value.");
            }

            // int-only evaluation: if every present input is a const int tensor
            // and the op is elementwise/shape math, evaluate now.
            bool allInt = n.Inputs.All(s => s.Length == 0 || (V(_byName[s]).IsInt));
            if (allInt && IntOps.Contains(op))
            {
                EvalInt(n);
                return;
            }
            // Cast on int → still int
            if (op == "Cast" && V(Input(0, n)).IsInt)
            {
                int to = (int)n.IntAttr("to");
                CValue x = V(Input(0, n));
                if (to == 1)
                    AddValue(new CValue { Name = n.Outputs[0], Shape = x.Shape, DType = 1, Floats = x.Ints!.Select(v => (float)v).ToArray() });
                else
                    AddValue(new CValue { Name = n.Outputs[0], Shape = x.Shape, DType = 7, Ints = x.Ints });
                return;
            }

            // ---- float ops ----
            switch (op)
            {
                case "Pad": EmitPad(n); break;
                case "Conv": EmitConv(n); break;
                case "MaxPool": EmitPool(n); break;
                case "Relu": EmitUnary(n, ExecOp.Relu); break;
                case "Sigmoid": EmitUnary(n, ExecOp.Sigmoid); break;
                case "Erf": EmitUnary(n, ExecOp.Erf); break;
                case "LayerNormalization": EmitLayerNorm(n); break;
                case "Gemm": EmitGemm(n); break;
                case "Resize": EmitResize(n); break;
                case "Transpose": EmitTranspose(n); break;
                case "Slice": EmitSlice(n); break;
                case "Concat": EmitConcat(n); break;
                case "Split": EmitSplit(n); break;
                case "Reshape": EmitReshape(n); break;
                case "Squeeze": EmitSqueeze(n); break;
                case "Unsqueeze": EmitUnsqueeze(n); break;
                case "Add" or "Sub" or "Mul" or "Div": EmitElementwise(n, op); break;
                case "Identity": Alias(n); break;
                default:
                    throw new NotSupportedException($"Op '{op}' not supported (node '{n.Name}').");
            }
        }

        // ---------- int evaluation ----------
        private void EvalInt(OnnxNodeData n)
        {
            string op = n.OpType;
            CValue In(int i) => V(Input(i, n));
            CValue Out(long[] data, int[] shape)
            {
                var v = new CValue { Name = n.Outputs[0], Shape = shape, DType = 7, Ints = data };
                AddValue(v);
                return v;
            }
            switch (op)
            {
                case "Squeeze":
                {
                    CValue x = In(0);
                    long[]? axes = n.Inputs.Count > 1 && n.Inputs[1].Length > 0 ? In(1).Ints : null;
                    int[] shape;
                    if (axes is null)
                        shape = x.Shape.Where(d => d != 1).ToArray();
                    else
                    {
                        var dims = x.Shape.ToList();
                        foreach (long a in axes.OrderByDescending(v => v))
                        {
                            int ax = (int)(a < 0 ? a + dims.Count : a);
                            if (dims[ax] != 1) throw new InvalidDataException("Squeeze axis not 1.");
                            dims.RemoveAt(ax);
                        }
                        shape = dims.ToArray();
                    }
                    Out(x.Ints!, shape);
                    return;
                }
                case "Unsqueeze":
                {
                    CValue x = In(0);
                    long[] axes = In(1).Ints!;
                    var dims = x.Shape.ToList();
                    foreach (long a in axes.OrderBy(v => v))
                    {
                        int ax = (int)(a < 0 ? a + dims.Count + 1 : a);
                        dims.Insert(ax, 1);
                    }
                    Out(x.Ints!, dims.ToArray());
                    return;
                }
                case "Reshape":
                {
                    CValue x = In(0);
                    long[] spec = In(1).Ints!;
                    bool allowZero = n.IntAttr("allowzero") == 1;
                    Out(x.Ints!, ResolveReshape(x.Shape, spec, allowZero, x.Ints!.Length));
                    return;
                }
                case "Concat":
                {
                    int axis = (int)n.IntAttr("axis");
                    var inputs = Inputs(n).Select(V).ToArray();
                    int rank = inputs[0].Shape.Length;
                    if (axis < 0) axis += rank;
                    int[] shape = (int[])inputs[0].Shape.Clone();
                    shape[axis] = inputs.Sum(v => v.Shape[axis]);
                    long[] data = IntConcat(inputs.Select(v => v.Ints!).ToArray(),
                        inputs.Select(v => v.Shape).ToArray(), axis, shape);
                    Out(data, shape);
                    return;
                }
                case "Slice":
                {
                    CValue x = In(0);
                    long[] starts = In(1).Ints!;
                    long[] ends = n.Inputs.Count > 2 && n.Inputs[2].Length > 0 ? In(2).Ints! : x.Shape.Select(d => (long)d).ToArray();
                    long[] axes = n.Inputs.Count > 3 && n.Inputs[3].Length > 0 ? In(3).Ints! : Enumerable.Range(0, starts.Length).Select(i => (long)i).ToArray();
                    long[] steps = n.Inputs.Count > 4 && n.Inputs[4].Length > 0 ? In(4).Ints! : starts.Select(_ => 1L).ToArray();
                    SliceBounds(x.Shape, starts, ends, axes, steps,
                        out int[] outShape, out int[] st, out int[] en, out int[] stepI);
                    long[] data = IntSlice(x.Ints!, x.Shape,
                        axes.Select(a => (int)(a < 0 ? a + x.Shape.Length : a)).ToArray(), st, stepI, outShape);
                    Out(data, outShape);
                    return;
                }
                case "Gather":
                {
                    CValue x = In(0); CValue idx = In(1);
                    int axis = (int)n.IntAttr("axis");
                    if (axis < 0) axis += x.Shape.Length;
                    if (x.Shape.Length != 1 || axis != 0) throw new NotSupportedException("Gather eval supports 1-D axis 0.");
                    Out(idx.Ints!.Select(i => x.Ints![checked((int)i)]).ToArray(), idx.Shape);
                    return;
                }
                case "Identity":
                {
                    CValue x = In(0);
                    Out(x.Ints!, x.Shape);
                    return;
                }
                case "Add" or "Sub" or "Mul" or "Div" or "Mod" or "Min" or "Max":
                {
                    long[] a = In(0).Ints!, b = In(1).Ints!;
                    long[] r = new long[Math.Max(a.Length, b.Length)];
                    for (int i = 0; i < r.Length; i++)
                    {
                        long av = a.Length == 1 ? a[0] : a[i], bv = b.Length == 1 ? b[0] : b[i];
                        r[i] = op switch
                        {
                            "Add" => av + bv, "Sub" => av - bv, "Mul" => av * bv,
                            "Div" => av / bv, "Mod" => ((av % bv) + bv) % bv,
                            "Min" => Math.Min(av, bv), _ => Math.Max(av, bv),
                        };
                    }
                    int[] shape = In(0).Shape.Length >= In(1).Shape.Length ? In(0).Shape : In(1).Shape;
                    Out(r, shape);
                    return;
                }
                case "Neg" or "Abs" or "Floor" or "Ceil":
                {
                    long[] a = In(0).Ints!;
                    // Floor/Ceil on ints are identity
                    long[] r = a.Select(v => op == "Neg" ? -v : op == "Abs" ? Math.Abs(v) : v).ToArray();
                    Out(r, In(0).Shape);
                    return;
                }
                default:
                    throw new NotSupportedException($"Int eval for '{op}' not implemented (node '{n.Name}').");
            }
        }

        // ---------- float ops ----------
        private CValue ActivationValue(string name, int[] shape)
        {
            int elems = 1;
            foreach (int d in shape) elems = checked(elems * d);
            return new CValue { Name = name, Shape = shape, DType = 1, Buf = NewBuf(elems) };
        }

        private void Alias(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            AddValue(new CValue { Name = n.Outputs[0], Shape = x.Shape, DType = 1, Buf = x.Buf, Floats = x.Floats, Ints = x.Ints });
        }

        private void EmitPad(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            long[] pads = V(Input(1, n)).Ints!;
            float val = n.Inputs.Count > 2 && n.Inputs[2].Length > 0 ? V(Input(2, n)).Floats![0] : 0f;
            PadMode mode = n.StringAttr("mode", "constant") switch
            {
                "reflect" => PadMode.Reflect,
                "constant" => PadMode.Constant,
                var s => throw new NotSupportedException($"Pad mode '{s}'"),
            };
            int[] shape = x.Shape.ToArray();
            for (int i = 0; i < shape.Length; i++)
                shape[i] += checked((int)(pads[i] + pads[i + shape.Length]));
            Emit(ExecOp.Pad, [Input(0, n)], ActivationValue(n.Outputs[0], shape),
                new PadP { Pads = pads.Select(p => (int)p).ToArray(), Mode = mode, Value = val }, n);
        }

        private void EmitConv(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            CValue w = V(Input(1, n));
            int[] wd = w.Shape;
            int[] pads = n.IntsAttr("pads").Select(v => (int)v).ToArray();
            if (pads.Length == 0) pads = new int[2 * (wd.Length - 2)];
            int[] strides = n.IntsAttr("strides").Select(v => (int)v).ToArray();
            if (strides.Length == 0) strides = Enumerable.Repeat(1, wd.Length - 2).ToArray();
            int[] dil = n.IntsAttr("dilations").Select(v => (int)v).ToArray();
            if (dil.Length == 0) dil = Enumerable.Repeat(1, wd.Length - 2).ToArray();
            int group = (int)n.IntAttr("group", 1);
            int[] outShape = new int[wd.Length];
            outShape[0] = x.Shape[0];
            outShape[1] = wd[0];
            for (int d = 0; d < wd.Length - 2; d++)
            {
                int eff = dil[d] * (wd[2 + d] - 1) + 1;
                outShape[2 + d] = (x.Shape[2 + d] + pads[d] + pads[d + wd.Length - 2] - eff) / strides[d] + 1;
            }
            int biasV = n.Inputs.Count > 2 && n.Inputs[2].Length > 0 ? Input(2, n) : -1;
            Emit(ExecOp.Conv, [Input(0, n), Input(1, n)], ActivationValue(n.Outputs[0], outShape),
                new ConvP
                {
                    Pads = pads, Strides = strides, Dilations = dil,
                    Kernel = wd[2..], Group = group, BiasValue = biasV,
                }, n);
        }

        private void EmitPool(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            int[] kernel = n.IntsAttr("kernel_shape").Select(v => (int)v).ToArray();
            int[] strides = n.IntsAttr("strides").Select(v => (int)v).ToArray();
            if (strides.Length == 0) strides = (int[])kernel.Clone();
            int[] pads = n.IntsAttr("pads").Select(v => (int)v).ToArray();
            if (pads.Length == 0) pads = new int[2 * kernel.Length];
            int[] outShape = (int[])x.Shape.Clone();
            for (int d = 0; d < kernel.Length; d++)
                outShape[2 + d] = (x.Shape[2 + d] + pads[d] + pads[d + kernel.Length] - kernel[d]) / strides[d] + 1;
            Emit(ExecOp.MaxPool, [Input(0, n)], ActivationValue(n.Outputs[0], outShape),
                new PoolP { Kernel = kernel, Strides = strides, Pads = pads }, n);
        }

        private void EmitUnary(OnnxNodeData n, ExecOp op)
        {
            CValue x = V(Input(0, n));
            Emit(op, [Input(0, n)], ActivationValue(n.Outputs[0], x.Shape), null, n);
        }

        private void EmitLayerNorm(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            float eps = n.FloatAttr("epsilon", 1e-5f);
            int axis = (int)n.IntAttr("axis", -1);
            if (axis < 0) axis += x.Shape.Length;
            if (axis != x.Shape.Length - 1)
                throw new NotSupportedException("LayerNorm only over last dim.");
            Emit(ExecOp.LayerNorm, [Input(0, n), Input(1, n), Input(2, n)],
                ActivationValue(n.Outputs[0], x.Shape), new LnP { Eps = eps }, n);
        }

        private void EmitGemm(OnnxNodeData n)
        {
            CValue a = V(Input(0, n));
            CValue b = V(Input(1, n));
            bool transB = n.IntAttr("transB") == 1;
            float alpha = n.FloatAttr("alpha", 1f);
            if (alpha != 1f) throw new NotSupportedException("Gemm alpha != 1.");
            int k = transB ? b.Shape[1] : b.Shape[0];
            int ncol = transB ? b.Shape[0] : b.Shape[1];
            if (a.Shape[^1] != k) throw new InvalidDataException("Gemm K mismatch.");
            int[] outShape = (int[])a.Shape.Clone();
            outShape[^1] = ncol;
            int biasV = n.Inputs.Count > 2 && n.Inputs[2].Length > 0 ? Input(2, n) : -1;
            Emit(ExecOp.Gemm, [Input(0, n), Input(1, n)], ActivationValue(n.Outputs[0], outShape),
                new GemmP { TransB = transB, BiasValue = biasV }, n);
        }

        private void EmitResize(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            float[] scales;
            int[]? sizesI = null;
            if (n.Inputs.Count > 3 && n.Inputs[3].Length > 0)
            {
                // sizes input (int tensor): authoritative output shape
                long[] sizes = V(Input(3, n)).Ints!;
                sizesI = sizes.Select(s => (int)s).ToArray();
                scales = sizes.Select((s, i) => (float)s / x.Shape[i]).ToArray();
            }
            else
            {
                scales = V(Input(2, n)).Floats!;
            }
            if (n.StringAttr("mode") != "linear")
                throw new NotSupportedException("Resize mode != linear.");
            int[] outShape = sizesI ?? (int[])x.Shape.Clone();
            if (sizesI == null)
                for (int i = 0; i < outShape.Length; i++)
                    outShape[i] = (int)MathF.Floor(x.Shape[i] * scales[i]);
            Emit(ExecOp.ResizeBilinear, [Input(0, n)], ActivationValue(n.Outputs[0], outShape),
                new ResizeP { Scales = scales }, n);
        }

        private void EmitTranspose(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            int[] perm = n.IntsAttr("perm").Select(v => (int)v).ToArray();
            if (perm.Length == 0) perm = Enumerable.Range(0, x.Shape.Length).Reverse().ToArray();
            int[] outShape = perm.Select(p => x.Shape[p]).ToArray();
            Emit(ExecOp.Transpose, [Input(0, n)], ActivationValue(n.Outputs[0], outShape),
                new TransposeP { Perm = perm }, n);
        }

        private void EmitSlice(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            long[] starts = V(Input(1, n)).Ints!;
            long[] ends = V(Input(2, n)).Ints!;
            long[] axes = n.Inputs.Count > 3 && n.Inputs[3].Length > 0
                ? V(Input(3, n)).Ints! : Enumerable.Range(0, starts.Length).Select(i => (long)i).ToArray();
            long[] steps = n.Inputs.Count > 4 && n.Inputs[4].Length > 0
                ? V(Input(4, n)).Ints! : starts.Select(_ => 1L).ToArray();
            SliceBounds(x.Shape, starts, ends, axes, steps,
                out int[] outShape, out int[] st, out int[] en, out int[] stepI);
            Emit(ExecOp.Slice, [Input(0, n)], ActivationValue(n.Outputs[0], outShape),
                new SliceP { Starts = st, Ends = en,
                    Axes = axes.Select(a => (int)a).ToArray(), Steps = stepI }, n);
        }

        private void EmitConcat(OnnxNodeData n)
        {
            int axis = (int)n.IntAttr("axis");
            int[] ins = Inputs(n);
            int[] shape = (int[])V(ins[0]).Shape.Clone();
            int rank = shape.Length;
            if (axis < 0) axis += rank;
            shape[axis] = ins.Sum(id => V(id).Shape[axis]);
            Emit(ExecOp.Concat, ins, ActivationValue(n.Outputs[0], shape),
                new ConcatP { Axis = axis }, n);
        }

        private void EmitSplit(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            int axis = (int)n.IntAttr("axis");
            if (axis < 0) axis += x.Shape.Length;
            int[] sizes;
            if (n.Inputs.Count > 1 && n.Inputs[1].Length > 0)
                sizes = V(Input(1, n)).Ints!.Select(v => (int)v).ToArray();
            else
            {
                int outputs = (int)n.IntAttr("num_outputs", n.Outputs.Count);
                int each = x.Shape[axis] / outputs;
                sizes = Enumerable.Repeat(each, outputs).ToArray();
                sizes[^1] += x.Shape[axis] - each * outputs;
            }
            // Split is implemented as a per-output Slice at exec time.
            int[] starts = new int[x.Shape.Length];
            for (int i = 0; i < sizes.Length; i++)
            {
                int[] shape = (int[])x.Shape.Clone();
                shape[axis] = sizes[i];
                var slice = new SliceP
                {
                    Starts = (int[])starts.Clone(),
                    Ends = (int[])x.Shape.Clone(),
                    Axes = Enumerable.Range(0, x.Shape.Length).ToArray(),
                    Steps = Enumerable.Repeat(1, x.Shape.Length).ToArray(),
                };
                slice.Ends[axis] = starts[axis] + sizes[i];
                _nodes.Add(new ExecNode { Op = ExecOp.Slice, In = [Input(0, n)], Out = [AddValue(ActivationValue(n.Outputs[i], shape))], P = slice, SrcNode = _nodes.Count, Name = n.Name });
                starts[axis] += sizes[i];
            }
        }

        private void EmitReshape(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            long[] spec = V(Input(1, n)).Ints!;
            bool allowZero = n.IntAttr("allowzero") == 1;
            int[] shape = ResolveReshape(x.Shape, spec, allowZero, checked((int)x.ElementCount));
            // zero-copy alias (consts keep their payload so consumers see a materialized tensor)
            AddValue(new CValue { Name = n.Outputs[0], Shape = shape, DType = 1, Buf = x.Buf, Floats = x.Floats });
        }

        private void EmitSqueeze(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            long[]? axes = n.Inputs.Count > 1 && n.Inputs[1].Length > 0 ? V(Input(1, n)).Ints : null;
            int[] shape;
            if (axes is null) shape = x.Shape.Where(d => d != 1).ToArray();
            else
            {
                var dims = x.Shape.ToList();
                foreach (long a in axes.OrderByDescending(v => v)) dims.RemoveAt((int)(a < 0 ? a + dims.Count : a));
                shape = dims.ToArray();
            }
            AddValue(new CValue { Name = n.Outputs[0], Shape = shape, DType = 1, Buf = x.Buf, Floats = x.Floats });
        }

        private void EmitUnsqueeze(OnnxNodeData n)
        {
            CValue x = V(Input(0, n));
            long[] axes = V(Input(1, n)).Ints!;
            var dims = x.Shape.ToList();
            foreach (long a in axes.OrderBy(v => v)) dims.Insert((int)(a < 0 ? a + dims.Count + 1 : a), 1);
            AddValue(new CValue { Name = n.Outputs[0], Shape = dims.ToArray(), DType = 1, Buf = x.Buf, Floats = x.Floats });
        }

        private void EmitElementwise(OnnxNodeData n, string op)
        {
            int a = Input(0, n), b = Input(1, n);
            CValue va = V(a), vb = V(b);
            int[] shape = BroadcastShape(va.Shape, vb.Shape);
            var eop = op switch { "Add" => EwOp.Add, "Sub" => EwOp.Sub, "Mul" => EwOp.Mul, "Div" => EwOp.Div, _ => throw new() };
            Emit(ExecOp.Elementwise, [a, b], ActivationValue(n.Outputs[0], shape), new EwP { Op = eop }, n);
        }

        // ---------- helpers ----------
        private static int[] BroadcastShape(int[] a, int[] b)
        {
            int rank = Math.Max(a.Length, b.Length);
            int[] r = new int[rank];
            for (int i = 0; i < rank; i++)
            {
                int ad = i - (rank - a.Length) >= 0 ? a[i - (rank - a.Length)] : 1;
                int bd = i - (rank - b.Length) >= 0 ? b[i - (rank - b.Length)] : 1;
                if (ad != bd && ad != 1 && bd != 1)
                    throw new InvalidDataException($"Broadcast mismatch {ad} vs {bd}.");
                r[i] = Math.Max(ad, bd);
            }
            return r;
        }

        internal static int[] ResolveReshape(int[] inShape, long[] spec, bool allowZero, int inElems)
        {
            int[] shape = new int[spec.Length];
            int neg = -1; long prod = 1;
            for (int i = 0; i < spec.Length; i++)
            {
                long s = spec[i];
                if (s == -1) { neg = i; shape[i] = 1; }
                else if (s == 0 && !allowZero) { shape[i] = inShape[i]; prod *= shape[i]; }
                else { shape[i] = checked((int)s); prod *= shape[i]; }
            }
            if (neg >= 0) shape[neg] = checked((int)(inElems / prod));
            return shape;
        }

        internal static void SliceBounds(int[] inShape, long[] starts, long[] ends,
            long[] axes, long[] steps, out int[] outShape, out int[] stOut, out int[] enOut, out int[] stepOut)
        {
            int rank = inShape.Length;
            stOut = new int[starts.Length];
            enOut = new int[starts.Length];
            stepOut = new int[starts.Length];
            outShape = (int[])inShape.Clone();
            for (int i = 0; i < starts.Length; i++)
            {
                int ax = (int)(axes[i] < 0 ? axes[i] + rank : axes[i]);
                int dim = inShape[ax];
                int step = (int)steps[i];
                long s = starts[i] < 0 ? starts[i] + dim : starts[i];
                long e = ends[i] < 0 ? ends[i] + dim : ends[i];
                if (step > 0)
                {
                    s = Math.Clamp(s, 0, dim);
                    e = Math.Clamp(e, 0, dim);
                }
                else
                {
                    s = Math.Clamp(s, -1, dim - 1);
                    e = Math.Clamp(e, -1, dim - 1);
                }
                int len = step > 0 ? Math.Max(0, (int)(e - s + step - 1) / step)
                                   : Math.Max(0, (int)(s - e - step - 1) / -step);
                stOut[i] = (int)s;
                enOut[i] = (int)e;
                stepOut[i] = step;
                outShape[ax] = len;
            }
        }

        private static long[] IntConcat(long[][] inputs, int[][] shapes, int axis, int[] outShape)
        {
            // small tensors only — generic n-d concat
            int outElems = outShape.Aggregate(1, (a, b) => a * b);
            long[] dst = new long[outElems];
            int[] strides = Strides(outShape);
            int[] coord = new int[outShape.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                int[] inStrides = Strides(shapes[i]);
                int inElems = shapes[i].Aggregate(1, (a, b) => a * b);
                for (int e = 0; e < inElems; e++)
                {
                    int rem = e, srcOff = 0;
                    for (int d = 0; d < shapes[i].Length; d++)
                    {
                        coord[d] = rem / inStrides[d];
                        srcOff += coord[d] * inStrides[d];
                        rem %= inStrides[d];
                    }
                    int dstOff = 0;
                    for (int d = 0; d < outShape.Length; d++)
                        dstOff += (coord[d] + (d == axis ? shapes.Take(i).Sum(s => s[axis]) : 0)) * strides[d];
                    dst[dstOff] = inputs[i][srcOff];
                }
            }
            return dst;
        }

        private static long[] IntSlice(long[] src, int[] inShape, int[] axes, int[] starts, int[] steps, int[] outShape)
        {
            int rank = inShape.Length;
            int[] fullStart = new int[rank], fullStep = Enumerable.Repeat(1, rank).ToArray();
            for (int i = 0; i < axes.Length; i++) { fullStart[axes[i]] = starts[i]; fullStep[axes[i]] = steps[i]; }
            int outElems = outShape.Aggregate(1, (a, b) => a * b);
            long[] dst = new long[outElems];
            int[] inStrides = Strides(inShape), outStrides = Strides(outShape);
            for (int e = 0; e < outElems; e++)
            {
                int rem = e, srcOff = 0;
                for (int d = 0; d < rank; d++)
                {
                    int cc = rem / outStrides[d];
                    rem %= outStrides[d];
                    srcOff += (fullStart[d] + cc * fullStep[d]) * inStrides[d];
                }
                dst[e] = src[srcOff];
            }
            return dst;
        }

        internal static int[] Strides(int[] shape)
        {
            int[] s = new int[shape.Length];
            int acc = 1;
            for (int i = shape.Length - 1; i >= 0; i--) { s[i] = acc; acc *= shape[i]; }
            return s;
        }
    }
}
