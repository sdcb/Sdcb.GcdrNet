using Sdcb.GcdrNet.Engine;

namespace Sdcb.GcdrNet.Backends.Vulkan;

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
/// Per-(device, ExecPlan) half of the Vulkan backend: constant tensor upload
/// and the node list compiled to <see cref="VkDispatch"/> records. Sessions own
/// the activation arena and resolve slot ids to their live buffer handles.
/// </summary>
internal sealed class VkGraphModel
{
    private readonly VkDevice _dev;
    private readonly ExecPlan _plan;
    internal readonly VkBuffer Consts;       // all const float tensors, packed
    private readonly Dictionary<int, ulong> _constOff = new();
    internal readonly VkBuffer Dummy;        // 16B zero buffer for unused bindings
    internal readonly VkDispatch[] Dispatches;
    internal readonly ulong ConstBytes;

    internal VkGraphModel(VkDevice dev, ExecPlan plan)
    {
        _dev = dev;
        _plan = plan;

        // ---- pack all float consts into one device buffer ----
        ulong total = 0;
        var offs = new List<(int v, ulong off, int floats)>();
        for (int i = 0; i < plan.Values.Length; i++)
        {
            CValue v = plan.Values[i];
            if (v.Floats == null) continue;
            total = (total + 15) & ~15UL;
            _constOff[i] = total;
            offs.Add((i, total, v.Floats.Length));
            total += (ulong)v.Floats.Length * 4;
        }
        ConstBytes = Math.Max(total, 16);
        Consts = dev.NewStorageBuffer(ConstBytes, hostVisible: dev.CoherentDeviceLocal);
        Dummy = dev.NewStorageBuffer(64, hostVisible: dev.CoherentDeviceLocal);
        unsafe
        {
            if ((Consts.Flags & VkConst.MemHostVisible) != 0)
            {
                float* p = (float*)Consts.Map();
                foreach ((int v, ulong off, _) in offs)
                    plan.Values[v].Floats!.CopyTo(new Span<float>(p + off / 4, (int)plan.Values[v].ElementCount));
                Consts.Flush(0, total);
                Consts.Unmap();
            }
            else
            {
                VkBuffer staging = dev.NewStorageBuffer(ConstBytes, hostVisible: true);
                float* p = (float*)staging.Map();
                foreach ((int v, ulong off, _) in offs)
                    plan.Values[v].Floats!.CopyTo(new Span<float>(p + off / 4, (int)plan.Values[v].ElementCount));
                staging.Flush(0, total);
                staging.Unmap();
                lock (dev.Sync)
                {
                    IntPtr cmd = dev.NewCommandBuffer();
                    IntPtr fence = dev.NewFence();
                    var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
                    Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
                    Vk.VkBufferCopy r = new() { Size = total };
                    Vk.vkCmdCopyBuffer(cmd, staging.Buffer, Consts.Buffer, 1, &r);
                    Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
                    dev.Submit(cmd, fence);
                    dev.WaitFence(fence);
                    dev.DestroyFence(fence);
                    dev.FreeCommandBuffer(cmd);
                }
                staging.Free();
            }
        }

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

    /// <summary>Dispatch index range per source ExecNode (for per-node debug runs).</summary>
    internal readonly (int start, int count)[] NodeRanges;

    // ---- value → bind resolution ----
    private VkBind Bind(int valueId)
    {
        CValue v = _plan.Values[valueId];
        ulong bytes = (ulong)Math.Max(v.ElementCount, 1) * 4;
        if (v.Buf >= 0)
            return new VkBind(VkSlot.Arena, (ulong)_plan.BufOffset[v.Buf] * 4, bytes);
        if (_constOff.TryGetValue(valueId, out ulong off))
            return new VkBind(VkSlot.Consts, off, bytes);
        if (v.Floats != null)
            throw new InvalidOperationException($"const value '{v.Name}' not packed");
        throw new InvalidOperationException($"value '{v.Name}' has no GPU storage (ints must be folded)");
    }

    private static VkBind BindBuf(VkSlot s, ulong off, ulong bytes) => new(s, off, bytes);

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

    private IEnumerable<VkDispatch> Emit(ExecNode n)
    {
        CValue[] V = _plan.Values;
        int[] shapeIn(int k) => V[n.In[k]].Shape;
        int[] shapeOut() => V[n.Out[0]].Shape;
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
                int[] os = Pad4(shapeOut());
                int[] ostr = Strides(os);
                int[] astr = BroadcastStrides(Pad4(shapeIn(0)), ostr);
                int[] bstr = BroadcastStrides(Pad4(shapeIn(1)), ostr);
                uint act = p.Act switch { ActivationKind.Relu => 1u, ActivationKind.Sigmoid => 2u, _ => 0u };
                // contiguous same-shape fast path: vec4, no broadcast math
                if (nOut() % 4 == 0 && shapeIn(0).SequenceEqual(shapeOut()) && shapeIn(1).SequenceEqual(shapeOut()))
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
                int[] si = shapeIn(0), sw_ = V[n.In[1]].Shape, so = shapeOut();
                uint ic = (uint)si[1], ih = (uint)si[2], iw = (uint)si[3];
                uint oc = (uint)sw_[0], oh = (uint)so[2], ow = (uint)so[3];
                if (p.PadMode == PadMode.Constant && p.PadValue != 0f)
                    throw new NotSupportedException("Vulkan conv: nonzero constant pad");
                uint flags = (p.BiasValue >= 0 ? 1u : 0u)
                    | (p.PadMode == PadMode.Reflect ? 2u : 0u)
                    | (p.Act == ActivationKind.Relu ? 4u : 0u)
                    | (p.AddValue >= 0 ? 8u : 0u);
                if (p.Group > 1)
                {
                    if (p.Group != ic) throw new NotSupportedException("grouped non-depthwise conv");
                    yield return ND("depthwise",
                        [Bind(n.In[0]), Bind(n.In[1]), B(p.BiasValue), Bind(n.Out[0])],
                        PcOf(ic, ih, iw, oh, ow, (uint)p.Kernel[0], (uint)p.Kernel[1],
                            (uint)p.Strides[0], (uint)p.Strides[1], (uint)p.Pads[0], (uint)p.Pads[1],
                            flags & ~8u, 0),
                        G(ic * oh * ow), name: "depthwise");
                    break;
                }
                string convShader =
                    p.Kernel[0] == 3 && p.Kernel[1] == 3 && p.Strides[0] == 1 && p.Strides[1] == 1 ? "conv_k33s1" :
                    p.Kernel[0] == 3 && p.Kernel[1] == 3 && p.Strides[0] == 2 && p.Strides[1] == 2 ? "conv_k33s2" :
                    p.Kernel[0] == 1 && p.Kernel[1] == 1 && p.Strides[0] == 1 && p.Strides[1] == 1 ? "conv_k11s1" :
                    "conv_gen";
                yield return ND(convShader,
                    [Bind(n.In[0]), Bind(n.In[1]), B(p.BiasValue), B(p.AddValue), Bind(n.Out[0])],
                    PcOf(ic, ih, iw, oc, oh, ow, (uint)p.Kernel[0], (uint)p.Kernel[1],
                        (uint)p.Strides[0], (uint)p.Strides[1], (uint)p.Pads[0], (uint)p.Pads[1],
                        flags, 0),
                    (uint)((ow + 15) / 16), (uint)((oh + 7) / 8), (uint)((oc + 63) / 64), name: "conv");
                break;
            }

            case ExecOp.MaxPool:
            {
                var p = (PoolP)n.P!;
                int[] si = shapeIn(0), so = shapeOut();
                yield return ND("maxpool", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf((uint)si[1], (uint)si[2], (uint)si[3], (uint)so[2], (uint)so[3],
                        (uint)p.Kernel[0], (uint)p.Kernel[1], (uint)p.Strides[0], (uint)p.Strides[1],
                        (uint)p.Pads[0], (uint)p.Pads[1], p.FuseRelu ? 1u : 0u),
                    G((uint)so[1] * (uint)so[2] * (uint)so[3]));
                break;
            }

            case ExecOp.LayerNorm:
            {
                var p = (LnP)n.P!;
                uint len = (uint)shapeIn(0)[^1];
                uint rows = (uint)(V[n.In[0]].ElementCount / len);
                yield return ND("layernorm",
                    [Bind(n.In[0]), Bind(n.In[1]), Bind(n.In[2]), Bind(n.Out[0])],
                    PcOfF([len], p.Eps), rows, name: "layernorm");
                break;
            }

            case ExecOp.Gemm:
            {
                var p = (GemmP)n.P!;
                int[] sa = shapeIn(0);
                uint m = (uint)(V[n.In[0]].ElementCount / sa[^1]);
                uint k = (uint)sa[^1];
                uint nn = (uint)shapeOut()[^1];
                yield return ND("gemm",
                    [Bind(n.In[0]), Bind(n.In[1]), B(p.BiasValue), Bind(n.Out[0])],
                    PcOf(m, nn, k, (p.TransB ? 1u : 0u) | (p.BiasValue >= 0 ? 2u : 0u)),
                    (nn + 63) / 64, (m + 63) / 64, name: "gemm");
                break;
            }

            case ExecOp.ResizeBilinear:
            {
                var p = (ResizeP)n.P!;
                int[] si = shapeIn(0), so = shapeOut();
                yield return ND("bilinear", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOfF([(uint)si[1], (uint)si[2], (uint)si[3], (uint)so[2], (uint)so[3],
                        p.FuseRelu ? 1u : 0u],
                        p.Scales[2], p.Scales[3]),
                    G((uint)so[1] * (uint)so[2] * (uint)so[3]));
                break;
            }

            case ExecOp.Pad:
            {
                var p = (PadP)n.P!;
                int[] id4 = Pad4(shapeIn(0)), od4 = Pad4(shapeOut());
                int[] pb4 = new int[4];
                for (int d = 0; d < 4; d++) pb4[d] = p.Pads[d];
                yield return ND("pad", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOfF(U(id4[0], id4[1], id4[2], id4[3], od4[0], od4[1], od4[2], od4[3],
                        pb4[0], pb4[1], pb4[2], pb4[3], p.Mode == PadMode.Reflect ? 1 : 0),
                        p.Value),
                    G((uint)V[n.Out[0]].ElementCount));
                break;
            }

            case ExecOp.Transpose:
            {
                var p = (TransposeP)n.P!;
                int[] is4 = Pad4(shapeIn(0));
                int[] os4 = Pad4(shapeOut());
                int[] ostr = Strides(os4), istr = Strides(is4);
                // perm indexes padded rank-4 (original perm on tail dims)
                int[] perm4 = PermuteTo4(p.Perm, shapeIn(0).Length);
                if (perm4.SequenceEqual([0, 2, 1, 3]) && is4[0] == 1)
                {
                    // pure 2D swap of the last two dims: [1,R,C] -> [1,C,R]
                    uint rr = (uint)is4[1], cc = (uint)(is4[2] * is4[3]);
                    yield return ND("transpose0231", [Bind(n.In[0]), Bind(n.Out[0])],
                        PcOf(rr, cc), (cc + 31) / 32, (rr + 31) / 32);
                    break;
                }
                if (perm4.SequenceEqual([0, 1, 3, 2]) && is4[0] == 1 && is4[1] == 1)
                {
                    // [1,1,R,C] -> [1,1,C,R]
                    uint rr = (uint)is4[2], cc = (uint)is4[3];
                    yield return ND("transpose0231", [Bind(n.In[0]), Bind(n.Out[0])],
                        PcOf(rr, cc), (cc + 31) / 32, (rr + 31) / 32);
                    break;
                }
                if (perm4.SequenceEqual([0, 2, 3, 1]) && is4[0] == 1)
                {
                    // [1,A,B,C] -> [1,B,C,A]: matrix [A, B*C] -> [B*C, A]
                    uint rr = (uint)is4[1], cc = (uint)(is4[2] * is4[3]);
                    yield return ND("transpose0231", [Bind(n.In[0]), Bind(n.Out[0])],
                        PcOf(rr, cc), (cc + 31) / 32, (rr + 31) / 32);
                    break;
                }
                yield return ND("transpose", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf(U(nOut(), ostr[0], ostr[1], ostr[2],
                        istr[perm4[0]], istr[perm4[1]], istr[perm4[2]], istr[perm4[3]])),
                    G((uint)nOut()));
                break;
            }

            case ExecOp.Slice:
            {
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
                var p = (ConcatP)n.P!;
                int[] os = shapeOut();
                int axis = p.Axis;
                int inner = 1, dstAxis = os[axis];
                for (int d = axis + 1; d < os.Length; d++) inner *= os[d];
                int axisOff = 0;
                foreach (int iv in n.In)
                {
                    int[] ishape = V[iv].Shape;
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
                int[] si = shapeIn(0);
                var ge = p.GroupEnds; var of = p.Offsets;
                var pc = new List<uint> { (uint)si[1], (uint)si[2], (uint)si[3], (uint)p.Axis };
                for (int i = 0; i < 5; i++) pc.Add((uint)(i < ge.Length ? ge[i] : si[1]));
                for (int i = 0; i < 5; i++) pc.Add((uint)(i < of.Length ? of[i] : 0));
                yield return ND("shift", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf(pc.ToArray()), G((uint)nOut()));
                break;
            }

            default:
                throw new NotSupportedException($"Vulkan: ExecOp {n.Op} not implemented");
        }

        VkDispatch ND(string shader, VkBind[] binds, byte[] pc, uint gx, uint gy = 1, uint gz = 1, string? name = null)
        {
            var d = new VkDispatch
            {
                Pipe = _dev.GetPipeline(shader, binds.Length, pc.Length),
                Pc = pc, Binds = binds, Gx = gx, Gy = gy, Gz = gz,
                Name = $"{shader}({n.Name})",
            };
            return d;
        }

        VkBind B(int valueId) => valueId >= 0 ? Bind(valueId) : new VkBind(VkSlot.Dummy, 0, 64);
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

    // broadcast strides for bShape4 onto outShape4 strides
    private static int[] BroadcastStrides(int[] bShape4, int[] outStr4)
    {
        int[] s = new int[4];
        int acc = 1;
        for (int i = 3; i >= 0; i--) { s[i] = acc; acc *= bShape4[i]; }
        for (int i = 0; i < 4; i++) if (bShape4[i] == 1) s[i] = 0;
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

    public void Dispose()
    {
        Consts.Free();
        Dummy.Free();
    }
}
