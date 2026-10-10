using Sdcb.GcdrNet.Kernels;

namespace Sdcb.GcdrNet.Engine;

/// <summary>
/// CPU-private packed weights (conv panels + Winograd transforms, Gemm panels,
/// UpHead lanes) stored in the plan's op params. Per-size plans of one model
/// carry identical weights, so a <see cref="Cache"/> shared across sizes packs
/// each layer once: entries are keyed by node identity and only reused when the
/// weights (and bias) are element-for-element equal.
/// </summary>
internal static class CpuKernels
{
    internal sealed class Cache
    {
        private readonly Dictionary<(ExecOp op, string name, int src), (float[] w, float[]? b, object kp)> _d = new();

        internal object GetOrPack(ExecNode n, float[] w, float[]? b, Func<object> pack)
        {
            var key = (n.Op, n.Name, n.SrcNode);
            lock (_d)
                if (_d.TryGetValue(key, out var e) && Same(e.w, w) && Same(e.b, b)) return e.kp;
            object kp = pack();
            lock (_d) _d[key] = (w, b, kp);
            return kp;
        }

        private static bool Same(float[]? a, float[]? b) =>
            ReferenceEquals(a, b) || (a is not null && b is not null && a.AsSpan().SequenceEqual(b));
    }

    /// <summary>Fills missing kernel plans of an NHWC plan (idempotent; plans may be shared).</summary>
    internal static void Pack(ExecPlan plan, Cache? cache)
    {
        if (!plan.Nhwc) return;
        CValue[] V = plan.Values;
        foreach (ExecNode n in plan.Nodes)
            switch (n.P)
            {
                case GemmP gp when gp.KernelPlan is null && V[n.In[1]].Floats is { } bw:
                {
                    int[] bs = V[n.In[1]].Shape;
                    int kk = gp.TransB ? bs[1] : bs[0], nn = gp.TransB ? bs[0] : bs[1];
                    float[]? gb = gp.BiasValue >= 0 ? V[gp.BiasValue].Floats : null;
                    object kp = Get(cache, n, bw, gb, () => Nhwc.PackGemm(bw, kk, nn, gp.TransB, gb));
                    Interlocked.CompareExchange(ref gp.KernelPlan, kp, null);
                    break;
                }
                case ConvP cp when cp.KernelPlan is null:
                {
                    CValue w = V[n.In[1]];
                    float[]? b = cp.BiasValue >= 0 ? V[cp.BiasValue].Floats : null;
                    int ic = V[n.In[0]].Shape[1];
                    object kp = Get(cache, n, w.Floats!, b, () =>
                    {
                        NhwcConvPlan p = Nhwc.Pack(w.Floats!, w.Shape, b, cp.Group, ic);
                        if (Nhwc.WinogradEligible(cp, w.Shape, ic)) p.WinoU = Nhwc.PackWinograd(w.Floats!, w.Shape[0], ic);
                        return p;
                    });
                    Interlocked.CompareExchange(ref cp.KernelPlan, kp, null);
                    break;
                }
                case UpHeadP up when up.KernelPlan is null:
                {
                    CValue w = V[n.In[2]];
                    float[]? b = n.In.Length > 3 ? V[n.In[3]].Floats : null;
                    object kp = Get(cache, n, w.Floats!, b, () => Nhwc.PackUpHead(w.Floats!, w.Shape[0], w.Shape[1], b));
                    Interlocked.CompareExchange(ref up.KernelPlan, kp, null);
                    break;
                }
            }
    }

    private static object Get(Cache? cache, ExecNode n, float[] w, float[]? b, Func<object> pack) =>
        cache is null ? pack() : cache.GetOrPack(n, w, b, pack);
}
