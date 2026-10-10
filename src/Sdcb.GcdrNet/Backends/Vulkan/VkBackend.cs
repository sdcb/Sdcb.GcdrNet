using System.Collections.Concurrent;
using Sdcb.GcdrNet.Engine;
using Sdcb.GcdrNet.Onnx;

namespace Sdcb.GcdrNet.Backends.Vulkan;

/// <summary>
/// Process-wide Vulkan device + per-(model,size) compiled graph cache.
/// Constructed lazily on first Vulkan <see cref="GcdrEnhancer"/> use.
/// </summary>
internal sealed unsafe class VkBackend : IDisposable
{
    internal readonly VkDevice Dev;
    // (model, c, h, w) -> compiled graph (weights upload + dispatch list)
    private readonly ConcurrentDictionary<(OnnxModel m, int c, int h, int w), VkGraphModel> _graphs = new();

    private static readonly Lazy<VkBackend> s_default = new(() => new VkBackend());
    internal static VkBackend Default => s_default.Value;

    internal VkBackend()
    {
        Dev = VkDevice.Create();
    }

    /// <summary>Standalone session for one model at [1,c,h,w] (test path).</summary>
    public VkInferenceSession NewSession(OnnxModel m, int[] shape)
    {
        VkGraphModel g = GraphFor(m, shape);
        g.Acquire();   // caller ref: later GraphFor evictions can't free it under us
        try { return new VkInferenceSession(Dev, GraphCompiler.Compile(m, shape), g); }
        finally { g.Release(); }
    }

    /// <summary>Compiled GPU graph for a model+shape (cached, bounded).</summary>
    internal VkGraphModel GraphFor(OnnxModel m, int[] shape)
    {
        var key = (m, shape[1], shape[2], shape[3]);
        if (_graphs.TryGetValue(key, out var g)) return g;
        g = new VkGraphModel(Dev, GraphCompiler.Compile(m, [1, shape[1], shape[2], shape[3]]));
        if (!_graphs.TryAdd(key, g)) { g.Dispose(); return _graphs[key]; }
        // bound the cache: evict an arbitrary other graph beyond the cap
        while (_graphs.Count > 32)
        {
            var victim = _graphs.Keys.FirstOrDefault(k => !k.Equals(key));
            if (victim.Equals(default)) break;
            if (_graphs.TryRemove(victim, out var old)) old.Release();
        }
        return g;
    }

    /// <summary>Creates a new GPU pipeline for a (gc, dr) model pair at a given source size
    /// (not cached — pipelines are stateful; callers pool them).</summary>
    internal VkGcdrPipeline NewPipe(OnnxModel gc, OnnxModel dr, int ih, int iw)
    {
        int h = ih + ((-ih) & 31), w = iw + ((-iw) & 31);
        ExecPlan gcPlan = GraphCompiler.Compile(gc, [1, 3, h, w]);
        ExecPlan drPlan = GraphCompiler.Compile(dr, [1, 6, h, w]);
        // hold caller refs: GraphFor(dr)'s eviction loop may pick the gc graph we
        // just got and Release its cache ref — without ours the session ctor would
        // resurrect a graph whose consts were already freed.
        VkGraphModel gg = GraphFor(gc, [1, 3, h, w]); gg.Acquire();
        VkGraphModel gd = GraphFor(dr, [1, 6, h, w]); gd.Acquire();
        try { return new VkGcdrPipeline(Dev, gcPlan, drPlan, gg, gd, ih, iw); }
        finally { gg.Release(); gd.Release(); }
    }

    public void Dispose() => Dev.Dispose();
}
