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
    private readonly ConcurrentDictionary<(OnnxModel m, int c, int h, int w, bool physIo), VkGraphModel> _graphs = new();

    private static readonly Lazy<VkBackend> s_default = new(() => new VkBackend());
    internal static VkBackend Default => s_default.Value;

    /// <summary>
    /// Plan options for GPU compiles: always channels-last (shaders address NHWC).
    /// Standalone sessions keep logical NCHW I/O (boundary relayouts in the plan);
    /// the fused pipeline keeps I/O physical and its glue shaders speak NHWC.
    /// </summary>
    internal static CompileOptions PlanOptions => new(Nhwc: true, PhysicalIo: false, Gpu: true);
    internal static CompileOptions PipeOptions => new(Nhwc: true, PhysicalIo: true, Gpu: true);

    /// <summary>fp16 activation/weight storage (needs storageBuffer16BitAccess; GCDRNET_VK_F16=0 forces fp32).</summary>
    internal bool F16 => Dev.StorageBuffer16 && Environment.GetEnvironmentVariable("GCDRNET_VK_F16") != "0";

    internal VkBackend()
    {
        Dev = VkDevice.Create();
    }

    /// <summary>Standalone session for one model at [1,c,h,w] (test path).</summary>
    public VkInferenceSession NewSession(OnnxModel m, int[] shape)
    {
        VkGraphModel g = AcquireGraph(m, shape, physIo: false);   // caller ref: later evictions can't free it under us
        try { return new VkInferenceSession(Dev, g.Plan, g); }
        finally { g.Release(); }
    }

    /// <summary>Compiled GPU graph for a model+shape (cached, bounded).</summary>
    internal VkGraphModel GraphFor(OnnxModel m, int[] shape, bool physIo = false)
    {
        var key = (m, shape[1], shape[2], shape[3], physIo);
        if (_graphs.TryGetValue(key, out var g)) return g;
        // weights live in a per-model const store shared across sizes, so a graph is
        // just a plan + dispatch list (no per-size device memory)
        g = new VkGraphModel(Dev, GraphCompiler.Compile(m, [1, shape[1], shape[2], shape[3]],
            physIo ? PipeOptions : PlanOptions), F16, constKey: m);
        if (!_graphs.TryAdd(key, g)) { g.Dispose(); return _graphs[key]; }
        // bound the cache: evict an arbitrary other graph beyond the cap
        while (_graphs.Count > MaxGraphs)
        {
            var victim = _graphs.Keys.FirstOrDefault(k => !k.Equals(key));
            if (victim.Equals(default)) break;
            if (_graphs.TryRemove(victim, out var old)) old.Release();
        }
        return g;
    }

    // graphs hold no device memory beyond the shared const store; the bound is for
    // the managed plans (each holds its own copy of the model's const floats)
    private const int MaxGraphs = 32;

    /// <summary><see cref="GraphFor"/> plus a caller reference. Retries when a concurrent
    /// eviction freed the cached graph between lookup and acquire (never resurrects it).</summary>
    internal VkGraphModel AcquireGraph(OnnxModel m, int[] shape, bool physIo)
    {
        while (true)
        {
            VkGraphModel g = GraphFor(m, shape, physIo);
            if (g.TryAcquire()) return g;
        }
    }

    /// <summary>New size-agnostic GPU worker for a (gc, dr) model pair (stateful; callers pool them).</summary>
    internal VkGcdrPipeline NewWorker(OnnxModel gc, OnnxModel dr) => new(this, gc, dr);

    /// <summary>New GPU worker pre-bound to one source size (debug/bench helper).</summary>
    internal VkGcdrPipeline NewPipe(OnnxModel gc, OnnxModel dr, int ih, int iw)
    {
        var p = new VkGcdrPipeline(this, gc, dr);
        try { p.Bind(ih, iw); return p; }
        catch { p.Dispose(); throw; }
    }

    public void Dispose() => Dev.Dispose();
}
