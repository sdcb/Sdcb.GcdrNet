using Sdcb.GcdrNet.Engine;
using Sdcb.GcdrNet.Onnx;

namespace Sdcb.GcdrNet.Backends.Metal;

/// <summary>
/// Metal backend root: the device, the per-(model, shape) graph cache, and
/// factories for inference sessions and fused enhancer pipelines. Thread-safe;
/// dispose with the process or when the backend is torn down.
/// </summary>
internal sealed class MetalBackend : IDisposable
{
    internal readonly MtlDevice Dev;
    private readonly Dictionary<(OnnxModel m, string shape, bool physIo), MtlGraphModel> _graphs = new();
    private readonly object _gate = new();

    private static readonly Lazy<MetalBackend> s_default = new(() => new MetalBackend());
    internal static MetalBackend Default => s_default.Value;

    internal static CompileOptions PlanOptions => new(Nhwc: true, PhysicalIo: false, Gpu: true);
    internal static CompileOptions PipeOptions => new(Nhwc: true, PhysicalIo: true, Gpu: true);

    internal MetalBackend()
    {
        Dev = MtlDevice.Probe() ?? throw new PlatformNotSupportedException("Metal: no device");
        if (Environment.GetEnvironmentVariable("GCDRNET_MTL_VERBOSE") == "1")
            Console.Error.WriteLine($"[metal] device: {Dev.Name}");
    }

    /// <summary>New inference session for a model and input shape (non-physical I/O plan).</summary>
    public MtlInferenceSession NewSession(OnnxModel m, int[] shape)
    {
        MtlGraphModel g = AcquireGraph(m, shape, physIo: false);   // caller ref: later evictions can't free it under us
        try { return new MtlInferenceSession(Dev, g.Plan, g); }
        finally { g.Release(); }
    }

    /// <summary>The cached graph for (model, shape); acquires a reference for the caller.</summary>
    internal MtlGraphModel GraphFor(OnnxModel m, int[] shape, bool physIo = false)
    {
        var key = (m, string.Join('x', shape), physIo);
        lock (_gate)
        {
            if (_graphs.TryGetValue(key, out var gg))
            {
                gg.Acquire();
                return gg;
            }
        }
        ExecPlan plan = GraphCompiler.Compile(m, shape, physIo ? PipeOptions : PlanOptions);
        var g = new MtlGraphModel(Dev, plan, constKey: m);
        lock (_gate)
        {
            if (_graphs.TryGetValue(key, out var won)) { g.Release(); won.Acquire(); return won; }
            _graphs[key] = g;
            // bound the cache: graphs share the model's const store, so eviction only
            // frees the small per-size plan+dispatch list
            while (_graphs.Count > MaxGraphs)
            {
                var victim = _graphs.Keys.First(k => !k.Equals(key));
                _graphs.Remove(victim, out var old);
                old?.Release();
            }
            g.Acquire();
            return g;
        }
    }
    private const int MaxGraphs = 32;

    /// <summary>Alias: acquires the graph for the caller (caller must Release).</summary>
    internal MtlGraphModel AcquireGraph(OnnxModel m, int[] shape, bool physIo)
        => GraphFor(m, shape, physIo);

    /// <summary>New fused enhancer worker (size-agnostic; binds per frame).</summary>
    internal MtlGcdrPipeline NewWorker(OnnxModel gc, OnnxModel dr) => new(this, gc, dr);

    /// <summary>New fused enhancer worker pre-bound to a size.</summary>
    internal MtlGcdrPipeline NewPipe(OnnxModel gc, OnnxModel dr, int ih, int iw)
    {
        var p = new MtlGcdrPipeline(this, gc, dr);
        p.Bind(ih, iw);
        return p;
    }

    public void Dispose() => Dev.Dispose();
}
