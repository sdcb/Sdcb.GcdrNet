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
    // (gcModel, drModel, ih, iw) -> fused pipeline
    private readonly ConcurrentDictionary<(OnnxModel gc, OnnxModel dr, int ih, int iw), VkGcdrPipeline> _pipes = new();

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
        return new VkInferenceSession(Dev, GraphCompiler.Compile(m, shape), g);
    }

    /// <summary>Compiled GPU graph for a model+shape (cached).</summary>
    internal VkGraphModel GraphFor(OnnxModel m, int[] shape)
        => _graphs.GetOrAdd((m, shape[1], shape[2], shape[3]),
            k => new VkGraphModel(Dev, GraphCompiler.Compile(k.m, [1, k.c, k.h, k.w])));

    /// <summary>GPU pipeline for a (gc, dr) model pair at a given source size.</summary>
    internal VkGcdrPipeline PipeFor(OnnxModel gc, OnnxModel dr, int ih, int iw)
        => _pipes.GetOrAdd((gc, dr, ih, iw), k =>
        {
            int h = k.ih + ((-k.ih) & 31), w = k.iw + ((-k.iw) & 31);
            ExecPlan gcPlan = GraphCompiler.Compile(k.gc, [1, 3, h, w]);
            ExecPlan drPlan = GraphCompiler.Compile(k.dr, [1, 6, h, w]);
            return new VkGcdrPipeline(Dev, gcPlan, drPlan, GraphFor(k.gc, [1, 3, h, w]), GraphFor(k.dr, [1, 6, h, w]), k.ih, k.iw);
        });

    public void Dispose() => Dev.Dispose();
}
