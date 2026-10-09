using System.Collections.Concurrent;
using Sdcb.GcdrNet.Engine;
using Sdcb.GcdrNet.Kernels;
using Sdcb.GcdrNet.Onnx;

namespace Sdcb.GcdrNet;

/// <summary>Compute backend selection for <see cref="GcdrEnhancer"/>.</summary>
public enum GcdrBackend
{
    /// <summary>CPU: Scalar → Vector&lt;float&gt; → AVX2 dispatch.</summary>
    Cpu,
    /// <summary>Vulkan compute on a discrete GPU.</summary>
    Vulkan,
    /// <summary>Vulkan if a compute-capable device exists, else CPU.</summary>
    Auto,
}

/// <summary>
/// GCDRNet two-stage document enhancer: GC-Net shading estimation → division
/// correction → DR-Net detail restoration. BGR uint8 in, BGR uint8 out.
/// Thread-safe: sessions are pooled per input size.
/// </summary>
public sealed class GcdrEnhancer : IDisposable
{
    private readonly OnnxModel _gcModel, _drModel;
    private readonly GcdrBackend _backend;
    private readonly ConcurrentDictionary<int, InferenceSession> _gcSessions = new();
    private readonly ConcurrentDictionary<int, InferenceSession> _drSessions = new();
    private readonly ConcurrentBag<InferenceSession> _pool = new();

    private GcdrEnhancer(OnnxModel gc, OnnxModel dr, GcdrBackend backend)
    {
        _gcModel = gc;
        _drModel = dr;
        _backend = backend;
    }

    public static GcdrEnhancer FromFiles(string gcnetOnnxPath, string drnetOnnxPath,
        GcdrBackend backend = GcdrBackend.Cpu) =>
        new(OnnxModel.Load(gcnetOnnxPath), OnnxModel.Load(drnetOnnxPath), backend);

    public static GcdrEnhancer FromBytes(ReadOnlySpan<byte> gcnetOnnx, ReadOnlySpan<byte> drnetOnnx,
        GcdrBackend backend = GcdrBackend.Cpu) =>
        new(OnnxModel.Load(gcnetOnnx.ToArray()), OnnxModel.Load(drnetOnnx.ToArray()), backend);

    /// <summary>Compile both models for a given padded input size.</summary>
    private InferenceSession SessionFor(bool gc, int h, int w)
    {
        var cache = gc ? _gcSessions : _drSessions;
        int key = h * 65536 + w;
        return cache.GetOrAdd(key, _ =>
        {
            ExecPlan plan = GraphCompiler.Compile(gc ? _gcModel : _drModel,
                [1, gc ? 3 : 6, h, w]);
            return new InferenceSession(plan);
        });
    }

    /// <summary>
    /// Enhances a BGR uint8 HWC image in place semantics: reads
    /// <paramref name="bgrSrc"/> (height×width×3, stride <paramref name="srcStride"/>)
    /// and writes <paramref name="bgrDst"/> (same dims).
    /// </summary>
    public void Enhance(ReadOnlySpan<byte> bgrSrc, int width, int height, int srcStride,
        Span<byte> bgrDst, int dstStride)
    {
        bool useVk = _backend == GcdrBackend.Vulkan
            || (_backend == GcdrBackend.Auto && VulkanOk());
        if (useVk)
        {
            Sdcb.GcdrNet.Backends.Vulkan.VkBackend.Default
                .PipeFor(_gcModel, _drModel, height, width)
                .Enhance(bgrSrc, srcStride, bgrDst, dstStride);
            return;
        }
        int ph = (-height) & 31, pw = (-width) & 31;
        int h = height + ph, w = width + pw;

        InferenceSession gc = SessionFor(true, h, w);
        InferenceSession dr = SessionFor(false, h, w);

        // stage 0: fill GC input — BGR u8 → float NCHW /255, top/left replicate pad
        Span<float> x = gc.Input;
        PixelOps.BgrToNchwPad(bgrSrc, width, height, srcStride, x, h, w, ph, pw);

        // stage 1: GC-Net → shading map
        gc.Run();
        ReadOnlySpan<float> shadow = gc.Output;
        int[] oshape = gc.OutputShape;
        if (oshape[2] != h || oshape[3] != w)
            throw new NotSupportedException($"GC-Net returned {oshape[2]}x{oshape[3]}, expected {h}x{w}.");

        // stage 2: corrected = clamp(x / max(shadow,1e-6), 0,1); concat into DR input
        Span<float> x6 = dr.Input;
        int hw = h * w;
        x.CopyTo(x6.Slice(0, 3 * hw));
        PixelOps.DivClampInto(x, shadow, x6.Slice(3 * hw, 3 * hw));

        // stage 3: DR-Net
        dr.Run();

        // stage 4: crop pad → u8 BGR
        PixelOps.NchwToBgrCrop(dr.Output, h, w, ph, pw, bgrDst, width, height, dstStride);
    }

    private static bool VulkanOk()
    {
        try { _ = Sdcb.GcdrNet.Backends.Vulkan.VkBackend.Default.Dev.DeviceName; return true; }
        catch { return false; }
    }

    /// <summary>Convenience: enhance a tightly-packed BGR HWC image, returns new array.</summary>
    public byte[] Enhance(byte[] bgrSrc, int width, int height)
    {
        byte[] dst = new byte[width * height * 3];
        Enhance(bgrSrc, width, height, width * 3, dst, width * 3);
        return dst;
    }

    public void Dispose()
    {
        foreach (var s in _gcSessions.Values) s.Dispose();
        foreach (var s in _drSessions.Values) s.Dispose();
        _gcSessions.Clear(); _drSessions.Clear();
    }
}
