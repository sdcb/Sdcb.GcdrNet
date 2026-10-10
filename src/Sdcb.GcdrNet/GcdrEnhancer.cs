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
/// Thread-safe: sessions/pipelines are pooled per input size (rent/return).
/// </summary>
public sealed class GcdrEnhancer : IDisposable
{
    private readonly OnnxModel _gcModel, _drModel;
    private readonly GcdrBackend _backend;

    // per-size pools: bounded by MaxSizes distinct sizes, MaxPerSize idle instances
    // per size, and a memory budget — arena floats scale ~1KB/px across both
    // sessions, so a handful of multi-MP pipelines would otherwise exhaust the
    // device (12GB) or the host before the count cap ever matters
    private const int MaxSizes = 8, MaxPerSize = 4;
    private const long VkPoolBudget = 3L << 30;   // ~3 GB device memory across vk pools
    private const long CpuPoolBudget = 4L << 30;  // ~4 GB host memory per cpu pool dict
    private readonly ConcurrentDictionary<long, SizePool<InferenceSession>> _gcPool = new();
    private readonly ConcurrentDictionary<long, SizePool<InferenceSession>> _drPool = new();
    private readonly ConcurrentDictionary<long, SizePool<Backends.Vulkan.VkGcdrPipeline>> _vkPool = new();
    private long _poolsCreated, _workersCreated, _poolsEvicted;

    /// <summary>Pool diagnostics: distinct size pools created, worker instances created, pools evicted.</summary>
    public (long pools, long workers, long evicted) PoolStats =>
        (_poolsCreated, _workersCreated, _poolsEvicted);

    /// <summary>Debug/test hook: invoked with the just-used vulkan pipeline after each Enhance.</summary>
    internal Action<Backends.Vulkan.VkGcdrPipeline>? DebugAfterEnhance;

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

    private static long Key(int h, int w) => (long)h * 65536 + w;

    private SizePool<InferenceSession> CpuPool(bool gc, int h, int w)
    {
        var pools = gc ? _gcPool : _drPool;
        long key = Key(h, w);
        if (pools.TryGetValue(key, out var p)) return p;
        var model = gc ? _gcModel : _drModel;
        int c = gc ? 3 : 6;
        p = new SizePool<InferenceSession>(MaxPerSize,
            () => { Interlocked.Increment(ref _workersCreated); return new InferenceSession(GraphCompiler.Compile(model, [1, c, h, w])); });
        if (!pools.TryAdd(key, p)) { p.Dispose(); return pools[key]; }
        Interlocked.Increment(ref _poolsCreated);
        EvictIfNeeded(pools, key, CpuPoolBudget);
        return p;
    }

    private SizePool<Backends.Vulkan.VkGcdrPipeline> VkPool(int ih, int iw)
    {
        long key = Key(ih, iw);
        if (_vkPool.TryGetValue(key, out var p)) return p;
        p = new SizePool<Backends.Vulkan.VkGcdrPipeline>(MaxPerSize,
            () => { Interlocked.Increment(ref _workersCreated); return Backends.Vulkan.VkBackend.Default.NewPipe(_gcModel, _drModel, ih, iw); });
        if (!_vkPool.TryAdd(key, p)) { p.Dispose(); return _vkPool[key]; }
        Interlocked.Increment(ref _poolsCreated);
        EvictIfNeeded(_vkPool, key, VkPoolBudget);
        return p;
    }

    /// <summary>Per-pool footprint estimate for the eviction budget: ~1KB/px activations + 32MB base.</summary>
    private static long Footprint(long key) => ((key >> 16) * (key & 0xFFFF) + 1) * 1024 + (32L << 20);

    /// <summary>Bounded size cache: evict pools until total estimated footprint fits the
    /// budget and count ≤ MaxSizes. A single over-budget pool is kept.</summary>
    private void EvictIfNeeded<T>(ConcurrentDictionary<long, SizePool<T>> pools, long keep, long budget)
        where T : IDisposable
    {
        long total = 0; int n = 0;
        foreach (long k in pools.Keys) { total += Footprint(k); n++; }
        while ((n > MaxSizes || total > budget) && pools.Count > 1)
        {
            long victim = -1;
            foreach (long k in pools.Keys) if (k != keep) { victim = k; break; }
            if (victim < 0) return;
            if (pools.TryRemove(victim, out var p))
            {
                p.Dispose();
                total -= Footprint(victim); n--;
                Interlocked.Increment(ref _poolsEvicted);
            }
        }
    }

    /// <summary>
    /// Enhances a BGR uint8 HWC image: reads <paramref name="bgrSrc"/>
    /// (height×width×3, stride <paramref name="srcStride"/>) and writes
    /// <paramref name="bgrDst"/> (same dims).
    /// </summary>
    public void Enhance(ReadOnlySpan<byte> bgrSrc, int width, int height, int srcStride,
        Span<byte> bgrDst, int dstStride)
    {
        bool useVk = _backend == GcdrBackend.Vulkan
            || (_backend == GcdrBackend.Auto && VulkanOk());
        if (useVk)
        {
            var pool = VkPool(height, width);
            var pipe = pool.Rent();
            try
            {
                pipe.Enhance(bgrSrc, srcStride, bgrDst, dstStride);
                DebugAfterEnhance?.Invoke(pipe);
            }
            finally { pool.Return(pipe); }
            return;
        }
        int ph = (-height) & 31, pw = (-width) & 31;
        int h = height + ph, w = width + pw;

        var gcPool = CpuPool(true, h, w);
        var drPool = CpuPool(false, h, w);
        InferenceSession gc = gcPool.Rent();
        InferenceSession dr = drPool.Rent();
        try
        {
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
        finally
        {
            drPool.Return(dr);
            gcPool.Return(gc);
        }
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
        foreach (var s in _gcPool.Values) s.Dispose();
        foreach (var s in _drPool.Values) s.Dispose();
        foreach (var s in _vkPool.Values) s.Dispose();
        _gcPool.Clear(); _drPool.Clear(); _vkPool.Clear();
    }

    /// <summary>Bounded rent/return pool of stateful workers for one input size.</summary>
    private sealed class SizePool<T> : IDisposable where T : IDisposable
    {
        private readonly ConcurrentQueue<T> _idle = new();
        private readonly Func<T> _factory;
        private readonly int _max;
        private int _idleCount;
        private volatile bool _disposed;

        public SizePool(int max, Func<T> factory) { _max = max; _factory = factory; }

        public T Rent()
        {
            if (_idle.TryDequeue(out T? t)) { Interlocked.Decrement(ref _idleCount); return t; }
            return _factory();
        }

        public void Return(T t)
        {
            // pool may already be evicted/disposed — don't park into a dead pool
            if (_disposed) { t.Dispose(); return; }
            if (Interlocked.Increment(ref _idleCount) <= _max) { _idle.Enqueue(t); return; }
            Interlocked.Decrement(ref _idleCount);
            t.Dispose();
        }

        public void Dispose()
        {
            _disposed = true;
            while (_idle.TryDequeue(out T? t)) t.Dispose();
            _idleCount = 0;
        }
    }
}
