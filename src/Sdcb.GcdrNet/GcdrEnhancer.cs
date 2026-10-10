using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using Sdcb.GcdrNet.Engine;
using Sdcb.GcdrNet.Kernels;
using Sdcb.GcdrNet.Onnx;

namespace Sdcb.GcdrNet;

/// <summary>Compute backend selection for <see cref="GcdrEnhancer"/>.</summary>
public enum GcdrBackend
{
    /// <summary>CPU: Scalar → Vector&lt;float&gt; → AVX2 → AVX-512 dispatch.</summary>
    Cpu,
    /// <summary>Vulkan compute on a discrete GPU.</summary>
    Vulkan,
    /// <summary>Vulkan if a compute-capable device exists, else CPU.</summary>
    Auto,
}

/// <summary>
/// GCDRNet two-stage document enhancer: GC-Net shading estimation → division
/// correction → DR-Net detail restoration. BGR uint8 in, BGR uint8 out.
/// Thread-safe: size-agnostic workers (CPU arenas / Vulkan pipelines) are pooled
/// and rented per call; per-size plans are cached and share packed weights.
/// </summary>
public sealed class GcdrEnhancer : IDisposable
{
    private readonly OnnxModel _gcModel, _drModel;
    private readonly GcdrBackend _backend;

    // CPU: per-size plans are managed metadata (cached, at most MaxSizes; packed
    // weights shared across sizes through the kernel caches). Workers own grow-only
    // native arenas (CpuArenaBudget, see CpuWorker) and serve every size; idle
    // workers beyond the first are kept only while their arenas fit the budget.
    private const int MaxSizes = 8, MaxPerSize = 4;
    private const long CpuArenaBudget = 4L << 30;
    private readonly ConcurrentDictionary<long, (ExecPlan gc, ExecPlan dr)> _cpuPlans = new();
    private readonly CpuKernels.Cache _gcKernels = new(), _drKernels = new();
    private SizePool<CpuWorker>? _cpuPool;
    // Vulkan workers are size-agnostic (budgeted grow-only arenas, re-record on size
    // change): one pool serves every size; idle workers beyond the first are kept
    // only while their device memory fits the budget
    // (3 GB; integrated GPUs: at most a quarter of the shared device-local heap)
    private static long VkIdleBudget => (long)Backends.Vulkan.VkGcdrPipeline.BudgetFor(Backends.Vulkan.VkBackend.Default.Dev);
    private SizePool<Backends.Vulkan.VkGcdrPipeline>? _vkPool;
    private long _poolsCreated, _workersCreated, _poolsEvicted;

    /// <summary>Diagnostics: CPU size plans compiled (Vulkan: worker pools created), worker
    /// instances created, CPU size plans evicted.</summary>
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

    // pipeline sessions keep I/O in plan layout: the glue kernels below read/write
    // channels-last directly, so the plan has no boundary relayouts
    private static readonly CompileOptions CpuPlan = new(CompileOptions.DefaultNhwc, PhysicalIo: true);
    private static bool CpuNhwc => CpuPlan.Nhwc;

    private static long Key(int h, int w) => (long)h * 65536 + w;

    /// <summary>gc/dr plans for one padded size (compiled once, weights packed via the shared caches).</summary>
    private (ExecPlan gc, ExecPlan dr) CpuPlans(int h, int w)
    {
        long key = Key(h, w);
        if (_cpuPlans.TryGetValue(key, out var p)) return p;
        // the two nets compile independently: build them side by side
        var tgc = Task.Run(() =>
        {
            ExecPlan g = GraphCompiler.Compile(_gcModel, [1, 3, h, w], CpuPlan);
            CpuKernels.Pack(g, _gcKernels);
            return g;
        });
        ExecPlan dr = GraphCompiler.Compile(_drModel, [1, 6, h, w], CpuPlan);
        CpuKernels.Pack(dr, _drKernels);
        ExecPlan gc = tgc.GetAwaiter().GetResult();
        p = (gc, dr);
        if (!_cpuPlans.TryAdd(key, p)) return _cpuPlans.TryGetValue(key, out var won) ? won : p;
        Interlocked.Increment(ref _poolsCreated);
        // bound the cache (plans are managed; in-flight users keep theirs alive)
        while (_cpuPlans.Count > MaxSizes)
        {
            long victim = -1;
            foreach (long k in _cpuPlans.Keys) if (k != key) { victim = k; break; }
            if (victim < 0 || !_cpuPlans.TryRemove(victim, out _)) break;
            Interlocked.Increment(ref _poolsEvicted);
        }
        return p;
    }

    private SizePool<CpuWorker> CpuPool()
    {
        if (_cpuPool is { } p) return p;
        p = new SizePool<CpuWorker>(MaxPerSize,
            () => { Interlocked.Increment(ref _workersCreated); return new CpuWorker(); },
            w => w.Bytes, CpuArenaBudget);
        if (Interlocked.CompareExchange(ref _cpuPool, p, null) is { } won) { p.Dispose(); return won; }
        return p;
    }

    private SizePool<Backends.Vulkan.VkGcdrPipeline> VkPool()
    {
        if (_vkPool is { } p) return p;
        p = new SizePool<Backends.Vulkan.VkGcdrPipeline>(MaxPerSize,
            () => { Interlocked.Increment(ref _workersCreated); return Backends.Vulkan.VkBackend.Default.NewWorker(_gcModel, _drModel); },
            w => w.DeviceBytes, VkIdleBudget);
        if (Interlocked.CompareExchange(ref _vkPool, p, null) is { } won) { p.Dispose(); return won; }
        Interlocked.Increment(ref _poolsCreated);
        return p;
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
            var pool = VkPool();
            var pipe = pool.Rent();
            try
            {
                pipe.Enhance(bgrSrc, height, width, srcStride, bgrDst, dstStride);
                DebugAfterEnhance?.Invoke(pipe);
            }
            finally { pool.Return(pipe); }
            return;
        }
        int ph = (-height) & 31, pw = (-width) & 31;
        int h = height + ph, w = width + pw;

        var (gcPlan, drPlan) = CpuPlans(h, w);
        var cpuPool = CpuPool();
        CpuWorker worker = cpuPool.Rent();
        try
        {
            var (gcOff, total) = StackArenas(gcPlan, drPlan);
            nint arena = worker.Arena(total * sizeof(float));
            using var gc = new InferenceSession(gcPlan, arena + (nint)(gcOff * sizeof(float)));
            using var dr = new InferenceSession(drPlan, arena);

            // stage 0: fill GC input — BGR u8 → float /255, top/left replicate pad
            Span<float> x = gc.Input;
            if (CpuNhwc) PixelOps.BgrToNhwcPad(bgrSrc, width, height, srcStride, x, h, w, ph, pw);
            else PixelOps.BgrToNchwPad(bgrSrc, width, height, srcStride, x, h, w, ph, pw);

            // stage 1: GC-Net → shading map
            gc.Run();
            ReadOnlySpan<float> shadow = gc.Output;
            int[] oshape = gc.OutputShape;
            if (oshape[2] != h || oshape[3] != w)
                throw new NotSupportedException($"GC-Net returned {oshape[2]}x{oshape[3]}, expected {h}x{w}.");

            // stage 2: corrected = clamp(x / max(shadow,1e-6), 0,1); concat into DR input
            Span<float> x6 = dr.Input;
            int hw = h * w;
            if (CpuNhwc) PixelOps.DivConcatNhwc(x, shadow, x6);
            else
            {
                x.CopyTo(x6.Slice(0, 3 * hw));
                PixelOps.DivClampInto(x, shadow, x6.Slice(3 * hw, 3 * hw));
            }

            // stage 3: DR-Net
            dr.Run();

            // stage 4: crop pad → u8 BGR
            if (CpuNhwc) PixelOps.NhwcToBgrCrop(dr.Output, h, w, ph, pw, bgrDst, width, height, dstStride);
            else PixelOps.NchwToBgrCrop(dr.Output, h, w, ph, pw, bgrDst, width, height, dstStride);
        }
        finally { cpuPool.Return(worker); }
    }

    /// <summary>
    /// One arena for both stages (gc runs to completion before dr starts): the gc arena is
    /// stacked right past the dr input window, so gc's input/output survive the
    /// stage-2 glue that writes the dr input, and dr then reuses everything above.
    /// Returns the gc arena offset and the total size, in floats.
    /// </summary>
    internal static (long gcOff, long total) StackArenas(ExecPlan gc, ExecPlan dr, long align = 16)
    {
        CValue din = dr.Values[dr.InputValue];
        long o = dr.BufOffset[din.Buf], end = (o + din.ElementCount + align - 1) / align * align;
        long gcOff = o >= gc.ArenaFloats ? 0 : end;
        return (gcOff, Math.Max(dr.ArenaFloats, gcOff + gc.ArenaFloats));
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
        Interlocked.Exchange(ref _cpuPool, null)?.Dispose();
        Interlocked.Exchange(ref _vkPool, null)?.Dispose();
        _cpuPlans.Clear();
    }

    /// <summary>
    /// Size-agnostic CPU worker: one native arena holding both stages (see
    /// <see cref="StackArenas"/>), borrowed by per-call sessions. Within
    /// <see cref="CpuArenaBudget"/> it only grows (25% headroom on regrowth), so mixed sizes
    /// neither reallocate nor re-fault pages; an over-budget arena (one huge image) drops
    /// to exact size once a size that fits comes along.
    /// </summary>
    private sealed unsafe class CpuWorker : IDisposable
    {
        private nint _p;
        private long _bytes;

        public long Bytes => _bytes;

        public nint Arena(long need)
        {
            need = Math.Max(need, 64);
            bool shrink = _p != 0 && _bytes > CpuArenaBudget && need <= CpuArenaBudget;
            if (_p != 0 && _bytes >= need && !shrink) return _p;
            bool headroom = _p != 0 && !shrink && need / 4 * 5 <= CpuArenaBudget;
            Release();
            long cap = headroom ? need / 4 * 5 : need;
            _p = (nint)NativeMemory.AllocZeroed((nuint)cap);
            _bytes = cap;
            GC.AddMemoryPressure(cap);
            return _p;
        }

        private void Release()
        {
            if (_p == 0) return;
            NativeMemory.Free((void*)_p);
            GC.RemoveMemoryPressure(_bytes);
            _p = 0; _bytes = 0;
        }

        public void Dispose()
        {
            Release();
            GC.SuppressFinalize(this);
        }

        ~CpuWorker() => Release();
    }

    /// <summary>Bounded rent/return pool of stateful workers (at most <c>max</c> idle and,
    /// with a <c>weight</c>, at most <c>budget</c> idle bytes beyond the first idle worker).</summary>
    private sealed class SizePool<T> : IDisposable where T : IDisposable
    {
        private readonly ConcurrentQueue<T> _idle = new();
        private readonly Func<T> _factory;
        private readonly int _max;
        private readonly Func<T, long>? _weight;
        private readonly long _budget;
        private int _idleCount;
        private long _idleBytes;
        private volatile bool _disposed;

        public SizePool(int max, Func<T> factory, Func<T, long>? weight = null, long budget = long.MaxValue)
        {
            _max = max; _factory = factory; _weight = weight; _budget = budget;
        }

        public T Rent()
        {
            if (_idle.TryDequeue(out T? t))
            {
                Interlocked.Decrement(ref _idleCount);
                if (_weight is not null) Interlocked.Add(ref _idleBytes, -_weight(t));
                return t;
            }
            return _factory();
        }

        public void Return(T t)
        {
            // pool may already be evicted/disposed — don't park into a dead pool
            if (_disposed) { t.Dispose(); return; }
            long w = _weight?.Invoke(t) ?? 0;
            if (Interlocked.Increment(ref _idleCount) <= _max
                && (Volatile.Read(ref _idleCount) == 1 || Interlocked.Read(ref _idleBytes) + w <= _budget))
            {
                Interlocked.Add(ref _idleBytes, w);
                _idle.Enqueue(t);
                // raced with Dispose after the check above: drain so nothing is parked in a dead pool
                if (_disposed) while (_idle.TryDequeue(out T? d)) d.Dispose();
                return;
            }
            Interlocked.Decrement(ref _idleCount);
            t.Dispose();
        }

        public void Dispose()
        {
            _disposed = true;
            while (_idle.TryDequeue(out T? t)) t.Dispose();
            _idleCount = 0;
            _idleBytes = 0;
        }
    }
}
