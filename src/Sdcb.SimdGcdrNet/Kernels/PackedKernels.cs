using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Wasm;

using V4 = System.Runtime.Intrinsics.Vector128<float>;

namespace Sdcb.SimdGcdrNet.Kernels;

/// <summary>WebAssembly SIMD128 (PackedSimd) versions of the non-conv ops that have AVX2 / NEON kernels.</summary>
internal static unsafe class PackedKernels
{
    private static readonly ParallelOptions PO = new() { MaxDegreeOfParallelism = Avx2Kernels.Threads };

    /// <summary>Row-wise LayerNorm over the last axis (c); rows split across threads in groups
    /// (<see cref="NeonKernels.LayerNorm"/> with separate multiply + add).</summary>
    public static void LayerNorm(ReadOnlySpan<float> x, int rows, int c,
        ReadOnlySpan<float> w, ReadOnlySpan<float> b, Span<float> dst, float eps)
    {
        int per = Math.Max(1, 16384 / c), tasks = (rows + per - 1) / per;
        fixed (float* xp0 = x, wp0 = w, bp0 = b, dp0 = dst)
        {
            float* xp = xp0, wp = wp0, bp = bp0, dp = dp0;
            Parallel.For(0, tasks, PO, t =>
            {
                for (int r = t * per, re = Math.Min(rows, r + per); r < re; r++)
                    LayerNormRow(xp + (long)r * c, dp + (long)r * c, c, wp, bp, eps);
            });
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void LayerNormRow(float* xr, float* dr, int c, float* wp, float* bp, float eps)
    {
        V4 sum = V4.Zero, sq = V4.Zero;
        int i = 0;
        for (; i + 4 <= c; i += 4)
        {
            V4 v = PackedSimd.LoadVector128(xr + i);
            sum = PackedSimd.Add(sum, v);
            sq = PackedSimd.Add(sq, PackedSimd.Multiply(v, v));
        }
        float s = Vector128.Sum(sum), ss = Vector128.Sum(sq);
        for (; i < c; i++) { float v = xr[i]; s += v; ss += v * v; }
        float mean = s / c;
        float inv = 1f / MathF.Sqrt(MathF.Max(ss / c - mean * mean, 0f) + eps);
        V4 vm = PackedSimd.Splat(mean), vi = PackedSimd.Splat(inv);
        for (i = 0; i + 4 <= c; i += 4)
        {
            V4 t = PackedSimd.Subtract(PackedSimd.LoadVector128(xr + i), vm);
            V4 g = PackedSimd.Multiply(vi, PackedSimd.LoadVector128(wp + i));
            PackedSimd.Store(dr + i, PackedSimd.Add(PackedSimd.LoadVector128(bp + i), PackedSimd.Multiply(t, g)));
        }
        for (; i < c; i++) dr[i] = (xr[i] - mean) * inv * wp[i] + bp[i];
    }
}
