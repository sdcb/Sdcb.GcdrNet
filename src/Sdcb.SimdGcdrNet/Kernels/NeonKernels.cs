using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

using V4 = System.Runtime.Intrinsics.Vector128<float>;

namespace Sdcb.SimdGcdrNet.Kernels;

/// <summary>AArch64 AdvSimd (NEON) versions of the non-conv ops that have AVX2 kernels.</summary>
internal static unsafe class NeonKernels
{
    private static readonly ParallelOptions PO = new() { MaxDegreeOfParallelism = Avx2Kernels.Threads };

    /// <summary>Row-wise LayerNorm over the last axis (c); rows split across threads in groups.</summary>
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
            V4 v = AdvSimd.LoadVector128(xr + i);
            sum = AdvSimd.Add(sum, v);
            sq = AdvSimd.FusedMultiplyAdd(sq, v, v);
        }
        float s = Vector128.Sum(sum), ss = Vector128.Sum(sq);
        for (; i < c; i++) { float v = xr[i]; s += v; ss += v * v; }
        float mean = s / c;
        float inv = 1f / MathF.Sqrt(MathF.Max(ss / c - mean * mean, 0f) + eps);
        V4 vm = Vector128.Create(mean), vi = Vector128.Create(inv);
        for (i = 0; i + 4 <= c; i += 4)
        {
            V4 t = AdvSimd.Subtract(AdvSimd.LoadVector128(xr + i), vm);
            AdvSimd.Store(dr + i, AdvSimd.FusedMultiplyAdd(AdvSimd.LoadVector128(bp + i), t,
                AdvSimd.Multiply(vi, AdvSimd.LoadVector128(wp + i))));
        }
        for (; i < c; i++) dr[i] = (xr[i] - mean) * inv * wp[i] + bp[i];
    }
}
