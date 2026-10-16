using System.Numerics;
using System.Runtime.CompilerServices;
using Sdcb.SimdOnnx.Kernels;

namespace Sdcb.SimdGcdrNet.Kernels;

/// <summary>
/// Pipeline glue kernels between the two nets (division correction, concat) and the Metal
/// pipeline's BGR24 rows; pixels ↔ tensors go through Sdcb.SimdOnnx.Imaging.ImageTensor.
/// </summary>
internal static class PixelOps
{
    /// <summary>
    /// corrected = clamp(x / max(shadow, 1e-6), 0, 1) for all 3·hw elements.
    /// </summary>
    public static void DivClampInto(ReadOnlySpan<float> x, ReadOnlySpan<float> shadow,
        Span<float> dst)
    {
        int n = x.Length, i = 0, VL = Vector<float>.Count;
        var eps = new Vector<float>(1e-6f);
        var one = Vector<float>.One;
        var zero = Vector<float>.Zero;
        for (; i + VL <= n; i += VL)
        {
            var xv = Simd.VectorLoad(x, i);
            var sv = Vector.Max(Simd.VectorLoad(shadow, i), eps);
            var r = Vector.Min(Vector.Max(xv / sv, zero), one);
            Simd.VectorStore(dst, i, r);
        }
        for (; i < n; i++)
            dst[i] = Math.Clamp(x[i] / Math.Max(shadow[i], 1e-6f), 0f, 1f);
    }

    // The NHWC glue below runs once per image on the calling thread between the two nets;
    // single-threaded it was ~90 ms of a ~630 ms 3.7 MP frame on 32 threads. Every element is
    // independent, so row bands go wide (same per-element ops, same bits).
    private static readonly ParallelOptions PO = new() { MaxDegreeOfParallelism = Avx2Kernels.Threads };

    /// <summary>Runs body(r0, r1) over [0, rows) in bands of at least ~64K elements each.</summary>
    private static void Rows(int rows, long elemsPerRow, Action<int, int> body)
    {
        int bands = (int)Math.Clamp(rows * elemsPerRow / 65536, 1, Math.Min(rows, 4L * Avx2Kernels.Threads));
        if (bands == 1) { body(0, rows); return; }
        Parallel.For(0, bands, PO, b => body((int)((long)rows * b / bands), (int)((long)rows * (b + 1) / bands)));
    }

    /// <summary>
    /// NHWC div6: dst[px,0..2] = x[px], dst[px,3..5] = clamp(x / max(shadow,1e-6), 0, 1).
    /// x / shadow are [hw][3]; dst is [hw][6].
    /// </summary>
    public static unsafe void DivConcatNhwc(ReadOnlySpan<float> x, ReadOnlySpan<float> shadow, Span<float> dst)
    {
        int px = x.Length / 3;
        if (shadow.Length < px * 3 || dst.Length < px * 6) throw new ArgumentException("div6 buffer sizes");
        fixed (float* xp0 = x, sp0 = shadow, dp0 = dst)
        {
            float* xp = xp0, sp = sp0, dp = dp0;
            Rows(px, 6, (i0, i1) =>
            {
                for (int i = i0; i < i1; i++)
                {
                    int s = i * 3, d = i * 6;
                    for (int c = 0; c < 3; c++)
                    {
                        float v = xp[s + c];
                        dp[d + c] = v;
                        dp[d + 3 + c] = Math.Clamp(v / Math.Max(sp[s + c], 1e-6f), 0f, 1f);
                    }
                }
            });
        }
    }

    /// <summary>One row of interleaved u8 pixels in <paramref name="format"/> → tightly packed BGR24.</summary>
    public static void RowToBgr24(ReadOnlySpan<byte> src, Span<byte> bgr, int width, GcdrPixelFormat format)
    {
        int bpp = format.BytesPerPixel();
        int ob = format.SwapRB() ? 2 : 0, or = 2 - ob;
        for (int x = 0; x < width; x++)
        {
            int s = x * bpp, d = x * 3;
            bgr[d] = src[s + ob];
            bgr[d + 1] = src[s + 1];
            bgr[d + 2] = src[s + or];
        }
    }

    /// <summary>
    /// One row of tightly packed BGR24 → <paramref name="format"/>; 4-byte formats take each
    /// pixel's alpha from <paramref name="alphaRow"/> (same format as the output, may alias it).
    /// </summary>
    public static void RowFromBgr24(ReadOnlySpan<byte> bgr, Span<byte> dst, int width, GcdrPixelFormat format,
        ReadOnlySpan<byte> alphaRow)
    {
        int bpp = format.BytesPerPixel();
        int ob = format.SwapRB() ? 2 : 0, or = 2 - ob;
        for (int x = 0; x < width; x++)
        {
            int s = x * 3, d = x * bpp;
            dst[d + ob] = bgr[s];
            dst[d + 1] = bgr[s + 1];
            dst[d + or] = bgr[s + 2];
            if (bpp == 4) dst[d + 3] = alphaRow[d + 3];
        }
    }

    /// <summary>float NCHW ↔ raw grayscale compare helper used in tests.</summary>
    public static float MaxDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }
}
