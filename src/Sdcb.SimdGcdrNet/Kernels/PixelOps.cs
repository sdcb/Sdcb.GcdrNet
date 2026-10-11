using System.Numerics;
using System.Runtime.CompilerServices;

namespace Sdcb.SimdGcdrNet.Kernels;

/// <summary>
/// Pipeline glue kernels: BGR↔NCHW float conversion, replicate padding,
/// division correction, output crop. All vectorized over the innermost dim.
/// </summary>
internal static class PixelOps
{
    /// <summary>
    /// Interleaved u8 HWC (<paramref name="bpp"/> 3 or 4 bytes per pixel, B,G,R order unless
    /// <paramref name="swapRB"/>) → float NCHW /255 with top/left replicate padding.
    /// dst layout: [3][h][w] float, channels B,G,R.
    /// </summary>
    public static void BgrToNchwPad(ReadOnlySpan<byte> src, int width, int height, int srcStride,
        Span<float> dst, int h, int w, int padTop, int padLeft, int bpp = 3, bool swapRB = false)
    {
        int hw = h * w;
        int ob = swapRB ? 2 : 0, or = 2 - ob;
        for (int y = 0; y < h; y++)
        {
            int sy = Math.Max(0, y - padTop);
            ReadOnlySpan<byte> row = src.Slice(sy * srcStride, width * bpp);
            Span<float> d0 = dst.Slice(y * w, w);
            Span<float> d1 = dst.Slice(hw + y * w, w);
            Span<float> d2 = dst.Slice(2 * hw + y * w, w);
            for (int x = 0; x < w; x++)
            {
                int sx = Math.Clamp(x - padLeft, 0, width - 1) * bpp;
                d0[x] = row[sx + ob] / 255f;
                d1[x] = row[sx + 1] / 255f;
                d2[x] = row[sx + or] / 255f;
            }
        }
    }

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

    /// <summary>
    /// float NCHW [3,h,w] (B,G,R) → interleaved u8 HWC, cropping (padTop,padLeft) margins.
    /// <paramref name="bpp"/> 4: the alpha byte of each pixel is copied from
    /// <paramref name="alphaSrc"/> (same layout, row stride <paramref name="alphaStride"/>).
    /// </summary>
    public static void NchwToBgrCrop(ReadOnlySpan<float> src, int h, int w,
        int padTop, int padLeft, Span<byte> dst, int width, int height, int dstStride,
        int bpp = 3, bool swapRB = false, ReadOnlySpan<byte> alphaSrc = default, int alphaStride = 0)
    {
        int hw = h * w;
        int ob = swapRB ? 2 : 0, or = 2 - ob;
        for (int y = 0; y < height; y++)
        {
            int sy = y + padTop;
            ReadOnlySpan<float> r0 = src.Slice(sy * w, w);
            ReadOnlySpan<float> r1 = src.Slice(hw + sy * w, w);
            ReadOnlySpan<float> r2 = src.Slice(2 * hw + sy * w, w);
            Span<byte> drow = dst.Slice(y * dstStride, width * bpp);
            ReadOnlySpan<byte> arow = bpp == 4 ? alphaSrc.Slice(y * alphaStride, width * 4) : default;
            for (int x = 0; x < width; x++)
            {
                int sx = x + padLeft, d = x * bpp;
                drow[d + ob] = ToU8(r0[sx]);
                drow[d + 1] = ToU8(r1[sx]);
                drow[d + or] = ToU8(r2[sx]);
                if (bpp == 4) drow[d + 3] = arow[d + 3];
            }
        }
        static byte ToU8(float v) => (byte)Math.Clamp((int)(v * 255f + 0.5f), 0, 255);
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
    /// Interleaved u8 HWC (<paramref name="bpp"/> 3 or 4 bytes per pixel, B,G,R order unless
    /// <paramref name="swapRB"/>) → float NHWC [h][w][3] (B,G,R) /255 with top/left replicate padding.
    /// </summary>
    public static unsafe void BgrToNhwcPad(ReadOnlySpan<byte> src, int width, int height, int srcStride,
        Span<float> dst, int h, int w, int padTop, int padLeft, int bpp = 3, bool swapRB = false)
    {
        if (src.Length < (long)(height - 1) * srcStride + width * bpp || dst.Length < (long)h * w * 3)
            throw new ArgumentException("pad buffer sizes");
        int ob = swapRB ? 2 : 0, or = 2 - ob;
        fixed (byte* sp0 = src)
        fixed (float* dp0 = dst)
        {
            byte* sp = sp0;
            float* dp = dp0;
            Rows(h, (long)w * 3, (y0, y1) =>
            {
                for (int y = y0; y < y1; y++)
                {
                    byte* row = sp + (long)Math.Max(0, y - padTop) * srcStride;
                    float* d = dp + (long)y * w * 3;
                    for (int x = 0; x < w; x++)
                    {
                        int sx = Math.Clamp(x - padLeft, 0, width - 1) * bpp;
                        d[x * 3] = row[sx + ob] / 255f;
                        d[x * 3 + 1] = row[sx + 1] / 255f;
                        d[x * 3 + 2] = row[sx + or] / 255f;
                    }
                }
            });
        }
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

    /// <summary>
    /// float NHWC [h][w][3] (B,G,R) → interleaved u8 HWC, cropping (padTop,padLeft) margins.
    /// <paramref name="bpp"/> 4: the alpha byte of each pixel is copied from
    /// <paramref name="alphaSrc"/> (same layout, row stride <paramref name="alphaStride"/>;
    /// may alias <paramref name="dst"/>).
    /// </summary>
    public static unsafe void NhwcToBgrCrop(ReadOnlySpan<float> src, int h, int w,
        int padTop, int padLeft, Span<byte> dst, int width, int height, int dstStride,
        int bpp = 3, bool swapRB = false, ReadOnlySpan<byte> alphaSrc = default, int alphaStride = 0)
    {
        if (src.Length < (long)h * w * 3 || dst.Length < (long)(height - 1) * dstStride + width * bpp)
            throw new ArgumentException("crop buffer sizes");
        if (bpp == 4 && alphaSrc.Length < (long)(height - 1) * alphaStride + width * 4)
            throw new ArgumentException("alpha source size");
        int ob = swapRB ? 2 : 0, or = 2 - ob;
        fixed (float* sp0 = src)
        fixed (byte* dp0 = dst, ap0 = alphaSrc)
        {
            float* sp = sp0;
            byte* dp = dp0, ap = ap0;
            Rows(height, (long)width * 3, (y0, y1) =>
            {
                for (int y = y0; y < y1; y++)
                {
                    float* r = sp + ((long)(y + padTop) * w + padLeft) * 3;
                    byte* drow = dp + (long)y * dstStride;
                    if (bpp == 3 && !swapRB)
                    {
                        for (int i = 0; i < width * 3; i++)
                            drow[i] = ToU8(r[i]);
                        continue;
                    }
                    byte* arow = ap + (long)y * alphaStride;
                    for (int x = 0; x < width; x++)
                    {
                        byte* d = drow + x * bpp;
                        d[ob] = ToU8(r[x * 3]);
                        d[1] = ToU8(r[x * 3 + 1]);
                        d[or] = ToU8(r[x * 3 + 2]);
                        if (bpp == 4) d[3] = arow[x * 4 + 3];
                    }
                }
            });
        }
        static byte ToU8(float v) => (byte)Math.Clamp((int)(v * 255f + 0.5f), 0, 255);
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
