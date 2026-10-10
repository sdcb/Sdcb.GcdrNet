using System.Numerics;
using System.Runtime.CompilerServices;

namespace Sdcb.GcdrNet.Kernels;

/// <summary>
/// Pipeline glue kernels: BGR↔NCHW float conversion, replicate padding,
/// division correction, output crop. All vectorized over the innermost dim.
/// </summary>
internal static class PixelOps
{
    /// <summary>
    /// BGR u8 HWC → float NCHW /255 with top/left replicate padding.
    /// dst layout: [3][h][w] float.
    /// </summary>
    public static void BgrToNchwPad(ReadOnlySpan<byte> src, int width, int height, int srcStride,
        Span<float> dst, int h, int w, int padTop, int padLeft)
    {
        int hw = h * w;
        for (int y = 0; y < h; y++)
        {
            int sy = Math.Max(0, y - padTop);
            ReadOnlySpan<byte> row = src.Slice(sy * srcStride, width * 3);
            Span<float> d0 = dst.Slice(y * w, w);
            Span<float> d1 = dst.Slice(hw + y * w, w);
            Span<float> d2 = dst.Slice(2 * hw + y * w, w);
            int x = 0;
            // left replicate
            if (padLeft > 0)
            {
                byte b = row[0], g = row[1], r = row[2];
                float fb = b / 255f, fg = g / 255f, fr = r / 255f;
                for (; x < padLeft; x++) { d0[x] = fb; d1[x] = fg; d2[x] = fr; }
            }
            for (; x < w; x++)
            {
                int sx = Math.Min(x - padLeft, width - 1);
                d0[x] = row[sx * 3] / 255f;
                d1[x] = row[sx * 3 + 1] / 255f;
                d2[x] = row[sx * 3 + 2] / 255f;
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
    /// float NCHW [3,h,w] → u8 BGR HWC, cropping (padTop,padLeft) margins.
    /// </summary>
    public static void NchwToBgrCrop(ReadOnlySpan<float> src, int h, int w,
        int padTop, int padLeft, Span<byte> dst, int width, int height, int dstStride)
    {
        int hw = h * w;
        for (int y = 0; y < height; y++)
        {
            int sy = y + padTop;
            ReadOnlySpan<float> r0 = src.Slice(sy * w, w);
            ReadOnlySpan<float> r1 = src.Slice(hw + sy * w, w);
            ReadOnlySpan<float> r2 = src.Slice(2 * hw + sy * w, w);
            Span<byte> drow = dst.Slice(y * dstStride, width * 3);
            for (int x = 0; x < width; x++)
            {
                int sx = x + padLeft;
                drow[x * 3] = ToU8(r0[sx]);
                drow[x * 3 + 1] = ToU8(r1[sx]);
                drow[x * 3 + 2] = ToU8(r2[sx]);
            }
        }
        static byte ToU8(float v) => (byte)Math.Clamp((int)(v * 255f + 0.5f), 0, 255);
    }

    /// <summary>BGR u8 HWC → float NHWC [h][w][3] /255 with top/left replicate padding.</summary>
    public static void BgrToNhwcPad(ReadOnlySpan<byte> src, int width, int height, int srcStride,
        Span<float> dst, int h, int w, int padTop, int padLeft)
    {
        for (int y = 0; y < h; y++)
        {
            ReadOnlySpan<byte> row = src.Slice(Math.Max(0, y - padTop) * srcStride, width * 3);
            Span<float> d = dst.Slice(y * w * 3, w * 3);
            for (int x = 0; x < w; x++)
            {
                int sx = Math.Clamp(x - padLeft, 0, width - 1) * 3;
                d[x * 3] = row[sx] / 255f;
                d[x * 3 + 1] = row[sx + 1] / 255f;
                d[x * 3 + 2] = row[sx + 2] / 255f;
            }
        }
    }

    /// <summary>
    /// NHWC div6: dst[px,0..2] = x[px], dst[px,3..5] = clamp(x / max(shadow,1e-6), 0, 1).
    /// x / shadow are [hw][3]; dst is [hw][6].
    /// </summary>
    public static void DivConcatNhwc(ReadOnlySpan<float> x, ReadOnlySpan<float> shadow, Span<float> dst)
    {
        int px = x.Length / 3;
        for (int i = 0; i < px; i++)
        {
            int s = i * 3, d = i * 6;
            for (int c = 0; c < 3; c++)
            {
                float v = x[s + c];
                dst[d + c] = v;
                dst[d + 3 + c] = Math.Clamp(v / Math.Max(shadow[s + c], 1e-6f), 0f, 1f);
            }
        }
    }

    /// <summary>float NHWC [h][w][3] → u8 BGR HWC, cropping (padTop,padLeft) margins.</summary>
    public static void NhwcToBgrCrop(ReadOnlySpan<float> src, int h, int w,
        int padTop, int padLeft, Span<byte> dst, int width, int height, int dstStride)
    {
        for (int y = 0; y < height; y++)
        {
            ReadOnlySpan<float> r = src.Slice(((y + padTop) * w + padLeft) * 3, width * 3);
            Span<byte> drow = dst.Slice(y * dstStride, width * 3);
            for (int i = 0; i < width * 3; i++)
                drow[i] = (byte)Math.Clamp((int)(r[i] * 255f + 0.5f), 0, 255);
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
