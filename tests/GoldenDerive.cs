// Golden fixtures that are pure functions of {tag}_src.png and the two onnxruntime outputs
// ({tag}_shadow.f32, {tag}_output.f32), so they are recomputed instead of committed.
// Bit-exact with what tools/GenGolden fed onnxruntime. Linked into the tests and benchmarks.
namespace Sdcb.SimdGcdrNet.Testing;

internal static class GoldenDerive
{
    public const int Pad = 32;

    /// <summary>GC-Net input [1,3,h,w]: tight BGR24 / 255, top/left edge-replicate padded.</summary>
    public static float[] Input(ReadOnlySpan<byte> bgr, int oh, int ow, int ph, int pw)
    {
        int h = oh + ph, w = ow + pw;
        var x = new float[3 * h * w];
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < h; y++)
                for (int xx = 0; xx < w; xx++)
                {
                    int sy = Math.Max(y - ph, 0), sx = Math.Max(xx - pw, 0);
                    x[(c * h + y) * w + xx] = bgr[(sy * ow + sx) * 3 + c] / 255f;
                }
        return x;
    }

    /// <summary>DR-Net input [1,6,h,w] = [x; clamp(x / max(shadow, 1e-6), 0, 1)].</summary>
    public static float[] DrIn(ReadOnlySpan<float> input, ReadOnlySpan<float> shadow)
    {
        int n = input.Length;
        var d = new float[2 * n];
        input.CopyTo(d);
        for (int i = 0; i < n; i++)
            d[n + i] = Math.Clamp(input[i] / Math.Max(shadow[i], 1e-6f), 0f, 1f);
        return d;
    }

    /// <summary>Final BGR24 [oh,ow,3]: (int)(output * 255) clamped, top/left pad cropped.</summary>
    public static byte[] FinalU8(ReadOnlySpan<float> output, int oh, int ow, int ph, int pw)
    {
        int h = oh + ph, w = ow + pw;
        var u8 = new byte[oh * ow * 3];
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < oh; y++)
                for (int xx = 0; xx < ow; xx++)
                {
                    double v = output[c * h * w + (y + ph) * w + xx + pw] * 255.0;
                    u8[(y * ow + xx) * 3 + c] = (byte)Math.Clamp((int)v, 0, 255);
                }
        return u8;
    }
}
