namespace Sdcb.SimdGcdrNet;

/// <summary>
/// Interleaved 8-bit pixel layouts accepted by <see cref="GcdrEnhancer.Enhance(ReadOnlySpan{byte}, int, int, int, Span{byte}, int, GcdrPixelFormat)"/>.
/// The output is written in the same layout as the input.
/// </summary>
public enum GcdrPixelFormat
{
    /// <summary>3 bytes, B,G,R. OpenCV <c>Mat</c> / GDI+ 24bpp memory order.</summary>
    Bgr24 = 0,
    /// <summary>3 bytes, R,G,B.</summary>
    Rgb24 = 1,
    /// <summary>4 bytes, B,G,R,A. SkiaSharp <c>Bgra8888</c> / GDI+ 32bpp ARGB memory order; alpha is copied through.</summary>
    Bgra32 = 2,
    /// <summary>4 bytes, R,G,B,A. Browser canvas <c>ImageData</c> / ImageSharp <c>Rgba32</c>; alpha is copied through.</summary>
    Rgba32 = 3,
}

internal static class GcdrPixelFormatExtensions
{
    public static int BytesPerPixel(this GcdrPixelFormat f) => f switch
    {
        GcdrPixelFormat.Bgr24 or GcdrPixelFormat.Rgb24 => 3,
        GcdrPixelFormat.Bgra32 or GcdrPixelFormat.Rgba32 => 4,
        _ => throw new ArgumentOutOfRangeException(nameof(f), f, "unknown pixel format"),
    };

    /// <summary>R and B swapped relative to the model's BGR channel order.</summary>
    public static bool SwapRB(this GcdrPixelFormat f) => f is GcdrPixelFormat.Rgb24 or GcdrPixelFormat.Rgba32;
}
