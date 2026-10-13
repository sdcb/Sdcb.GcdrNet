using System.Diagnostics;
using System.Runtime.InteropServices;
using Sdcb.SimdGcdrNet;
using Sdcb.SimdGcdrNet.Models;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

// usage: PaddleOCR.Console [image] [output]
// Cleans a photographed page with SimdGcdrNet (shadows, uneven lighting, show-through from the
// back side), then runs SimdPaddleOCR on the original and the enhanced image. The default
// ocr-example.jpg has heavy show-through, which OCR reads as garbage lines on the original.
// Both libraries take raw interleaved pixels, so the decoded RGBA buffer is shared without re-encoding.
string input = args.Length > 0 ? args[0] : Path.Combine(AppContext.BaseDirectory, "ocr-example.jpg");
string output = args.Length > 1 ? args[1] : Path.ChangeExtension(Path.GetFileName(input), null) + "-enhanced.jpg";
Console.OutputEncoding = System.Text.Encoding.UTF8;

DecoderOptions decoderOptions = new() { Configuration = Configuration.Default.Clone() };
// One contiguous pixel buffer, so both libraries can read it as a single span.
decoderOptions.Configuration.PreferContiguousImageBuffers = true;

using Image<Rgba32> original = Image.Load<Rgba32>(decoderOptions, input);
original.Mutate(x => x.AutoOrient());
using Image<Rgba32> enhanced = original.Clone();
int width = original.Width, height = original.Height, stride = width * 4;
Console.WriteLine($"{input}: {width}x{height}");

using GcdrEnhancer enhancer = GcdrEnhancer.FromBytes(GcdrModels.GcNet, GcdrModels.DrNet, GcdrBackend.Auto);
using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default);

Stopwatch sw = Stopwatch.StartNew();
Span<byte> pixels = Pixels(enhanced);
enhancer.Enhance(pixels, width, height, stride, pixels, stride, GcdrPixelFormat.Rgba32);
Console.WriteLine($"enhanced in {sw.ElapsedMilliseconds} ms (first call includes plan compilation)");
enhanced.Save(output);
Console.WriteLine($"saved {Path.GetFullPath(output)}");

PaddleOcrResult before = ocr.Run(Pixels(original), width, height, stride, ImagePixelFormat.Rgba32);
PaddleOcrResult after = ocr.Run(Pixels(enhanced), width, height, stride, ImagePixelFormat.Rgba32);
Print("original", before);
Print("enhanced", after);

Console.WriteLine();
Console.WriteLine($"{"",-10}{"lines",8}{"chars",8}");
Summary("original", before);
Summary("enhanced", after);

static Span<byte> Pixels(Image<Rgba32> image) => image.DangerousTryGetSinglePixelMemory(out Memory<Rgba32> memory)
    ? MemoryMarshal.AsBytes(memory.Span)
    : throw new InvalidOperationException("pixel buffer is not contiguous");

static void Print(string title, PaddleOcrResult result)
{
    Console.WriteLine();
    Console.WriteLine($"===== {title} =====");
    foreach (PaddleOcrLine line in result.Lines)
        Console.WriteLine(line.Text);
}

static void Summary(string title, PaddleOcrResult result) =>
    Console.WriteLine($"{title,-10}{result.Lines.Length,8}{result.Lines.Sum(l => l.Text.Length),8}");
