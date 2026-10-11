<Query Kind="Program">
  <AllowUnsafe>true</AllowUnsafe>
  <NuGetReference>Sdcb.SimdGcdrNet</NuGetReference>
  <NuGetReference>Sdcb.SimdGcdrNet.Models</NuGetReference>
  <NuGetReference>Sdcb.SimdPaddleOCR</NuGetReference>
  <NuGetReference>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</NuGetReference>
  <NuGetReference Version="4.151.1">SkiaSharp</NuGetReference>
  <Namespace>Sdcb.SimdGcdrNet</Namespace>
  <Namespace>Sdcb.SimdGcdrNet.Models</Namespace>
  <Namespace>Sdcb.SimdPaddleOCR</Namespace>
  <Namespace>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</Namespace>
  <Namespace>SkiaSharp</Namespace>
  <Namespace>System.Net.Http</Namespace>
  <Namespace>System.Threading.Tasks</Namespace>
</Query>

// GcdrNet + SimdPaddleOCR: remove shadows from a photographed page, then OCR both versions.
// Both libraries take raw interleaved pixels, so one decoded SKBitmap (BGRA32) flows through
// enhancement and recognition without re-encoding.
async Task Main()
{
	using HttpClient http = new();
	byte[] webp = await http.GetByteArrayAsync("https://raw.githubusercontent.com/sdcb/SimdGcdrNet/main/tests/testdata/realdae/10_color.webp");

	using GcdrEnhancer enhancer = GcdrEnhancer.FromStreams(GcdrModels.OpenGcNet, GcdrModels.OpenDrNet, GcdrBackend.Auto);
	using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default);
	using SKBitmap original = DecodeBgra(webp);
	using SKBitmap enhanced = new(original.Info);

	Stopwatch sw = Stopwatch.StartNew();
	unsafe
	{
		enhancer.Enhance(
			new ReadOnlySpan<byte>((byte*)original.GetPixels(), original.RowBytes * original.Height),
			original.Width, original.Height, original.RowBytes,
			new Span<byte>((byte*)enhanced.GetPixels(), enhanced.RowBytes * enhanced.Height),
			enhanced.RowBytes, GcdrPixelFormat.Bgra32);
	}
	$"{original.Width}x{original.Height} enhanced in {sw.ElapsedMilliseconds} ms".Dump();

	Util.HorizontalRun(true,
		new { Image = "original", Preview = Util.Image(ToJpeg(original)), Ocr = Ocr(ocr, original) },
		new { Image = "enhanced", Preview = Util.Image(ToJpeg(enhanced)), Ocr = Ocr(ocr, enhanced) }).Dump();
}

static unsafe string Ocr(PaddleOcrAll ocr, SKBitmap bitmap)
{
	PaddleOcrResult result = ocr.Run(
		new ReadOnlySpan<byte>((byte*)bitmap.GetPixels(), bitmap.RowBytes * bitmap.Height),
		bitmap.Width, bitmap.Height, bitmap.RowBytes, ImagePixelFormat.Bgra32);
	return result.Text;
}

static byte[] ToJpeg(SKBitmap bitmap)
{
	using SKData data = bitmap.Encode(SKEncodedImageFormat.Jpeg, 90);
	return data.ToArray();
}

static SKBitmap DecodeBgra(byte[] encoded)
{
	SKBitmap decoded = SKBitmap.Decode(encoded) ?? throw new InvalidDataException("Failed to decode image");
	if (decoded.ColorType == SKColorType.Bgra8888) return decoded;
	SKBitmap? bgra = decoded.Copy(SKColorType.Bgra8888);
	decoded.Dispose();
	return bgra ?? throw new InvalidDataException("Failed to convert to BGRA");
}
