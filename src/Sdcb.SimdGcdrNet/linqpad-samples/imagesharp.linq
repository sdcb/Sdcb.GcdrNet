<Query Kind="Program">
  <NuGetReference>Sdcb.SimdGcdrNet</NuGetReference>
  <NuGetReference>Sdcb.SimdGcdrNet.Models</NuGetReference>
  <NuGetReference Version="3.1.11">SixLabors.ImageSharp</NuGetReference>
  <Namespace>Sdcb.SimdGcdrNet</Namespace>
  <Namespace>Sdcb.SimdGcdrNet.Models</Namespace>
  <Namespace>SixLabors.ImageSharp</Namespace>
  <Namespace>SixLabors.ImageSharp.Formats</Namespace>
  <Namespace>SixLabors.ImageSharp.PixelFormats</Namespace>
  <Namespace>System.Net.Http</Namespace>
  <Namespace>System.Runtime.InteropServices</Namespace>
  <Namespace>System.Threading.Tasks</Namespace>
</Query>

// ImageSharp (cross-platform, pure managed): enhance an Image<Rgba32> in place.
async Task Main()
{
	using HttpClient http = new();
	byte[] webp = await http.GetByteArrayAsync("https://raw.githubusercontent.com/sdcb/SimdGcdrNet/main/tests/testdata/realdae/10_color.webp");

	using GcdrEnhancer enhancer = GcdrEnhancer.FromStreams(GcdrModels.OpenGcNet, GcdrModels.OpenDrNet, GcdrBackend.Auto);
	using Image<Rgba32> image = Image.Load<Rgba32>(CreateDecoderOptions(), webp);
	byte[] before = ToJpeg(image);

	if (!image.DangerousTryGetSinglePixelMemory(out Memory<Rgba32> memory))
		throw new InvalidOperationException("pixel buffer is not contiguous");
	Span<byte> pixels = MemoryMarshal.AsBytes(memory.Span);

	Stopwatch sw = Stopwatch.StartNew();
	enhancer.Enhance(pixels, image.Width, image.Height, image.Width * 4, pixels, image.Width * 4, GcdrPixelFormat.Rgba32);
	$"{image.Width}x{image.Height} enhanced in {sw.ElapsedMilliseconds} ms (first call includes plan compilation)".Dump();

	Util.HorizontalRun(true, Util.Image(before), Util.Image(ToJpeg(image))).Dump();
}

static byte[] ToJpeg(Image image)
{
	using MemoryStream ms = new();
	image.SaveAsJpeg(ms);
	return ms.ToArray();
}

// Clone Default so the decoders stay registered. The default allocator splits large images
// into 4 MB chunks, which DangerousTryGetSinglePixelMemory cannot return as one span.
static DecoderOptions CreateDecoderOptions()
{
	Configuration configuration = Configuration.Default.Clone();
	configuration.PreferContiguousImageBuffers = true;
	return new DecoderOptions { Configuration = configuration };
}
