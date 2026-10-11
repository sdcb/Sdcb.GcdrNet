<Query Kind="Program">
  <AllowUnsafe>true</AllowUnsafe>
  <Reference>&lt;RuntimeDirectory&gt;\System.Drawing.dll</Reference>
  <NuGetReference>Sdcb.GcdrNet</NuGetReference>
  <NuGetReference>Sdcb.GcdrNet.Models</NuGetReference>
  <Namespace>Sdcb.GcdrNet</Namespace>
  <Namespace>Sdcb.GcdrNet.Models</Namespace>
  <Namespace>System.Drawing</Namespace>
  <Namespace>System.Drawing.Imaging</Namespace>
  <Namespace>System.Net.Http</Namespace>
  <Namespace>System.Threading.Tasks</Namespace>
</Query>

// System.Drawing (Windows): a Format24bppRgb bitmap is BGR24 in memory, so LockBits
// hands the pixels to GcdrNet with zero copies; the result is written back in place.
async Task Main()
{
	using HttpClient http = new();
	byte[] png = await http.GetByteArrayAsync("https://raw.githubusercontent.com/sdcb/Sdcb.GcdrNet/main/tests/testdata/mid_src.png");

	using GcdrEnhancer enhancer = GcdrModels.CreateEnhancer(GcdrBackend.Auto);
	using MemoryStream stream = new(png);
	using Bitmap original = new(stream);
	using Bitmap enhanced = original.Clone(new Rectangle(0, 0, original.Width, original.Height), PixelFormat.Format24bppRgb);

	Stopwatch sw = Stopwatch.StartNew();
	BitmapData data = enhanced.LockBits(new Rectangle(0, 0, enhanced.Width, enhanced.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
	try
	{
		unsafe
		{
			Span<byte> pixels = new((byte*)data.Scan0, data.Stride * data.Height);
			enhancer.Enhance(pixels, data.Width, data.Height, data.Stride, pixels, data.Stride, GcdrPixelFormat.Bgr24);
		}
	}
	finally
	{
		enhanced.UnlockBits(data);
	}
	$"{enhanced.Width}x{enhanced.Height} enhanced in {sw.ElapsedMilliseconds} ms (first call includes plan compilation)".Dump();

	Util.HorizontalRun(true, original, enhanced).Dump();
}
