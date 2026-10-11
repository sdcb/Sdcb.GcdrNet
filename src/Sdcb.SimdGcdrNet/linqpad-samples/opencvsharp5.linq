<Query Kind="Program">
  <AllowUnsafe>true</AllowUnsafe>
  <NuGetReference>OpenCvSharp5</NuGetReference>
  <NuGetReference>OpenCvSharp5.runtime.osx.arm64</NuGetReference>
  <NuGetReference>OpenCvSharp5.runtime.osx.x64</NuGetReference>
  <NuGetReference>OpenCvSharp5.runtime.win</NuGetReference>
  <NuGetReference>Sdcb.SimdGcdrNet</NuGetReference>
  <NuGetReference>Sdcb.SimdGcdrNet.Models</NuGetReference>
  <Namespace>OpenCvSharp</Namespace>
  <Namespace>Sdcb.SimdGcdrNet</Namespace>
  <Namespace>Sdcb.SimdGcdrNet.Models</Namespace>
  <Namespace>System.Net.Http</Namespace>
  <Namespace>System.Threading.Tasks</Namespace>
</Query>

// OpenCvSharp: a CV_8UC3 Mat is BGR24, the model's native order. CPU vs GPU timing on the
// same image; the second call of each enhancer reuses its compiled plan.
async Task Main()
{
	using HttpClient http = new();
	byte[] webp = await http.GetByteArrayAsync("https://raw.githubusercontent.com/sdcb/SimdGcdrNet/main/tests/testdata/realdae/10_color.webp");
	using Mat src = Cv2.ImDecode(webp, ImreadModes.Color);
	if (src.Empty()) throw new InvalidDataException("Failed to decode image");
	using Mat dst = new(src.Size(), MatType.CV_8UC3);

	foreach (GcdrBackend backend in new[] { GcdrBackend.Cpu, GcdrBackend.Auto })
	{
		using GcdrEnhancer enhancer = GcdrEnhancer.FromBytes(GcdrModels.GcNet, GcdrModels.DrNet, backend);
		for (int run = 1; run <= 2; run++)
		{
			Stopwatch sw = Stopwatch.StartNew();
			Enhance(enhancer, src, dst);
			$"{backend} run {run}: {sw.ElapsedMilliseconds} ms".Dump();
		}
	}

	Util.HorizontalRun(true, Util.Image(ToJpeg(src)), Util.Image(ToJpeg(dst))).Dump();
}

static byte[] ToJpeg(Mat mat)
{
	Cv2.ImEncode(".jpg", mat, out byte[] jpeg);
	return jpeg;
}

static unsafe void Enhance(GcdrEnhancer enhancer, Mat src, Mat dst)
{
	int srcStride = (int)src.Step(), dstStride = (int)dst.Step();
	enhancer.Enhance(
		new ReadOnlySpan<byte>((byte*)src.Data, srcStride * src.Height), src.Width, src.Height, srcStride,
		new Span<byte>((byte*)dst.Data, dstStride * dst.Height), dstStride, GcdrPixelFormat.Bgr24);
}
