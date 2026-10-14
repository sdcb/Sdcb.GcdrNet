using System.Text.Json;
using SkiaSharp;

namespace Sdcb.SimdGcdrNet.Tests;

/// <summary>
/// Throughput/robustness smoke over the RealDAE corpus (tests/testdata/realdae, fetched from
/// sdcb/DocGainFixture by tools/fetch-fixture.ps1; skipped when absent):
/// decode webp -> enhance -> sanity-check the output. A spread of ~10 images
/// covering small to 2600px inputs; not an oracle test (paired NN_gt.webp are
/// dataset ground truths, not implementation golden — PSNR lives in bench --corpus).
/// </summary>
public class CorpusTests
{
    private static (string file, int w, int h)[] Entries()
    {
        string dir = Golden.Dir;
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "realdae", "manifest.json")));
        return doc.RootElement.GetProperty("images").EnumerateArray()
            .Select(e => (e.GetProperty("name").GetString()!,
                          e.GetProperty("w").GetInt32(),
                          e.GetProperty("h").GetInt32()))
            .ToArray();
    }

    internal static byte[] DecodeBgr(string path, out int w, out int h)
    {
        using var decoded = SKBitmap.Decode(path) ?? throw new InvalidDataException($"cannot decode {path}");
        // the platform color type is RGBA on Linux: force BGRA so the raw bytes below are B,G,R
        using var bmp = decoded.ColorType == SKColorType.Bgra8888 ? decoded.Copy()
            : decoded.Copy(SKColorType.Bgra8888) ?? throw new InvalidDataException($"cannot convert {path}");
        w = bmp.Width; h = bmp.Height;
        var bgr = new byte[w * h * 3];
        IntPtr px = bmp.GetPixels();
        int rb = bmp.RowBytes;
        unsafe
        {
            for (int y = 0; y < h; y++)
            {
                byte* s = (byte*)px + y * rb;
                int d = y * w * 3;
                for (int x = 0; x < w; x++, s += 4, d += 3)
                {
                    bgr[d] = s[0]; bgr[d + 1] = s[1]; bgr[d + 2] = s[2];
                }
            }
        }
        return bgr;
    }

    internal static bool Present => File.Exists(Path.Combine(Golden.Dir, "realdae", "manifest.json"));

    public static IEnumerable<object[]> Picks()
    {
        if (!Present)
        {
            yield return new object[] { "(missing)", 0, 0 };
            yield break;
        }
        var all = Entries();
        int step = Math.Max(1, all.Length / 10);
        for (int i = 0; i < all.Length; i += step)
            yield return new object[] { all[i].file, all[i].w, all[i].h };
    }

    [SkippableTheory]
    [MemberData(nameof(Picks))]
    public void Corpus_Enhance_Smoke(string file, int w, int h)
    {
        Skip.IfNot(Present, "RealDAE corpus not present: run tools/fetch-fixture.ps1");
        string path = Path.Combine(Golden.Dir, "realdae", file);
        byte[] bgr = DecodeBgr(path, out int dw, out int dh);
        Assert.Equal((w, h), (dw, dh));

        using var enh = GcdrEnhancer.FromFiles(EngineTests.GcPath, EngineTests.DrPath);
        byte[] dst = enh.Enhance(bgr, w, h);
        Assert.Equal(w * h * 3, dst.Length);

        // sanity: output is a real image — differs from input somewhere and is
        // not a degenerate constant frame
        Assert.False(dst.AsSpan().SequenceEqual(bgr), $"{file}: output identical to input");
        long sum = 0, sq = 0;
        foreach (byte v in dst) { sum += v; sq += v * v; }
        double mean = sum / (double)dst.Length;
        double std = Math.Sqrt(sq / (double)dst.Length - mean * mean);
        Assert.True(std > 1.0, $"{file}: output std {std:F2} looks degenerate");
    }
}
