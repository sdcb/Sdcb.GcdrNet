#:package SkiaSharp@3.119.0
using SkiaSharp;

// Mixed RealDAE corpus builder: ~17 pairs each from task_{color,shadow,bleed}_test,
// evenly spaced over the alphabetically sorted stems for size/content diversity.
// Writes NN_<type>.webp (input) + NN_gt.webp (ground truth) into the target dir
// and regenerates manifest.json. Files stay untracked (gitignored).

string src = args[0];           // e.g. C:\Users\sdfly\Downloads\RealDAE\RealDAE
string dst = args[1];           // tests/testdata/realdae
Directory.CreateDirectory(dst);

const int MaxEdge = 2600, Quality = 85;

// (stem, srcDir, type) triples, evenly spaced
var picks = new List<(string stem, string dir, string type)>();
foreach (var (type, take) in new[] { ("color", 17), ("shadow", 17), ("bleed", 16) })
{
    string dir = Path.Combine(src, $"task_{type}_test");
    var stems = Directory.EnumerateFiles(dir, "*_in.*")
        .Select(f => Path.GetFileNameWithoutExtension(f))
        .Where(n => n.EndsWith("_in"))
        .Select(n => n[..^3])
        .OrderBy(n => n, StringComparer.Ordinal)
        .ToArray();
    for (int i = 0; i < take; i++)
    {
        int idx = take == 1 ? 0 : (int)Math.Round(i * (stems.Length - 1) / (double)(take - 1));
        picks.Add((stems[idx], dir, type));
    }
}
picks = picks.OrderBy(p => p.stem, StringComparer.Ordinal).ToList();

static byte[] EncodeWebp(string path, out int w, out int h)
{
    using var bmp = SKBitmap.Decode(path);
    int ow = bmp.Width, oh = bmp.Height;
    double sc = Math.Min(1.0, MaxEdge / (double)Math.Max(ow, oh));
    w = (int)Math.Round(ow * sc); h = (int)Math.Round(oh * sc);
    using var use = sc < 1 ? bmp.Resize(new SKImageInfo(w, h), new SKSamplingOptions(SKCubicResampler.Mitchell)) : bmp;
    using var img = SKImage.FromBitmap(use);
    using var data = img.Encode(SKEncodedImageFormat.Webp, Quality);
    return data.ToArray();
}

static string? SrcFile(string dir, string stem)
{
    foreach (var ext in new[] { ".jpg", ".png" })
        if (File.Exists(Path.Combine(dir, stem + ext))) return Path.Combine(dir, stem + ext);
    return null;
}

var entries = new List<(string name, int w, int h)>();
int n = 0;
foreach (var (stem, dir, type) in picks)
{
    n++;
    string nn = $"{n:d2}";
    string? inFile = SrcFile(dir, stem + "_in"), gtFile = SrcFile(dir, stem + "_gt");
    if (inFile == null) { Console.WriteLine($"MISS {stem}_in"); continue; }
    File.WriteAllBytes(Path.Combine(dst, $"{nn}_{type}.webp"), EncodeWebp(inFile, out int w, out int h));
    entries.Add(($"{nn}_{type}.webp", w, h));
    if (gtFile != null)
        File.WriteAllBytes(Path.Combine(dst, $"{nn}_gt.webp"), EncodeWebp(gtFile, out _, out _));
    Console.WriteLine($"{nn}_{type}  <- {type}/{stem}");
}

var json = new System.Text.StringBuilder();
json.Append("{\n \"source\": \"RealDAE task_{color,shadow,bleed}_test, mixed; max edge 2600, webp q85; NN_gt.webp pairs when present (local-only data)\",\n \"images\": [\n");
for (int i = 0; i < entries.Count; i++)
{
    var e = entries[i];
    json.Append($"  {{\n   \"name\": \"{e.name}\",\n   \"w\": {e.w},\n   \"h\": {e.h}\n  }}{(i < entries.Count - 1 ? "," : "")}\n");
}
json.Append(" ]\n}\n");
File.WriteAllText(Path.Combine(dst, "manifest.json"), json.ToString());
Console.WriteLine($"manifest: {entries.Count} images");
