// Regenerates tests/testdata golden fixtures with onnxruntime (the oracle).
// Usage (from repo root):
//   dotnet run -c Release --project tools/GenGolden -- --models tests/testdata/models
// For each tests/testdata/{tag}_src.png writes:
//   {tag}_src.bgr    raw source bytes (tight HWC BGR)
//   {tag}_input.f32  padded NCHW float input (GC-Net input)
//   {tag}_shadow.f32 GC-Net output (shading map)
//   {tag}_dr_in.f32  6-channel DR-Net input [src; corrected]
//   {tag}_output.f32 DR-Net output (padded size)
//   {tag}_final.u8   cropped BGR uint8 result
//   manifest.json    dims + shapes per tag
// Padding: top/left replicate to multiples of 32 (matches GCDRNet infer.py).

using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

const int Pad = 32;

string? modelsDir = null;
string dataDir = "tests/testdata";
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--models": modelsDir = args[++i]; break;
        case "--data": dataDir = args[++i]; break;
        default: Console.Error.WriteLine($"unknown arg {args[i]}"); return 2;
    }
}
if (modelsDir == null) { Console.Error.WriteLine("--models <dir> required"); return 2; }

using var gc = new InferenceSession(Path.Combine(modelsDir, "gcdrnet_gcnet_sim.onnx"));
using var dr = new InferenceSession(Path.Combine(modelsDir, "gcdrnet_drnet_sim.onnx"));
string gcIn = gc.InputNames[0], drIn = dr.InputNames[0];

var manifest = new Dictionary<string, object>();
foreach (string png in Directory.EnumerateFiles(dataDir, "*_src.png").OrderBy(p => p))
{
    string tag = Path.GetFileName(png)[..^8];   // strip "_src.png"
    byte[] bgr;
    int oh, ow;
    using (var bmp = SKBitmap.Decode(png))
    {
        oh = bmp.Height; ow = bmp.Width;
        bgr = new byte[oh * ow * 3];
        for (int y = 0; y < oh; y++)
            for (int x0 = 0; x0 < ow; x0++)
            {
                var px = bmp.GetPixel(x0, y);    // decode normalizes to RGBA
                int d = (y * ow + x0) * 3;
                bgr[d] = px.Blue; bgr[d + 1] = px.Green; bgr[d + 2] = px.Red;
            }
    }
    File.WriteAllBytes(Path.Combine(dataDir, $"{tag}_src.bgr"), bgr);

    int ph = (Pad - oh % Pad) % Pad, pw = (Pad - ow % Pad) % Pad;
    int h = oh + ph, w = ow + pw;

    // [1,3,h,w] float, pad top/left by edge-replicate, /255
    var x = new DenseTensor<float>(new[] { 1, 3, h, w });
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < h; y++)
            for (int xx = 0; xx < w; xx++)
            {
                int sy = Math.Max(y - ph, 0), sx = Math.Max(xx - pw, 0);
                x[0, c, y, xx] = bgr[(sy * ow + sx) * 3 + c] / 255f;
            }
    WriteF32(Path.Combine(dataDir, $"{tag}_input.f32"), x);

    float[] shadow;
    using (var r = gc.Run([NamedOnnxValue.CreateFromTensor(gcIn, x)]))
        shadow = r[0].AsTensor<float>().ToArray();
    WriteF32(Path.Combine(dataDir, $"{tag}_shadow.f32"), shadow);

    // corrected = clamp(x / max(shadow, 1e-6), 0, 1); dr_in = [x; corrected]
    var drInT = new DenseTensor<float>(new[] { 1, 6, h, w });
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < h * w; i++)
        {
            float v = x[0, c, i / w, i % w];
            float s = shadow[c * h * w + i];
            drInT[0, c, i / w, i % w] = v;
            drInT[0, 3 + c, i / w, i % w] = Math.Clamp(v / Math.Max(s, 1e-6f), 0f, 1f);
        }
    WriteF32(Path.Combine(dataDir, $"{tag}_dr_in.f32"), drInT);

    float[] output;
    using (var r = dr.Run([NamedOnnxValue.CreateFromTensor(drIn, drInT)]))
        output = r[0].AsTensor<float>().ToArray();
    WriteF32(Path.Combine(dataDir, $"{tag}_output.f32"), output);

    // u8 = clip(out*255+0.5) in HWC BGR, crop top/left pad
    var u8 = new byte[oh * ow * 3];
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < oh; y++)
            for (int xx = 0; xx < ow; xx++)
            {
                double v = output[c * h * w + (y + ph) * w + xx + pw] * 255.0;
                u8[(y * ow + xx) * 3 + c] = (byte)Math.Clamp((int)v, 0, 255);
            }
    File.WriteAllBytes(Path.Combine(dataDir, $"{tag}_final.u8"), u8);

    manifest[tag] = new
    {
        orig_h = oh,
        orig_w = ow,
        ph,
        pw,
        h,
        w,
        shapes = new
        {
            input = new[] { 1, 3, h, w },
            shadow = new[] { 1, 3, h, w },
            dr_in = new[] { 1, 6, h, w },
            output = new[] { 1, 3, h, w },
        },
        final_u8_shape = new[] { oh, ow, 3 },
    };
    Console.WriteLine($"{tag}: {ow}x{oh} -> {w}x{h} pad=({ph},{pw})");
}

File.WriteAllText(Path.Combine(dataDir, "manifest.json"),
    JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true, IndentSize = 1 }));
return 0;

static void WriteF32(string path, IEnumerable<float> a)
{
    using var fs = File.Create(path);
    using var bw = new BinaryWriter(fs);
    foreach (float v in a) bw.Write(v);
}
