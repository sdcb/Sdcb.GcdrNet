using Sdcb.GcdrNet;
using System.Buffers.Binary;
using System.Text;

// usage: EnhanceImage <gcnet.onnx> <drnet.onnx> <in.bgr> <w> <h> <out.bgr> [cpu|vulkan|auto]
// .bgr = raw tightly-packed BGR uint8 HWC bytes (decode with your favourite image lib).
if (args.Length < 6)
{
    Console.Error.WriteLine(
        "usage: EnhanceImage <gcnet.onnx> <drnet.onnx> <in.bgr> <w> <h> <out.bgr> [cpu|vulkan|auto]");
    return 1;
}

string gcPath = args[0], drPath = args[1], inPath = args[2], outPath = args[5];
int w = int.Parse(args[3]), h = int.Parse(args[4]);
GcdrBackend backend = args.Length > 6 ? args[6].ToLowerInvariant() switch
{
    "cpu" => GcdrBackend.Cpu,
    "vulkan" => GcdrBackend.Vulkan,
    _ => GcdrBackend.Auto,
} : GcdrBackend.Auto;

byte[] src = File.ReadAllBytes(inPath);
if (src.Length != w * h * 3) throw new InvalidDataException("input size != w*h*3");

using var enh = GcdrEnhancer.FromFiles(gcPath, drPath, backend);
byte[] dst = enh.Enhance(src, w, h);
File.WriteAllBytes(outPath, dst);
Console.WriteLine($"wrote {outPath} ({dst.Length} bytes)");
return 0;
