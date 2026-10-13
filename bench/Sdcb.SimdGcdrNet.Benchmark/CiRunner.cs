using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Sdcb.SimdGcdrNet.Benchmark;

// CI entry point over the RealDAE corpus (tools/fetch-fixture.ps1), one JSON result per run
// for tests/Sdcb.SimdGcdrNet.TestReport:
//   --ci smoke --backend cpu|metal|vulkan|auto --count N [--max-mp MP] --case-id ID --out file.json
//       correctness: sane output, run-to-run determinism, enhancement brings the images closer
//       to their ground truths; GPU backends must also match the CPU output.
//   --ci bench --backend ... --count N --case-id ID --replica R --out file.json
//       docs/corpus.md methodology on N images spread evenly over the manifest (the same N in
//       every run): the first warms up (untimed), every
//       other image is timed once, cold (per-size plans and buffers included), decode excluded.
public static partial class Program
{
    // GPU vs CPU on the same image: fp16 storage shifts a few u8 levels (Vulkan on a 3080 Ti:
    // 37.5 dB worst, 52-67 dB typical); 33 dB = 5.7 levels RMS.
    private const double MinGpuVsCpuPsnr = 33;

    private static int Ci(string corpusDir, string gcPath, string drPath, string[] args)
    {
        string mode = args.Length > 1 ? args[1] : "";
        string backendName = Arg(args, "--backend") ?? "cpu";
        int count = int.Parse(Arg(args, "--count") ?? "5");
        string caseId = Arg(args, "--case-id") ?? $"{mode}-{backendName}";
        int replica = int.Parse(Arg(args, "--replica") ?? "1");
        string outPath = Arg(args, "--out") ?? $"{caseId}.json";
        if (mode is not ("smoke" or "bench"))
        {
            Console.Error.WriteLine("usage: --ci smoke|bench [--backend cpu|metal|vulkan|auto] [--count N] [--max-mp MP] [--case-id ID] [--replica R] [--out file.json]");
            return 1;
        }
        string manifestPath = Path.Combine(corpusDir, "manifest.json");
        if (!File.Exists(manifestPath))
        {
            Console.Error.WriteLine($"RealDAE corpus not found at {corpusDir}: run tools/fetch-fixture.ps1");
            return 4;
        }
        GcdrBackend backend = ParseBackend(backendName);
        // --max-mp: only images up to that many megapixels (slow tiers such as scalar)
        double maxMp = double.Parse(Arg(args, "--max-mp") ?? "1e9", System.Globalization.CultureInfo.InvariantCulture);
        string[] all = JsonDocument.Parse(File.ReadAllText(manifestPath)).RootElement.GetProperty("images")
            .EnumerateArray()
            .Where(e => e.GetProperty("w").GetInt32() * (double)e.GetProperty("h").GetInt32() <= maxMp * 1e6)
            .Select(e => e.GetProperty("name").GetString()!).ToArray();
        if (all.Length == 0)
        {
            Console.Error.WriteLine($"no corpus image within --max-mp {maxMp}");
            return 1;
        }
        // evenly spaced over the manifest: the color / shadow / bleed groups and their sizes all show up
        count = Math.Clamp(count, 1, all.Length);
        string[] images = Enumerable.Range(0, count).Select(k => all[(int)((long)k * all.Length / count)]).ToArray();

        var doc = new JsonObject
        {
            ["kind"] = mode,
            ["caseId"] = caseId,
            ["replica"] = replica,
            ["backend"] = backendName,
            ["host"] = HostInfo(),
        };
        var rows = new JsonArray();
        doc["images"] = rows;
        var errors = new List<string>();
        var proc = Process.GetCurrentProcess();
        double peakWs = 0;
        void SampleWs() { proc.Refresh(); peakWs = Math.Max(peakWs, proc.WorkingSet64 / 1048576.0); }

        long tl = Stopwatch.GetTimestamp();
        using var enh = GcdrEnhancer.FromFiles(gcPath, drPath, backend);
        using var cpuRef = mode == "smoke" && backend != GcdrBackend.Cpu
            ? GcdrEnhancer.FromFiles(gcPath, drPath, GcdrBackend.Cpu) : null;
        doc["loadMs"] = Math.Round(Ms(tl), 1);

        byte[]? src = null, dst = null, again = null, gt = null, cpuOut = null;
        double psnrInSum = 0, psnrOutSum = 0; int psnrN = 0;
        double timedMs = 0, timedMp = 0; int timedN = 0;
        for (int i = 0; i < images.Length; i++)
        {
            string name = images[i];
            int len = DecodeBgr(Path.Combine(corpusDir, name), ref src);
            var (w, h) = ImageSize(Path.Combine(corpusDir, name));
            Span<byte> input = src.AsSpan(0, len), output = Grow(ref dst, len);
            long t = Stopwatch.GetTimestamp();
            enh.Enhance(input, w, h, w * 3, output, w * 3);
            double ms = Ms(t);
            SampleWs();
            var row = new JsonObject { ["name"] = name, ["w"] = w, ["h"] = h, ["ms"] = Math.Round(ms, 1) };
            bool warmup = mode == "bench" && i == 0;
            if (warmup) row["warmup"] = true;
            else { timedMs += ms; timedMp += w * h / 1e6; timedN++; }

            string gtPath = Path.Combine(corpusDir, name[..name.IndexOf('_')] + "_gt.webp");
            if (File.Exists(gtPath))
            {
                int glen = DecodeBgr(gtPath, ref gt);
                if (glen == len)
                {
                    double pin = Psnr(input, gt.AsSpan(0, glen)), pout = Psnr(output, gt.AsSpan(0, glen));
                    row["psnrIn"] = Math.Round(pin, 2);
                    row["psnr"] = Math.Round(pout, 2);
                    psnrInSum += pin; psnrOutSum += pout; psnrN++;
                }
            }

            if (mode == "smoke")
            {
                if (output.SequenceEqual(input)) errors.Add($"{name}: output identical to input");
                double std = StdDev(output);
                if (std <= 1.0) errors.Add($"{name}: output std {std:F2} looks degenerate");
                Span<byte> second = Grow(ref again, len);
                enh.Enhance(input, w, h, w * 3, second, w * 3);
                if (!second.SequenceEqual(output)) errors.Add($"{name}: second run differs from the first (nondeterministic)");
                if (cpuRef is not null)
                {
                    Span<byte> c = Grow(ref cpuOut, len);
                    cpuRef.Enhance(input, w, h, w * 3, c, w * 3);
                    double p = Psnr(output, c);
                    row["psnrVsCpu"] = double.IsInfinity(p) ? 99.0 : Math.Round(p, 2);
                    if (p < MinGpuVsCpuPsnr) errors.Add($"{name}: {backendName} vs cpu PSNR {p:F2} dB < {MinGpuVsCpuPsnr} dB");
                }
                SampleWs();
            }
            rows.Add(row);
            Console.WriteLine($"{name,-18} {w,5}x{h,-5} {ms,9:F1} ms{(warmup ? " (warmup)" : "")}" +
                (row["psnr"] is { } ps ? $"  psnr {ps} dB (input {row["psnrIn"]} dB)" : "") +
                (row["psnrVsCpu"] is { } pc ? $"  vs cpu {pc} dB" : ""));
        }

        if (mode == "smoke" && psnrN > 0 && psnrOutSum <= psnrInSum)
            errors.Add($"enhanced images are not closer to their ground truths: mean PSNR {psnrOutSum / psnrN:F2} dB vs input {psnrInSum / psnrN:F2} dB");

        doc["summary"] = new JsonObject
        {
            ["images"] = images.Length,
            ["timed"] = timedN,
            ["totalMs"] = Math.Round(timedMs, 1),
            ["meanMs"] = timedN > 0 ? Math.Round(timedMs / timedN, 1) : null,
            ["mpPerSec"] = timedMs > 0 ? Math.Round(timedMp / (timedMs / 1000), 3) : null,
            ["psnr"] = psnrN > 0 ? Math.Round(psnrOutSum / psnrN, 2) : null,
            ["psnrIn"] = psnrN > 0 ? Math.Round(psnrInSum / psnrN, 2) : null,
            ["peakWsMb"] = Math.Round(peakWs, 0),
        };
        doc["ok"] = errors.Count == 0;
        doc["errors"] = new JsonArray(errors.Select(e => (JsonNode)e).ToArray());
        File.WriteAllText(outPath, doc.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));

        Console.WriteLine($"{caseId}: {timedN} timed, mean {(timedN > 0 ? timedMs / timedN : 0):F1} ms" +
            (psnrN > 0 ? $", psnr {psnrOutSum / psnrN:F2} dB (input {psnrInSum / psnrN:F2} dB)" : "") + $", peak ws {peakWs:F0} MB -> {outPath}");
        foreach (string e in errors) Console.Error.WriteLine($"FAIL {e}");
        return errors.Count == 0 ? 0 : 3;
    }

    private static string? Arg(string[] args, string name)
    {
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }

    private static GcdrBackend ParseBackend(string s) => s.ToLowerInvariant() switch
    {
        "cpu" => GcdrBackend.Cpu,
        "vulkan" or "vk" => GcdrBackend.Vulkan,
        "metal" or "mtl" => GcdrBackend.Metal,
        "auto" => GcdrBackend.Auto,
        _ => throw new ArgumentException($"unknown backend {s}"),
    };

    private static double Ms(long t0) => (Stopwatch.GetTimestamp() - t0) * 1000.0 / Stopwatch.Frequency;

    private static double Psnr(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
    {
        double se = 0;
        for (int i = 0; i < a.Length; i++) { int d = a[i] - b[i]; se += (double)d * d; }
        return se == 0 ? double.PositiveInfinity : 10 * Math.Log10(255.0 * 255.0 / (se / a.Length));
    }

    private static double StdDev(ReadOnlySpan<byte> d)
    {
        long sum = 0, sq = 0;
        foreach (byte v in d) { sum += v; sq += v * v; }
        double mean = sum / (double)d.Length;
        return Math.Sqrt(sq / (double)d.Length - mean * mean);
    }

    private static JsonObject HostInfo()
    {
        // the tier the CPU kernels actually dispatch to (the library also honours DOTNET_EnableAVX512F=0)
        string isa = RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.X64 or Architecture.X86 =>
                Sdcb.SimdOnnx.Kernels.Nhwc.UseAvx512 ? "avx512" : Avx2.IsSupported ? "avx2" : Avx.IsSupported ? "avx" :
                System.Numerics.Vector.IsHardwareAccelerated ? "sse" : "scalar",
            Architecture.Arm64 => AdvSimd.IsSupported ? "neon" : "scalar",
            _ => System.Numerics.Vector.IsHardwareAccelerated ? "vector" : "scalar",
        };
        return new JsonObject
        {
            ["os"] = RuntimeInformation.OSDescription,
            ["rid"] = RuntimeInformation.RuntimeIdentifier,
            ["arch"] = RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant(),
            ["cpu"] = CpuName(),
            ["logicalCpu"] = Environment.ProcessorCount,
            ["memoryGb"] = Math.Round(GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / (double)(1L << 30), 1),
            ["isa"] = isa,
            ["runtime"] = RuntimeInformation.FrameworkDescription,
            ["env"] = string.Join(' ', Environment.GetEnvironmentVariables().Keys.Cast<string>()
                .Where(k => k.StartsWith("DOTNET_Enable", StringComparison.OrdinalIgnoreCase)).Order()
                .Select(k => $"{k}={Environment.GetEnvironmentVariable(k)}")),
        };
    }

    private static string CpuName()
    {
        try
        {
            if (OperatingSystem.IsWindows())
                return (Microsoft.Win32.Registry.GetValue(@"HKEY_LOCAL_MACHINE\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                    "ProcessorNameString", null) as string)?.Trim() ?? "unknown";
            if (OperatingSystem.IsMacOS())
            {
                using var p = Process.Start(new ProcessStartInfo("sysctl", "-n machdep.cpu.brand_string") { RedirectStandardOutput = true })!;
                string s = p.StandardOutput.ReadToEnd().Trim();
                p.WaitForExit();
                return s.Length > 0 ? s : "unknown";
            }
            if (File.Exists("/proc/cpuinfo"))
            {
                string[] lines = File.ReadAllLines("/proc/cpuinfo");
                string? model = lines.FirstOrDefault(l => l.StartsWith("model name", StringComparison.Ordinal));
                if (model is not null) return model[(model.IndexOf(':') + 1)..].Trim();
                // arm64 kernels list implementer / part ids instead of a name
                string? part = lines.FirstOrDefault(l => l.StartsWith("CPU part", StringComparison.Ordinal));
                string? impl = lines.FirstOrDefault(l => l.StartsWith("CPU implementer", StringComparison.Ordinal));
                if (part is not null) return $"arm implementer {impl?[(impl.IndexOf(':') + 1)..].Trim()} part {part[(part.IndexOf(':') + 1)..].Trim()}";
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or System.ComponentModel.Win32Exception) { }
        return "unknown";
    }
}
