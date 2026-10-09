using System.Diagnostics;
using System.Runtime.InteropServices;
using Sdcb.GcdrNet.Engine;
using Sdcb.GcdrNet.Onnx;

namespace Sdcb.GcdrNet.Benchmark;

public static class Program
{
    public static int Main(string[] args)
    {
        string dir = FindTestdata();
        string gcPath = Path.Combine(dir, "models", "gcdrnet_gcnet_sim.onnx");
        string drPath = Path.Combine(dir, "models", "gcdrnet_drnet_sim.onnx");

        if (args.Length > 0 && args[0] == "--check")
        {
            string tag = args.Length > 1 ? args[1] : "small";
            return Check(dir, gcPath, drPath, tag);
        }
        if (args.Length > 0 && args[0] == "--bench")
        {
            int h = args.Length > 1 ? int.Parse(args[1]) : 1152;
            int w = args.Length > 2 ? int.Parse(args[2]) : 1600;
            int reps = args.Length > 3 ? int.Parse(args[3]) : 10;
            return Bench(gcPath, drPath, h, w, reps);
        }
        if (args.Length > 0 && args[0] == "--benchvk")
        {
            int h = args.Length > 1 ? int.Parse(args[1]) : 1152;
            int w = args.Length > 2 ? int.Parse(args[2]) : 1600;
            int reps = args.Length > 3 ? int.Parse(args[3]) : 10;
            return BenchVk(gcPath, drPath, h, w, reps);
        }
        if (args.Length > 0 && args[0] == "--vkprof")
        {
            int h = args.Length > 1 ? int.Parse(args[1]) : 1152;
            int w = args.Length > 2 ? int.Parse(args[2]) : 1600;
            return VkProf(gcPath, drPath, h, w);
        }
        Console.WriteLine("usage: --check [tag] | --bench [h] [w] [reps] | --benchvk [h] [w] [reps] | --vkprof [h] [w]");
        return 1;
    }

    private static int Check(string dir, string gcPath, string drPath, string tag)
    {
        float[] input = File.ReadAllBytes(Path.Combine(dir, $"{tag}_input.f32")) is { } raw
            ? ToF32(raw) : throw new Exception("no input");
        int[] shape = ReadShape(dir, tag, "input");
        float[] expect = ToF32(File.ReadAllBytes(Path.Combine(dir, $"{tag}_shadow.f32")));

        var m = OnnxModel.Load(gcPath);
        var cm = CompiledModel.Compile(m, shape);
        Console.WriteLine($"gcnet: {cm.NodeCount} nodes, arena {cm.ArenaFloats * 4 / 1024.0 / 1024:F1} MB");
        var sess = cm.CreateSession();
        input.CopyTo(sess.Input);
        try { sess.Run(); }
        catch (Exception ex)
        {
            Console.WriteLine($"RUN FAILED: {ex}");
            return 2;
        }
        float max = 0; int argmax = 0;
        var outp = sess.Output;
        for (int i = 0; i < outp.Length; i++)
        {
            float d = MathF.Abs(outp[i] - expect[i]);
            if (d > max) { max = d; argmax = i; }
        }
        Console.WriteLine($"gcnet[{tag}] max diff = {max:E3} at {argmax} (got {outp[argmax]}, want {expect[argmax]})");

        // drnet
        float[] drIn = ToF32(File.ReadAllBytes(Path.Combine(dir, $"{tag}_dr_in.f32")));
        float[] drExpect = ToF32(File.ReadAllBytes(Path.Combine(dir, $"{tag}_output.f32")));
        var m2 = OnnxModel.Load(drPath);
        var cm2 = CompiledModel.Compile(m2, ReadShape(dir, tag, "dr_in"));
        var sess2 = cm2.CreateSession();
        drIn.CopyTo(sess2.Input);
        sess2.Run();
        var o2 = sess2.Output;
        float max2 = 0;
        for (int i = 0; i < o2.Length; i++) max2 = MathF.Max(max2, MathF.Abs(o2[i] - drExpect[i]));
        Console.WriteLine($"drnet[{tag}] max diff = {max2:E3}");
        return max < 2e-3 && max2 < 2e-3 ? 0 : 3;
    }

    private static int Bench(string gcPath, string drPath, int h, int w, int reps)
    {
        var gc = CompiledModel.Compile(OnnxModel.Load(gcPath), [1, 3, h, w]);
        var dr = CompiledModel.Compile(OnnxModel.Load(drPath), [1, 6, h, w]);
        var s1 = gc.CreateSession(); var s2 = dr.CreateSession();
        new Random(0).NextBytes(MemoryMarshal.AsBytes(s1.Input));
        new Random(1).NextBytes(MemoryMarshal.AsBytes(s2.Input));
        // warmup
        s1.Run(); s2.Run();
        double best1 = 1e9, best2 = 1e9;
        for (int i = 0; i < reps; i++)
        {
            long t = Stopwatch.GetTimestamp(); s1.Run();
            best1 = Math.Min(best1, Stopwatch.GetTimestamp() - t);
            t = Stopwatch.GetTimestamp(); s2.Run();
            best2 = Math.Min(best2, Stopwatch.GetTimestamp() - t);
        }
        Console.WriteLine($"gc {best1 / Stopwatch.Frequency * 1000:F1} ms, dr {best2 / Stopwatch.Frequency * 1000:F1} ms, total {(best1 + best2) / Stopwatch.Frequency * 1000:F1} ms");
        if (Environment.GetEnvironmentVariable("GCDRNET_PROF") == "1")
        {
            Console.WriteLine("--- gc profile ---");
            foreach (var kv in s1.Profile().OrderByDescending(k => k.Value)) Console.WriteLine($"  {kv.Key,-16} {kv.Value.TotalMilliseconds,8:F1} ms");
            Console.WriteLine("--- dr profile ---");
            foreach (var kv in s2.Profile().OrderByDescending(k => k.Value)) Console.WriteLine($"  {kv.Key,-16} {kv.Value.TotalMilliseconds,8:F1} ms");
        }
        return 0;
    }

    private static int BenchVk(string gcPath, string drPath, int h, int w, int reps)
    {
        // full pipeline: BGR u8 -> enhance -> BGR u8, everything on GPU
        using var enh = GcdrEnhancer.FromFiles(gcPath, drPath, GcdrBackend.Vulkan);
        byte[] src = new byte[w * h * 3];
        new Random(0).NextBytes(src);
        byte[] dst = new byte[w * h * 3];
        enh.Enhance(src, w, h, w * 3, dst, w * 3); // warmup + JIT + upload
        double best = 1e9;
        for (int i = 0; i < reps; i++)
        {
            long t = Stopwatch.GetTimestamp();
            enh.Enhance(src, w, h, w * 3, dst, w * 3);
            best = Math.Min(best, Stopwatch.GetTimestamp() - t);
        }
        Console.WriteLine($"vulkan e2e {w}x{h}: {best / Stopwatch.Frequency * 1000:F1} ms (incl. upload/download)");
        return 0;
    }

    private static int VkProf(string gcPath, string drPath, int h, int w)
    {
        var be = Backends.Vulkan.VkBackend.Default;
        foreach ((string tag, string mp, int cin) in new[] { ("gc", gcPath, 3), ("dr", drPath, 6) })
        {
            var m = OnnxModel.Load(mp);
            using var s = be.NewSession(m, [1, cin, h, w]);
            s.Run(); // warm
            var agg = s.DebugProfile();
            double tot = agg.Values.Sum();
            Console.WriteLine($"--- {tag} gpu per-shader (total {tot:F1} ms) ---");
            foreach (var kv in agg.OrderByDescending(k => k.Value)) Console.WriteLine($"  {kv.Key,-14} {kv.Value,8:F2} ms");
            if (Environment.GetEnvironmentVariable("VKPROF_NODES") == "1")
                foreach (var (name, ms) in s.PerDispatch!.OrderByDescending(x => x.Ms).Take(15))
                    Console.WriteLine($"    {ms,8:F2} ms  {name}");
        }
        return 0;
    }

    private static float[] ToF32(byte[] raw)
    {
        var f = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, f, 0, raw.Length);
        return f;
    }

    private static int[] ReadShape(string dir, string tag, string name)
    {
        var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "manifest.json")));
        return doc.RootElement.GetProperty(tag).GetProperty("shapes").GetProperty(name)
            .EnumerateArray().Select(e => e.GetInt32()).ToArray();
    }

    private static string FindTestdata()
    {
        var d = new DirectoryInfo(AppContext.BaseDirectory);
        while (d != null)
        {
            string cand = Path.Combine(d.FullName, "tests", "testdata");
            if (File.Exists(Path.Combine(cand, "manifest.json"))) return cand;
            cand = Path.Combine(d.FullName, "testdata");
            if (File.Exists(Path.Combine(cand, "manifest.json"))) return cand;
            d = d.Parent;
        }
        throw new DirectoryNotFoundException("testdata not found");
    }
}
