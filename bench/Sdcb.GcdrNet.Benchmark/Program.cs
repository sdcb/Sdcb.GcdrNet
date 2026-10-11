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
        if (args.Length > 0 && args[0] == "--benchcpu")
        {
            int h = args.Length > 1 ? int.Parse(args[1]) : 1152;
            int w = args.Length > 2 ? int.Parse(args[2]) : 1600;
            int reps = args.Length > 3 ? int.Parse(args[3]) : 10;
            return BenchE2e(gcPath, drPath, h, w, reps, GcdrBackend.Cpu, "cpu");
        }
        if (args.Length > 0 && args[0] == "--trace")
        {
            int h = args.Length > 1 ? int.Parse(args[1]) : 96;
            int w = args.Length > 2 ? int.Parse(args[2]) : 128;
            return Trace(gcPath, drPath, h, w);
        }
        if (args.Length > 0 && args[0] == "--vkprof")
        {
            int h = args.Length > 1 ? int.Parse(args[1]) : 1152;
            int w = args.Length > 2 ? int.Parse(args[2]) : 1600;
            return VkProf(gcPath, drPath, h, w);
        }
        if (args.Length > 0 && args[0] == "--corpus")
        {
            string backend = args.Length > 1 ? args[1] : "auto";
            return Corpus(gcPath, drPath, Path.Combine(dir, "realdae"), backend);
        }
        if (args.Length > 0 && args[0] == "--flip")
        {
            // nondeterminism hunt: enhance one image repeatedly while interleaving
            // another size to force pool eviction + pipeline recreation.
            return Flip(gcPath, drPath, Path.Combine(dir, "realdae"),
                args.Length > 1 ? args[1] : "18_shadow.webp");
        }
        if (args.Length > 0 && args[0] == "--analyze")
        {
            int ah = args.Length > 1 ? int.Parse(args[1]) : 2592;
            int aw = args.Length > 2 ? int.Parse(args[2]) : 2464;
            Analyze(gcPath, drPath, ah, aw);
            return 0;
        }
        if (args.Length > 0 && args[0] == "--vkdup")
        {
            int dh = args.Length > 1 ? int.Parse(args[1]) : 2560;
            int dw = args.Length > 2 ? int.Parse(args[2]) : 2464;
            int dr_ = args.Length > 3 ? int.Parse(args[3]) : 0;
            return VkDup(gcPath, drPath, dh, dw, dr_ != 0);
        }
        if (args.Length > 0 && args[0] == "--planchk")
        {
            // do two Compile() calls produce byte-identical buf layouts?
            for (int rep = 0; rep < 3; rep++)
            {
                var m = OnnxModel.Load(gcPath);
                int[] sh = [1, 3, 2592, 2464];
                var p1 = GraphCompiler.Compile(m, sh);
                var p2 = GraphCompiler.Compile(m, sh);
                int bad = 0;
                for (int i = 0; i < p1.BufOffset.Length; i++)
                    if (p1.BufOffset[i] != p2.BufOffset[i]) bad++;
                int nb = 0;
                for (int i = 0; i < p1.Nodes.Length; i++)
                    if (!p1.Nodes[i].In.AsSpan().SequenceEqual(p2.Nodes[i].In) ||
                        !p1.Nodes[i].Out.AsSpan().SequenceEqual(p2.Nodes[i].Out)) nb++;
                Console.WriteLine($"rep{rep}: bufs={p1.BufOffset.Length} diffOff={bad} arena={p1.ArenaFloats}/{p2.ArenaFloats} nodeDiff={nb} " +
                    $"inOff={p1.BufOffset[p1.Values[p1.InputValue].Buf]}/{p2.BufOffset[p2.Values[p2.InputValue].Buf]} " +
                    $"outOff={p1.BufOffset[p1.Values[p1.OutputValue].Buf]}/{p2.BufOffset[p2.Values[p2.OutputValue].Buf]}");
            }
            return 0;
        }
        if (args.Length > 0 && args[0] == "--probe")
        {
            string pimg = args.Length > 1 ? args[1] : "18_shadow.webp";
            return Probe(gcPath, drPath, Path.Combine(dir, "realdae", pimg), args.Skip(2).Select(int.Parse).ToArray());
        }
        if (args.Length > 0 && args[0] == "--bisect")
        {
            string pimg = args.Length > 1 ? args[1] : "18_shadow.webp";
            return Bisect(gcPath, drPath, Path.Combine(dir, "realdae", pimg), args.Skip(2).Select(int.Parse).ToArray());
        }
        Console.WriteLine("usage: --check [tag] | --bench [h] [w] [reps] | --benchvk [h] [w] [reps] | --vkprof [h] [w] | --corpus [cpu|vulkan|auto] | --flip [img] | --analyze [h] [w] | --vkdup [h] [w] [dr] | --probe [img] [px...]");
        return 1;
    }

    // determinism probe on a standalone VkInferenceSession: same input, run twice,
    // destroy, recreate — any hash change = stale-read/nondeterminism at session level.
    private static int VkDup(string gcPath, string drPath, int h, int w, bool useDr)
    {
        var be = Backends.Vulkan.VkBackend.Default;
        string path = useDr ? drPath : gcPath;
        int c = useDr ? 6 : 3;
        var m = OnnxModel.Load(path);
        static ulong Fhash(ReadOnlySpan<float> d)
        {
            ulong hsh = 1469598103934665603UL;
            foreach (float f in d) { hsh ^= BitConverter.SingleToUInt32Bits(f); hsh *= 1099511628211UL; }
            return hsh;
        }
        for (int round = 0; round < 4; round++)
        {
            using var s = be.NewSession(m, [1, c, h, w]);
            var rng = new Random(42);
            for (int i = 0; i < s.Input.Length; i++) s.Input[i] = rng.NextSingle();
            s.Run();
            Console.WriteLine($"session#{round} run1 hash={Fhash(s.Output):x16}");
            s.Run();
            Console.WriteLine($"session#{round} run2 hash={Fhash(s.Output):x16}");
            // interleave: run smaller sessions to churn memory
            foreach (int cs in new[] { 1024, 1312, 1152 })
            {
                using var s2 = be.NewSession(m, [1, c, cs, cs]);
                new Random(0).NextBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(s2.Input));
                s2.Run();
            }
        }
        return 0;
    }

    // read-before-write audit: replay the plan's arena writes in node order and flag
    // any input value whose byte range isn't fully covered by prior writes.
    private static void Analyze(string gcPath, string drPath, int h, int w)
    {
        foreach (var (path, c) in new[] { (gcPath, 3), (drPath, 6) })
        {
            var plan = GraphCompiler.Compile(OnnxModel.Load(path), [1, c, h, w]);
            var written = new List<(long s, long e)>();
            bool Covered(long s, long e) =>
                written.Any(r => r.s <= s && r.e >= e);
            // graph input counts as written by the host copy
            var iv = plan.Values[plan.InputValue];
            if (iv.Buf >= 0)
                written.Add((plan.BufOffset[iv.Buf] * 4L, plan.BufOffset[iv.Buf] * 4L + iv.Shape.Aggregate(1L, (a, x) => a * x) * 4));
            int flags = 0;
            for (int i = 0; i < plan.Nodes.Length; i++)
            {
                var nd = plan.Nodes[i];
                foreach (int vid in nd.In)
                {
                    var v = plan.Values[vid];
                    if (v.Buf < 0 || v.Shape.Length == 0) continue;
                    long s = plan.BufOffset[v.Buf] * 4L, e = s + v.Shape.Aggregate(1L, (a, x) => a * x) * 4;
                    if (!Covered(s, e))
                    {
                        flags++;
                        Console.WriteLine($"  read-unwritten node[{i}] {nd.Op} {nd.Name} vid={vid} buf={v.Buf} [{s},{e}) shape={string.Join('x', v.Shape)}");
                    }
                }
                foreach (int vid in nd.Out)
                {
                    var v = plan.Values[vid];
                    if (v.Buf < 0 || v.Shape.Length == 0) continue;
                    long s = plan.BufOffset[v.Buf] * 4L, e = s + v.Shape.Aggregate(1L, (a, x) => a * x) * 4;
                    if (!Covered(s, e)) written.Add((s, e));
                }
            }
            Console.WriteLine($"{Path.GetFileName(path)} [{c}x{h}x{w}]: {flags} unwritten-read flags, arena {plan.ArenaFloats * 4L / 1048576.0:F0} MB");
        }
    }

    // bisect nondeterminism: two same-size pipelines with churn pipelines created
    // between them; hash gc output + dr input + final dst per run to see which
    // stage first diverges.
    private static int Probe(string gcPath, string drPath, string imgPath, int[] churnPx)
    {
        var be = Backends.Vulkan.VkBackend.Default;
        var gc = OnnxModel.Load(gcPath); var dr = OnnxModel.Load(drPath);
        int iw, ih; byte[] bgr;
        using (var bmp = SkiaSharp.SKBitmap.Decode(imgPath)) { iw = bmp.Width; ih = bmp.Height; }
        bgr = DecodeBgr(imgPath);
        byte[] dst = new byte[bgr.Length];
        int pw = (iw + 31) / 32 * 32, ph = (ih + 31) / 32 * 32;
        var gcI = new float[3 * pw * ph]; var gcO = new float[3 * pw * ph]; var drI = new float[6 * pw * ph];

        static ulong Fb(byte[] d)
        {
            ulong h = 1469598103934665603UL;
            for (int i = 0; i < d.Length; i += 977) { h ^= d[i]; h *= 1099511628211UL; }
            return h;
        }
        static ulong Ff(float[] d)
        {
            ulong h = 1469598103934665603UL;
            for (int i = 0; i < d.Length; i += 977) { h ^= BitConverter.SingleToUInt32Bits(d[i]); h *= 1099511628211UL; }
            return h;
        }
        void Run(Backends.Vulkan.VkGcdrPipeline p, string tag)
        {
            p.Enhance(bgr, iw * 3, dst, iw * 3);
            p.DebugIntermediates(gcI, gcO, drI);
            Console.WriteLine($"{tag}: dst={Fb(dst):x16} gcIn={Ff(gcI):x16} gcOut={Ff(gcO):x16} drIn={Ff(drI):x16} live={Backends.Vulkan.VkDevice.LiveDeviceBytes >> 20}MB");
        }

        var p1 = be.NewPipe(gc, dr, ih, iw);
        Run(p1, "p1-run1");
        Run(p1, "p1-run2");
        var churn = new List<Backends.Vulkan.VkGcdrPipeline>();
        int k = 0;
        foreach (int px in churnPx)
        {
            int cw = Math.Max(64, (int)Math.Sqrt(px * 16.0 / 9) / 32 * 32);
            int chh = Math.Max(64, px / Math.Max(cw, 1) / 32 * 32);
            var cp = be.NewPipe(gc, dr, chh, cw);
            byte[] cb = new byte[chh * cw * 3]; new Random(px).NextBytes(cb);
            byte[] cd = new byte[chh * cw * 3];
            cp.Enhance(cb, cw * 3, cd, cw * 3);
            churn.Add(cp);
            if (churn.Count > 1) { churn[0].Dispose(); churn.RemoveAt(0); }
        }
        foreach (var c in churn) c.Dispose();
        var p2 = be.NewPipe(gc, dr, ih, iw);
        Run(p2, "p2-after-churn");
        p2.Dispose(); p1.Dispose();
        return 0;
    }

    // bisect which plan value first diverges between two pipeline instances.
    private static int Bisect(string gcPath, string drPath, string imgPath, int[] churnPx)
    {
        var be = Backends.Vulkan.VkBackend.Default;
        var gc = OnnxModel.Load(gcPath); var dr = OnnxModel.Load(drPath);
        int iw, ih;
        using (var bmp = SkiaSharp.SKBitmap.Decode(imgPath)) { iw = bmp.Width; ih = bmp.Height; }
        byte[] bgr = DecodeBgr(imgPath);
        byte[] dst = new byte[bgr.Length];

        var p1 = be.NewPipe(gc, dr, ih, iw);
        p1.Enhance(bgr, iw * 3, dst, iw * 3);
        // replicate corpus allocation history: run every corpus image sorted by px
        // (up to `churnPx[0]` predecessors, or all when empty) with real (h,w) sizes
        var manifest = System.Text.Json.JsonDocument.Parse(
            File.ReadAllText(Path.Combine(Path.GetDirectoryName(imgPath)!, "manifest.json")));
        var seq = manifest.RootElement.GetProperty("images").EnumerateArray()
            .Select(e => (name: e.GetProperty("name").GetString()!, w: e.GetProperty("w").GetInt32(), h: e.GetProperty("h").GetInt32()))
            .Where(e => !(e.w == iw && e.h == ih))
            .OrderBy(e => (long)e.w * e.h).ToArray();
        int take = churnPx.Length > 0 ? churnPx[0] : seq.Length;
        var alive = new List<Backends.Vulkan.VkGcdrPipeline>();
        foreach (var e in seq.Take(take))
        {
            var cp = be.NewPipe(gc, dr, e.h, e.w);
            byte[] cb = new byte[e.h * e.w * 3]; new Random(e.w * e.h).NextBytes(cb);
            byte[] cd = new byte[e.h * e.w * 3];
            cp.Enhance(cb, e.w * 3, cd, e.w * 3);
            alive.Add(cp);
            if (alive.Count > 1) { alive[0].Dispose(); alive.RemoveAt(0); }
        }
        foreach (var c in alive) c.Dispose();
        var p2 = be.NewPipe(gc, dr, ih, iw);
        p2.Enhance(bgr, iw * 3, dst, iw * 3);

        // dump BOTH whole gc arenas in one shot each, diff per plan value on host
        var plan = p1.DebugGc.DebugPlan;
        Console.WriteLine($"gc plan: nodes={plan.Nodes.Length} values={plan.Values.Length} bufs={plan.BufOffset.Length} arena={plan.ArenaFloats}");
        var a1 = new float[plan.ArenaFloats]; var a2 = new float[plan.ArenaFloats];
        p1.DebugGc.DebugArena(a1);
        p2.DebugGc.DebugArena(a2);
        var map = new Dictionary<int, List<int>>(); // value -> producing node idx
        for (int i = 0; i < plan.Nodes.Length; i++)
            foreach (int o in plan.Nodes[i].Out) { if (!map.TryGetValue(o, out var l)) map[o] = l = []; l.Add(i); }
        for (int vid = 0; vid < plan.Values.Length; vid++)
        {
            var v = plan.Values[vid];
            if (v.Buf < 0 || v.Buf >= plan.BufOffset.Length || v.Shape.Length == 0) continue;
            long s = plan.BufOffset[v.Buf], n = v.ElementCount;
            if (s < 0) continue;
            if (s + n > a1.Length) { Console.WriteLine($"  vid={vid} OVERFLOW buf={v.Buf} off={s} elems={n} arena={a1.Length}"); continue; }
            int diff = 0; double maxd = 0;
            for (long i = s; i < s + n; i += 97)
                if (a1[i] != a2[i]) { diff++; maxd = Math.Max(maxd, Math.Abs((double)a1[i] - a2[i])); }
            if (diff > 0)
            {
                string prod = map.TryGetValue(vid, out var l)
                    ? string.Join(",", l.Select(i => $"[{i}]{plan.Nodes[i].Op}:{plan.Nodes[i].Name}"))
                    : "input/none";
                Console.WriteLine($"  vid={vid} buf={v.Buf} off={s} elems={n} shape=[{string.Join(',', v.Shape)}] ndiff={diff} maxd={maxd:G3}  prod={prod}");
            }
        }
        // consts check: are the two graphs' weight buffers byte-identical?
        var (c1, _) = p1.DebugGc.DebugBuffers;
        var (c2, _) = p2.DebugGc.DebugBuffers;
        unsafe
        {
            byte[] cb1 = new byte[c1.Size], cb2 = new byte[c2.Size];
            fixed (byte* p = cb1) be.Dev.Download(c1, p, c1.Size);
            fixed (byte* p = cb2) be.Dev.Download(c2, p, c2.Size);
            long bdiff = 0, first = -1;
            for (long i = 0; i < Math.Min(cb1.LongLength, cb2.LongLength); i++)
                if (cb1[i] != cb2[i]) { bdiff++; if (first < 0) first = i; }
            Console.WriteLine($"consts: handle={c1.Buffer == c2.Buffer} size={c1.Size}/{c2.Size} diffBytes={bdiff} first@{first}");
        }
        p1.Dispose(); p2.Dispose();
        return 0;
    }

    private static int Flip(string gcPath, string drPath, string corpusDir, string imgName)
    {
        var manifest = System.Text.Json.JsonDocument.Parse(
            File.ReadAllText(Path.Combine(corpusDir, "manifest.json")));
        var images = manifest.RootElement.GetProperty("images").EnumerateArray()
            .Select(e => (name: e.GetProperty("name").GetString()!,
                          w: e.GetProperty("w").GetInt32(),
                          h: e.GetProperty("h").GetInt32())).ToArray();
        var a = images.First(i => i.name == imgName);
        var others = images.Where(i => i.name != imgName && (long)i.w * i.h > 3_000_000)
            .OrderBy(i => (long)i.w * i.h).ToArray();
        byte[] bgrA = DecodeBgr(Path.Combine(corpusDir, a.name));
        var bgrO = others.Select(o => (o, DecodeBgr(Path.Combine(corpusDir, o.name)))).ToArray();

        static ulong Fnv(byte[] d)
        {
            ulong hsh = 1469598103934665603UL;
            for (int i = 0; i < d.Length; i += 977) { hsh ^= d[i]; hsh *= 1099511628211UL; }
            return hsh;
        }

        using var enh = GcdrEnhancer.FromFiles(gcPath, drPath, GcdrBackend.Vulkan);
        byte[] dst = new byte[bgrA.Length], tmp = new byte[bgrA.Length];
        // CPU reference (slow, one shot) — the oracle-correct output
        using var cpu = GcdrEnhancer.FromFiles(gcPath, drPath, GcdrBackend.Cpu);
        byte[] dstCpu = new byte[bgrA.Length];
        cpu.Enhance(bgrA, a.w, a.h, a.w * 3, dstCpu, a.w * 3);
        Console.WriteLine($"cpu-ref       hash={Fnv(dstCpu):x16}");
        // pass 1: same pipeline twice (pool hit) — should be identical
        for (int k = 0; k < 2; k++)
        {
            enh.Enhance(bgrA, a.w, a.h, a.w * 3, dst, a.w * 3);
            Console.WriteLine($"same-pipe[{k}] hash={Fnv(dst):x16} firstDiff={FirstDiff(dst, dstCpu)}");
        }
        // pass 2: evict between runs — fresh pipeline each time, varying predecessor sizes
        for (int k = 0; k < Math.Min(bgrO.Length, 20); k++)
        {
            var (o, bgr) = bgrO[k % bgrO.Length];
            byte[] t2 = tmp.Length == bgr.Length ? tmp : new byte[bgr.Length];
            enh.Enhance(bgr, o.w, o.h, o.w * 3, t2, o.w * 3); // evicts a's pool (size differs)
            enh.Enhance(bgrA, a.w, a.h, a.w * 3, dst, a.w * 3);
            Console.WriteLine($"fresh-pipe[{k}] after {o.name[..Math.Min(20, o.name.Length)]} hash={Fnv(dst):x16} firstDiff={FirstDiff(dst, dstCpu)}");
        }
        return 0;

        static int FirstDiff(byte[] x, byte[] y)
        {
            for (int i = 0; i < x.Length; i++) if (x[i] != y[i]) return i;
            return -1;
        }
    }

    // Iterate the RealDAE corpus (manifest.json + *.webp; *_gt.webp for PSNR when present).
    // First image warms up global init (model load/JIT); every remaining image is timed once —
    // cold e2e including per-size setup (plans, graphs, buffer growth), excluding decode.
    private static int Corpus(string gcPath, string drPath, string corpusDir, string backend)
    {
        var be = backend.ToLowerInvariant() switch
        {
            "cpu" => GcdrBackend.Cpu,
            "vulkan" or "vk" => GcdrBackend.Vulkan,
            _ => GcdrBackend.Auto,
        };
        var manifest = System.Text.Json.JsonDocument.Parse(
            File.ReadAllText(Path.Combine(corpusDir, "manifest.json")));
        // manifest order (01..50): sizes interleave like real workloads — pool
        // eviction/churn is exercised instead of a sorted best case.
        var images = manifest.RootElement.GetProperty("images").EnumerateArray()
            .Select(e => (name: e.GetProperty("name").GetString()!,
                          w: e.GetProperty("w").GetInt32(),
                          h: e.GetProperty("h").GetInt32()))
            .ToArray();

        // Working set (MB, Refresh()ed): sampled before/after engine init and after every
        // Enhance (warmup + pass1 + pass2). ΔWS = last pass1 image − loaded; peak = max sample.
        var proc = System.Diagnostics.Process.GetCurrentProcess();
        double WsMb() { proc.Refresh(); return proc.WorkingSet64 / 1048576.0; }
        double wsInit0 = WsMb();
        using var enh = GcdrEnhancer.FromFiles(gcPath, drPath, be);
        double wsLoaded = WsMb(), wsPeak = wsLoaded;
        Console.WriteLine($"engine loaded: ws {wsInit0:F1} -> {wsLoaded:F1} MB ({wsLoaded - wsInit0:+0.0;-0.0} MB)");
        {   // global warmup: model load, JIT, first pipeline — untimed
            var (name, w, h) = images[0];
            byte[] bgr = DecodeBgr(Path.Combine(corpusDir, name));
            enh.Enhance(bgr, w, h, w * 3, new byte[bgr.Length], w * 3);
            double ws = WsMb(); wsPeak = Math.Max(wsPeak, ws);
            Console.WriteLine($"{name + " (warmup)",-52} {w,5}x{h,-5} ws {ws,7:F1} MB");
        }

        // GCDR_INTER=1: per-image intermediate hashes (gcIn/gcOut/drIn) to bisect
        // which stage diverges between pipeline instances.
        var inter = new Dictionary<string, (ulong gi, ulong go, ulong di)>();
        string curName = ""; int curW = 0, curH = 0; byte[]? curBgr = null;
        if (Environment.GetEnvironmentVariable("GCDR_INTER") == "1" && be != GcdrBackend.Cpu)
        {
            var ibuf = new float[12 * 2700 * 2700];
            var savedArenas = new Dictionary<string, float[]>();
            string? dumpDir = Environment.GetEnvironmentVariable("GCDR_DUMPPIPE");
            static ulong Ff(Span<float> d)
            {
                ulong hh = 1469598103934665603UL;
                for (int i = 0; i < d.Length; i += 977) { hh ^= BitConverter.SingleToUInt32Bits(d[i]); hh *= 1099511628211UL; }
                return hh;
            }
            enh.DebugAfterEnhance = p =>
            {
                int pw = (curW + 31) / 32 * 32, ph = (curH + 31) / 32 * 32, hw = pw * ph;
                var gi = ibuf.AsSpan(0, 3 * hw);
                var go = ibuf.AsSpan(3 * hw, 3 * hw);
                var di = ibuf.AsSpan(6 * hw, 6 * hw);
                p.DebugIntermediates(gi, go, di);
                int nz = 0; foreach (float f in go) if (f != 0) nz++;
                var cur = (gi: Ff(gi), go: Ff(go), di: Ff(di));
                bool hasPrev = inter.TryGetValue(curName, out var prev);
                if (dumpDir != null)
                {
                    if (!savedArenas.ContainsKey(curName) && savedArenas.Count < 12 && hw < 2_000_000)
                    {
                        var saved = new float[p.DebugGc.DebugPlan.ArenaFloats];
                        p.DebugGc.DebugArena(saved);
                        savedArenas[curName] = saved;
                    }
                    else if (hasPrev && prev != cur && savedArenas.TryGetValue(curName, out var saved))
                    {
                        // rerun on a fresh pipe and diff arena values
                        var a2 = new float[p.DebugGc.DebugPlan.ArenaFloats];
                        p.DebugGc.DebugArena(a2);
                        var plan = p.DebugGc.DebugPlan;
                        var map = new Dictionary<int, List<int>>();
                        for (int i = 0; i < plan.Nodes.Length; i++)
                            foreach (int o in plan.Nodes[i].Out) { if (!map.TryGetValue(o, out var l)) map[o] = l = []; l.Add(i); }
                        for (int vid = 0; vid < plan.Values.Length; vid++)
                        {
                            var v = plan.Values[vid];
                            if (v.Buf < 0 || v.Buf >= plan.BufOffset.Length || v.Shape.Length == 0) continue;
                            long s = plan.BufOffset[v.Buf], n = v.ElementCount;
                            if (s < 0 || s + n > a2.LongLength) continue;
                            int nd = 0;
                            for (long i = s; i < s + n; i += 97) if (saved[i] != a2[i]) nd++;
                            if (nd > 0)
                            {
                                string prod = map.TryGetValue(vid, out var l)
                                    ? string.Join(",", l.Select(i => $"[{i}]{plan.Nodes[i].Op}:{plan.Nodes[i].Name}")) : "in";
                                Console.WriteLine($"    vid={vid} buf={v.Buf} off={s} shape=[{string.Join(',', v.Shape)}] nd={nd} prod={prod}");
                            }
                        }
                    }
                }
                if (!hasPrev) inter[curName] = cur;
                else if (prev != cur)
                    Console.WriteLine($"  INTER DIFF {curName}: gcIn {(prev.gi == cur.gi ? "same" : $"{prev.gi:x8}->{cur.gi:x8}")} " +
                        $"gcOut {(prev.go == cur.go ? "same" : $"{prev.go:x8}->{cur.go:x8}")} " +
                        $"drIn {(prev.di == cur.di ? "same" : $"{prev.di:x8}->{cur.di:x8}")} nz={nz}/{go.Length} first={go[0]:G4},{go[1]:G4},{go[2]:G4},{go[3]:G4}");
            };
        }

        Backends.Vulkan.VkDevice.ResetPeakDeviceBytes();
        long gc0 = GC.GetTotalMemory(true);
        double wsLast = wsLoaded;
        var hashes = new Dictionary<string, ulong>();
        double totalMs = 0, totalMp = 0, psnrSum = 0; int psnrN = 0;
        int from = int.TryParse(Environment.GetEnvironmentVariable("GCDR_FROM"), out int f0) ? f0 : 1;
        string? only = Environment.GetEnvironmentVariable("GCDR_ONLY");
        var run = images.Skip(from).Where(i => only == null || i.name.Contains(only)).ToArray();
        // grow-only image buffers: per-image LOH garbage would otherwise pile up in the
        // working set (~45 MB per 5 MP image until a gen2 GC) and drown the engine's share
        byte[]? srcBuf = null, dstBuf = null, gtBuf = null;
        foreach ((string name, int w, int h) in run)
        {
            int len = DecodeBgr(Path.Combine(corpusDir, name), ref srcBuf);
            Span<byte> bgr = srcBuf.AsSpan(0, len), dst = Grow(ref dstBuf, len);
            curName = name; curW = w; curH = h;
            long t = Stopwatch.GetTimestamp();
            enh.Enhance(bgr, w, h, w * 3, dst, w * 3);
            double ms = (Stopwatch.GetTimestamp() - t) * 1000.0 / Stopwatch.Frequency;
            double mp = w * h / 1e6;
            totalMs += ms; totalMp += mp;
            wsLast = WsMb(); wsPeak = Math.Max(wsPeak, wsLast);
            ulong hsh = 1469598103934665603UL;
            for (int i = 0; i < dst.Length; i += 977) { hsh ^= dst[i]; hsh *= 1099511628211UL; }
            hashes[name] = hsh;
            string? dumpDir = Environment.GetEnvironmentVariable("GCDR_DUMP");
            if (dumpDir != null)
                File.WriteAllBytes(Path.Combine(dumpDir, name.Replace(".webp", ".bgr")), dst.ToArray());
            string quality = $"  {hsh:x8}";
            string gt = Path.Combine(corpusDir, name[..name.IndexOf('_')] + "_gt.webp");
            if (File.Exists(gt))
            {
                int glen = DecodeBgr(gt, ref gtBuf);
                ReadOnlySpan<byte> refBgr = gtBuf.AsSpan(0, glen);
                if (refBgr.Length == dst.Length)
                {
                    double se = 0, ae = 0;
                    for (int i = 0; i < dst.Length; i++)
                    { int d = dst[i] - refBgr[i]; se += (double)d * d; ae += Math.Abs(d); }
                    double psnr = 10 * Math.Log10(255.0 * 255.0 / (se / dst.Length));
                    psnrSum += psnr; psnrN++;
                    quality += $" {psnr,6:F2} dB  mae {ae / dst.Length,5:F2}";
                }
            }
            Console.WriteLine($"{name,-52} {w,5}x{h,-5} {mp,5:F2} MP {ms,9:F1} ms{quality}  ws {wsLast,7:F1} MB");
        }
        long gcD = GC.GetTotalMemory(true) - gc0;

        // second pass: measures pool-hit throughput AND cross-checks output hashes
        double warmMs = 0; int flaky = 0;
        foreach ((string name, int w, int h) in run)
        {
            int len = DecodeBgr(Path.Combine(corpusDir, name), ref srcBuf);
            Span<byte> bgr = srcBuf.AsSpan(0, len), dst = Grow(ref dstBuf, len);
            curName = name; curW = w; curH = h;
            long t = Stopwatch.GetTimestamp();
            enh.Enhance(bgr, w, h, w * 3, dst, w * 3);
            warmMs += (Stopwatch.GetTimestamp() - t) * 1000.0 / Stopwatch.Frequency;
            wsPeak = Math.Max(wsPeak, WsMb());
            ulong h2 = 1469598103934665603UL;
            for (int i = 0; i < dst.Length; i += 977) { h2 ^= dst[i]; h2 *= 1099511628211UL; }
            if (h2 != hashes[name]) { flaky++; Console.WriteLine($"  HASH MISMATCH pass1={hashes[name]:x8} pass2={h2:x8}  {name}"); }
        }
        if (flaky == 0) Console.WriteLine("  pass1/pass2 output hashes all identical");

        Console.WriteLine($"corpus[{backend}] {run.Length} timed images: {totalMs:F1} ms total, " +
            $"{totalMp / (totalMs / 1000):F2} MP/s, mean {totalMs / run.Length:F1} ms");
        Console.WriteLine($"  2nd pass: {warmMs:F1} ms total, {totalMp / (warmMs / 1000):F2} MP/s " +
            $"(delta = per-size setup {totalMs - warmMs:F0} ms, {100.0 * (totalMs - warmMs) / totalMs:F0}%)");
        if (psnrN > 0) Console.WriteLine($"  psnr mean {psnrSum / psnrN:F2} dB over {psnrN} images");
        var (pools, workers, evicted) = enh.PoolStats;
        Console.WriteLine($"  pools: {pools} pools created ({workers} workers), {evicted} evicted");
        Console.WriteLine($"  ws: init {wsInit0:F0} -> loaded {wsLoaded:F0} MB ({wsLoaded - wsInit0:+0;-0}), " +
            $"loaded -> last {wsLast:F0} MB (delta {wsLast - wsLoaded:+0;-0} MB), peak {wsPeak:F0} MB");
        Console.WriteLine($"  mem: managed {gcD / 1048576.0:+0;-0} MB (after forced GC), " +
            $"vkDevice live {Backends.Vulkan.VkDevice.LiveDeviceBytes / 1048576.0:F0} MB " +
            $"(peak {Backends.Vulkan.VkDevice.PeakDeviceBytes / 1048576.0:F0} MB)");
        return 0;
    }

    private static byte[] DecodeBgr(string path)
    {
        byte[]? buf = null;
        int len = DecodeBgr(path, ref buf);
        return buf!.Length == len ? buf : buf[..len];
    }

    private static Span<byte> Grow(ref byte[]? buf, int len)
    {
        if (buf is null || buf.Length < len) buf = new byte[len];
        return buf.AsSpan(0, len);
    }

    /// <summary>Decodes into a grow-only buffer; returns the BGR byte count.</summary>
    private static int DecodeBgr(string path, ref byte[]? buf)
    {
        using var bmp = SkiaSharp.SKBitmap.Decode(path);
        int w = bmp.Width, h = bmp.Height;
        var bgr = Grow(ref buf, w * h * 3);
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
        return w * h * 3;
    }

    private static int Check(string dir, string gcPath, string drPath, string tag)
    {
        float[] input = File.ReadAllBytes(Path.Combine(dir, $"{tag}_input.f32")) is { } raw
            ? ToF32(raw) : throw new Exception("no input");
        int[] shape = ReadShape(dir, tag, "input");
        float[] expect = ToF32(File.ReadAllBytes(Path.Combine(dir, $"{tag}_shadow.f32")));

        var m = OnnxModel.Load(gcPath);
        var cm = CompiledModel.Compile(m, shape);
        Console.WriteLine($"gcnet: {cm.NodeCount} nodes, arena {cm.ArenaFloats * 4L / 1024.0 / 1024:F1} MB");
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

    // run each node with progress printed — for hunting hangs at odd sizes
    private static int Trace(string gcPath, string drPath, int h, int w)
    {
        foreach (var (path, c) in new[] { (gcPath, 3), (drPath, 6) })
        {
            var plan = GraphCompiler.Compile(OnnxModel.Load(path), [1, c, h, w]);
            long maxT = plan.Values.Max(v => v.Shape.Aggregate(1L, (a, x) => a * x));
            Console.WriteLine($"max tensor {maxT * 4 / 1048576.0:F0} MB, arena {plan.ArenaFloats * 4L / 1048576.0:F0} MB");
            using var s = new InferenceSession(plan);
            new Random(0).NextBytes(MemoryMarshal.AsBytes(s.Input));
            Console.WriteLine($"== {Path.GetFileName(path)} [{c}x{h}x{w}] {plan.Nodes.Length} nodes ==");
            for (int i = 0; i < plan.Nodes.Length; i++)
            {
                var nd = plan.Nodes[i];
                string io = $"{string.Join('x', plan.Values[nd.In[0]].Shape)}->{string.Join('x', plan.Values[nd.Out[0]].Shape)}";
                if (nd.P is ConvP cp) io += $" w={string.Join('x', plan.Values[nd.In[1]].Shape)} g={cp.Group}";
                Console.Write($"[{i,3}] {nd.Op} {nd.Name} {io}  ");
                Console.Out.Flush();
                long t = Stopwatch.GetTimestamp();
                s.ExecNode(i);
                Console.WriteLine($"{(Stopwatch.GetTimestamp() - t) * 1000.0 / Stopwatch.Frequency:F1} ms");
                Console.Out.Flush();
            }
        }
        return 0;
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
            if (Environment.GetEnvironmentVariable("GCDRNET_PROF_NODES") == "1")
                foreach (var (tag, ss) in new[] { ("gc", s1), ("dr", s2) })
                {
                    Console.WriteLine($"--- {tag} top nodes (ms/run) ---");
                    foreach (var (i, n, t) in ss.NodeProfile().OrderByDescending(x => x.t).Take(int.TryParse(Environment.GetEnvironmentVariable("GCDRNET_PROF_TOP"), out int top) ? top : 14))
                    {
                        string extra = n.P is ConvP cp ? $" k{cp.Kernel[0]} s{cp.Strides[0]} g{cp.Group}" : "";
                        Console.WriteLine($"  #{i,-3} {n.Op,-14}{extra,-12} {string.Join('x', ss.ShapeOf(n.In[0]))}->{string.Join('x', ss.ShapeOf(n.Out[0]))}  {t.TotalMilliseconds / (reps + 1),7:F2}");
                    }
                }
        }
        return 0;
    }

    private static int BenchVk(string gcPath, string drPath, int h, int w, int reps) =>
        BenchE2e(gcPath, drPath, h, w, reps, GcdrBackend.Vulkan, "vulkan");

    private static int BenchE2e(string gcPath, string drPath, int h, int w, int reps, GcdrBackend backend, string tag)
    {
        // full pipeline: BGR u8 -> enhance -> BGR u8 (Vulkan: everything on GPU)
        using var enh = GcdrEnhancer.FromFiles(gcPath, drPath, backend);
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
        Console.WriteLine($"{tag} e2e {w}x{h}: {best / Stopwatch.Frequency * 1000:F1} ms" + (backend == GcdrBackend.Vulkan ? " (incl. upload/download)" : ""));
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
