using System.Globalization;
using System.Text;
using System.Text.Json;

// Merges the *.json written by `Sdcb.SimdGcdrNet.Benchmark --ci smoke|bench` into a markdown report:
//   dotnet run --project tests/Sdcb.SimdGcdrNet.TestReport -- --in results --out report.md
string inDir = "results", outPath = "report.md";
for (int i = 0; i + 1 < args.Length; i++)
{
    if (args[i] == "--in") inDir = args[++i];
    else if (args[i] == "--out") outPath = args[++i];
}
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

var results = new List<Result>();
var unreadable = new List<string>();
if (Directory.Exists(inDir))
    foreach (string f in Directory.EnumerateFiles(inDir, "*.json", SearchOption.AllDirectories).Order())
    {
        try { results.Add(Result.Load(f)); }
        catch (Exception ex) when (ex is JsonException or KeyNotFoundException or InvalidOperationException)
        { unreadable.Add($"{Path.GetFileName(f)}: {ex.Message}"); }
    }

// dotnet test --logger trx: one {case}.trx per job
var trx = new List<(string name, int total, int passed, int failed, int skipped, List<string> failures)>();
if (Directory.Exists(inDir))
    foreach (string f in Directory.EnumerateFiles(inDir, "*.trx", SearchOption.AllDirectories).Order())
    {
        try
        {
            var x = System.Xml.Linq.XDocument.Load(f);
            System.Xml.Linq.XNamespace ns = x.Root!.Name.Namespace;
            var c = x.Root.Element(ns + "ResultSummary")!.Element(ns + "Counters")!;
            int A(string n) => int.TryParse((string?)c.Attribute(n), out int v) ? v : 0;
            var failures = x.Root.Descendants(ns + "UnitTestResult")
                .Where(r => (string?)r.Attribute("outcome") == "Failed").Select(r => (string)r.Attribute("testName")!).ToList();
            trx.Add((Path.GetFileNameWithoutExtension(f), A("total"), A("passed"), A("failed"), A("total") - A("executed"), failures));
        }
        catch (Exception ex) when (ex is System.Xml.XmlException or NullReferenceException)
        { unreadable.Add($"{Path.GetFileName(f)}: {ex.Message}"); }
    }

var smoke = results.Where(r => r.Kind == "smoke").OrderBy(r => r.CaseId, StringComparer.Ordinal).ToList();
var bench = results.Where(r => r.Kind == "bench").ToList();

var md = new StringBuilder();
md.AppendLine("# SimdGcdrNet CI report").AppendLine();
md.AppendLine($"{trx.Count} test jobs ({trx.Count(t => t.failed > 0)} with failures), {smoke.Count} corpus smoke results " +
    $"({smoke.Count(r => !r.Ok)} failed), {bench.Count} benchmark results in {bench.Select(r => r.CaseId).Distinct().Count()} cases.").AppendLine();

md.AppendLine("## Unit tests (dotnet test)").AppendLine();
if (trx.Count == 0) md.AppendLine("_no test results_").AppendLine();
else
{
    md.AppendLine("| Job | Total | Passed | Failed | Skipped | Status |");
    md.AppendLine("|---|--:|--:|--:|--:|---|");
    foreach (var t in trx)
        md.AppendLine($"| {t.name} | {t.total} | {t.passed} | {t.failed} | {t.skipped} | {(t.failed == 0 ? "✅" : "❌")} |");
    md.AppendLine();
    foreach (var t in trx.Where(t => t.failed > 0))
    {
        md.AppendLine($"**{t.name}** failed:").AppendLine();
        foreach (string n in t.failures.Take(30)) md.AppendLine($"- `{n}`");
        md.AppendLine();
    }
}

md.AppendLine("## Corpus smoke (correctness)").AppendLine();
if (smoke.Count == 0) md.AppendLine("_no smoke results_").AppendLine();
else
{
    md.AppendLine("| Case | Backend | Runtime | CPU | ISA | Images | PSNR vs GT (input → output) | min vs CPU | Mean ms | Peak WS | Status |");
    md.AppendLine("|---|---|---|---|---|--:|---|--:|--:|--:|---|");
    foreach (Result r in smoke)
        md.AppendLine($"| {r.CaseId} | {r.Backend} | {r.Rid} | {r.Cpu} | {r.Isa}{Env(r)} | {r.Images.Count} | " +
            $"{F(r.PsnrIn, "0.00")} → {F(r.Psnr, "0.00")} dB | {F(r.MinVsCpu, "0.00", " dB")} | {F(r.MeanMs, "0")} | " +
            $"{F(r.PeakWsMb, "0", " MB")} | {(r.Ok ? "✅" : "❌")} |");
    md.AppendLine();
    foreach (Result r in smoke.Where(r => !r.Ok))
    {
        md.AppendLine($"**{r.CaseId}** failed:").AppendLine();
        foreach (string e in r.Errors) md.AppendLine($"- {e}");
        md.AppendLine();
    }
}

md.AppendLine("## Benchmark (performance)").AppendLine();
if (bench.Count == 0) md.AppendLine("_no benchmark results_").AppendLine();
else
{
    md.AppendLine("Per replica: mean ms over the timed images (first image is an untimed warmup; every other image " +
        "timed once, cold, decode excluded). Table: median / min / max over replicas; the ratio compares with " +
        "the platform's first case.").AppendLine();
    var cases = bench.GroupBy(r => r.CaseId).Select(g => new Case(g.Key, g.OrderBy(r => r.Replica).ToList()))
        .OrderBy(c => c.Platform, StringComparer.Ordinal).ThenBy(c => c.Order).ToList();
    md.AppendLine("| Case | Backend | CPU (runners) | ISA | Replicas | Images | Mean ms/img | min – max | MP/s | vs baseline | PSNR | Peak WS |");
    md.AppendLine("|---|---|---|---|--:|--:|--:|---|--:|--:|--:|--:|");
    foreach (var platform in cases.GroupBy(c => c.Platform))
    {
        Case baseline = platform.First();
        foreach (Case c in platform)
        {
            string ratio = c == baseline ? "1.00× (baseline)" : $"{baseline.Median / c.Median:0.00}×";
            md.AppendLine($"| {c.Id} | {c.Runs[0].Backend} | {c.Cpus} | {c.Runs[0].Isa}{Env(c.Runs[0])} | {c.Runs.Count} | " +
                $"{c.Runs[0].Timed} | **{c.Median:0.0}** | {c.Min:0.0} – {c.Max:0.0} | {c.MpPerSec:0.00} | {ratio} | " +
                $"{F(c.Runs[0].Psnr, "0.00", " dB")} | {c.Runs.Max(r => r.PeakWsMb ?? 0):0} MB |");
        }
    }
    md.AppendLine();

    md.AppendLine("<details><summary>Per-image ms (median over replicas)</summary>").AppendLine();
    string[] names = cases.SelectMany(c => c.Runs[0].Images.Where(i => !i.Warmup).Select(i => i.Name)).Distinct().Order().ToArray();
    md.AppendLine("| Image | " + string.Join(" | ", cases.Select(c => c.Id)) + " |");
    md.AppendLine("|---|" + string.Concat(cases.Select(_ => "--:|")));
    foreach (string n in names)
    {
        var cells = cases.Select(c =>
        {
            var ms = c.Runs.SelectMany(r => r.Images.Where(i => i.Name == n && !i.Warmup).Select(i => i.Ms)).ToList();
            return ms.Count == 0 ? "" : Median(ms).ToString("0");
        });
        var size = cases.SelectMany(c => c.Runs[0].Images).First(i => i.Name == n);
        md.AppendLine($"| {n} ({size.W}×{size.H}) | " + string.Join(" | ", cells) + " |");
    }
    md.AppendLine().AppendLine("</details>").AppendLine();
}

if (unreadable.Count > 0)
{
    md.AppendLine("## Unreadable result files").AppendLine();
    foreach (string u in unreadable) md.AppendLine($"- {u}");
    md.AppendLine();
}

Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
File.WriteAllText(outPath, md.ToString());
Console.WriteLine($"{results.Count} results -> {outPath}");
return 0;

static string F(double? v, string fmt, string unit = "") => v is { } d ? d.ToString(fmt) + unit : "–";

static string Env(Result r) => r.Env.Length > 0 ? $" ({r.Env})" : "";

static double Median(List<double> v)
{
    v.Sort();
    return v.Count % 2 == 1 ? v[v.Count / 2] : (v[v.Count / 2 - 1] + v[v.Count / 2]) / 2;
}

sealed record Image(string Name, int W, int H, double Ms, bool Warmup, double? PsnrVsCpu);

sealed record Result(string Kind, string CaseId, int Replica, string Backend, string Rid, string Cpu, string Isa, string Env,
    bool Ok, List<string> Errors, List<Image> Images, int Timed, double? MeanMs, double? MpPerSec, double? Psnr, double? PsnrIn,
    double? PeakWsMb)
{
    public double? MinVsCpu => Images.Where(i => i.PsnrVsCpu.HasValue).Select(i => i.PsnrVsCpu).Min();

    public static Result Load(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement r = doc.RootElement, host = r.GetProperty("host"), s = r.GetProperty("summary");
        static double? D(JsonElement e, string n) => e.TryGetProperty(n, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;
        var images = r.GetProperty("images").EnumerateArray().Select(i => new Image(
            i.GetProperty("name").GetString()!, i.GetProperty("w").GetInt32(), i.GetProperty("h").GetInt32(),
            i.GetProperty("ms").GetDouble(), i.TryGetProperty("warmup", out var w) && w.GetBoolean(), D(i, "psnrVsCpu"))).ToList();
        return new Result(r.GetProperty("kind").GetString()!, r.GetProperty("caseId").GetString()!, r.GetProperty("replica").GetInt32(),
            r.GetProperty("backend").GetString()!, host.GetProperty("rid").GetString()!, host.GetProperty("cpu").GetString()!,
            host.GetProperty("isa").GetString()!, host.GetProperty("env").GetString()!,
            r.GetProperty("ok").GetBoolean(), r.GetProperty("errors").EnumerateArray().Select(e => e.GetString()!).ToList(),
            images, s.GetProperty("timed").GetInt32(), D(s, "meanMs"), D(s, "mpPerSec"), D(s, "psnr"), D(s, "psnrIn"), D(s, "peakWsMb"));
    }
}

sealed class Case(string id, List<Result> runs)
{
    // case ids are "<os>-<arch>-<variant>": the platform groups rows, the variant orders them
    // (the CI lists each platform's baseline first)
    private static readonly string[] s_variantOrder = ["default", "cpu", "noavx512", "noavx2", "metal", "vulkan"];

    public string Id => id;
    public List<Result> Runs => runs;
    public string Platform => id.LastIndexOf('-') is > 0 and var i ? id[..i] : id;
    public int Order => Array.IndexOf(s_variantOrder, id[(id.LastIndexOf('-') + 1)..]) is >= 0 and var o ? o : s_variantOrder.Length;
    private List<double> Means => runs.Where(r => r.MeanMs.HasValue).Select(r => r.MeanMs!.Value).ToList();
    public double Median => Means.Count > 0 ? MedianOf(Means) : double.NaN;
    public double Min => Means.DefaultIfEmpty(double.NaN).Min();
    public double Max => Means.DefaultIfEmpty(double.NaN).Max();
    public double MpPerSec => MedianOf(runs.Where(r => r.MpPerSec.HasValue).Select(r => r.MpPerSec!.Value).DefaultIfEmpty(double.NaN).ToList());
    public string Cpus => string.Join("<br>", runs.GroupBy(r => r.Cpu).Select(g => g.Count() > 1 ? $"{g.Key} ×{g.Count()}" : g.Key));

    private static double MedianOf(List<double> v)
    {
        v.Sort();
        return v.Count % 2 == 1 ? v[v.Count / 2] : (v[v.Count / 2 - 1] + v[v.Count / 2]) / 2;
    }
}
