using Android.App;
using Android.OS;
using Android.Views;

namespace Sdcb.GcdrNet.AndroidBench;

/// <summary>
/// adb entry point. Extras: <c>args</c> (space-separated arguments for the linked
/// desktop <c>Program.Main</c>), <c>env</c> (<c>K=V;K=V</c>, applied before the
/// library is touched; runtime switches such as <c>DOTNET_EnableHWIntrinsic</c> must
/// instead be baked in with <c>-p:GcdrEnv=</c>, see the csproj) and <c>run</c>
/// (output id). Output goes to logcat (tag GcdrBench) and <c>files/out/&lt;run&gt;.log</c>;
/// <c>&lt;run&gt;.done</c> holds the exit code. The process kills itself afterwards
/// so every run starts cold.
/// </summary>
[Activity(Label = "GcdrBench", MainLauncher = true, Exported = true,
    Name = "com.sdcb.gcdrnet.bench.MainActivity")]
public class MainActivity : Activity
{
    protected override void OnCreate(Bundle? savedInstanceState)
    {
        base.OnCreate(savedInstanceState);
        Window?.AddFlags(WindowManagerFlags.KeepScreenOn);
        string args = Intent?.GetStringExtra("args") ?? "--bench";
        string env = Intent?.GetStringExtra("env") ?? "";
        string run = Intent?.GetStringExtra("run") ?? "run";
        string root = GetExternalFilesDir(null)!.AbsolutePath;
        string outDir = Path.Combine(root, "out");
        // app-created, so the app can read what adb pushes into them later
        // (directories made by the shell user are not readable by the app)
        foreach (string d in new[] { outDir, Path.Combine(root, "testdata") })
            Directory.CreateDirectory(d);
        File.Delete(Path.Combine(outDir, run + ".done"));

        System.Environment.SetEnvironmentVariable("GCDRNET_TESTDATA", Path.Combine(root, "testdata"));
        foreach (string kv in env.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            int eq = kv.IndexOf('=');
            if (eq > 0) System.Environment.SetEnvironmentVariable(kv[..eq], kv[(eq + 1)..]);
        }

        var worker = new Thread(() =>
        {
            int rc;
            using (var log = new TeeWriter(Path.Combine(outDir, run + ".log")))
            {
                Console.SetOut(log);
                Console.SetError(log);
                Console.WriteLine($"args: {args}");
                if (env.Length > 0) Console.WriteLine($"env: {env}");
                Console.WriteLine($"{System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription}; isa: AdvSimd={System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported} " +
                    $"Dp={System.Runtime.Intrinsics.Arm.Dp.IsSupported} Sve={System.Runtime.Intrinsics.Arm.Sve.IsSupported} " +
                    $"Vector<float>={System.Numerics.Vector<float>.Count} cpus={System.Environment.ProcessorCount} " +
                    $"apkEnv={System.Environment.GetEnvironmentVariable("GCDR_APK_ENV")}");
                try
                {
                    Directory.SetCurrentDirectory(root);
                    if (args.Trim() == "--init") rc = 0;
                    else rc = Sdcb.GcdrNet.Benchmark.Program.Main(args.Split(' ', StringSplitOptions.RemoveEmptyEntries));
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"FATAL {ex}");
                    rc = 99;
                }
                Console.WriteLine($"exit {rc}");
            }
            File.WriteAllText(Path.Combine(outDir, run + ".done"), rc.ToString());
            Process.KillProcess(Process.MyPid());
        }, 64 << 20);
        worker.Start();
    }
}

/// <summary>Line-buffered tee to logcat and a file.</summary>
sealed class TeeWriter(string path) : TextWriter
{
    private readonly StreamWriter _file = new(path, append: false) { AutoFlush = true };
    private readonly System.Text.StringBuilder _line = new();
    private readonly object _sync = new();
    public override System.Text.Encoding Encoding => System.Text.Encoding.UTF8;

    public override void Write(char value)
    {
        lock (_sync)
        {
            if (value == '\n') { FlushLine(); return; }
            if (value != '\r') _line.Append(value);
        }
    }

    public override void Write(string? value)
    {
        if (value is null) return;
        lock (_sync)
            foreach (char c in value)
            {
                if (c == '\n') FlushLine();
                else if (c != '\r') _line.Append(c);
            }
    }

    public override void WriteLine(string? value)
    {
        lock (_sync) { Write(value); FlushLine(); }
    }

    public override void Flush() { }

    private void FlushLine()
    {
        string s = _line.ToString();
        _line.Clear();
        Android.Util.Log.Info("GcdrBench", s);
        _file.WriteLine(s);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            lock (_sync) if (_line.Length > 0) FlushLine();
            _file.Dispose();
        }
        base.Dispose(disposing);
    }
}
