using Sdcb.GcdrNet.Backends.Vulkan;
using Sdcb.GcdrNet.Engine;
using Sdcb.GcdrNet.Onnx;
using Xunit;

namespace Sdcb.GcdrNet.Tests;

/// <summary>Node-by-node GPU vs CPU comparison. Enable with GCDRNET_VKDBG=1.</summary>
public class VkDebugTests
{
    private static string GcPath => Path.Combine(Golden.Dir, "models", "gcdrnet_gcnet_sim.onnx");
    private static string DrPath => Path.Combine(Golden.Dir, "models", "gcdrnet_drnet_sim.onnx");

    [Theory]
    [InlineData("gc", "small", "input")]
    [InlineData("dr", "small", "dr_in")]
    public void Nodes_Match(string which, string tag, string inputName)
    {
        if (Environment.GetEnvironmentVariable("GCDRNET_VKDBG") != "1") return;
        try { _ = VkBackend.Default.Dev.DeviceName; }
        catch { return; }   // no Vulkan device — env-gated debug test, skip quietly

        OnnxModel m = OnnxModel.Load(which == "gc" ? GcPath : DrPath);
        float[] input = Golden.F32(tag, inputName);
        int[] shape = Golden.ShapeOf(tag, inputName);

        ExecPlan plan = GraphCompiler.Compile(m, shape);
        using var cpu = new InferenceSession(plan);
        using var gpu = VkBackend.Default.NewSession(m, shape);

        input.CopyTo(cpu.Input);
        input.CopyTo(gpu.Input);
        gpu.DebugUploadInput();

        var log = new StreamWriter(Path.Combine(Path.GetTempPath(), $"vkdbg_{which}_{tag}.txt"));
        int bad = 0;
        for (int i = 0; i < plan.Nodes.Length; i++)
        {
            ExecNode n = plan.Nodes[i];
            cpu.ExecNode(i);
            var (ds, dc) = VkBackend.Default.GraphFor(m, shape).NodeRanges[i];
            gpu.DebugRunRange(ds, dc);
            int ov = n.Out[0];
            CValue v = plan.Values[ov];
            float[] got = new float[v.ElementCount];
            gpu.DebugDownload(ov, got);
            float[] want = cpu.ValueSpan(ov).ToArray();
            float maxd = 0; int at = -1;
            for (int j = 0; j < want.Length; j++)
            {
                float d = Math.Abs(got[j] - want[j]);
                if (d > maxd) { maxd = d; at = j; }
            }
            string status = maxd > 1e-3 ? "  <-- BAD" : "";
            if (maxd > 1e-3)
            {
                bad++;
                if (bad <= 2)
                {
                    int c = (int)v.ElementCount;
                    int[] sh = v.Shape;
                    int cw = sh[^1], chh = sh.Length >= 2 ? sh[^2] : 1;
                    for (int j = 0; j < c; j++)
                        if (Math.Abs(got[j] - want[j]) > 1e-3)
                        {
                            int y = chh > 1 && cw > 0 ? (j / cw) % chh : j / cw;
                            int x = cw > 0 ? j % cw : 0;
                            int ch = cw > 0 && chh > 0 ? j / (cw * chh) : 0;
                            log.WriteLine($"      [{j}] (c{ch},y{y},x{x}) gpu={got[j],12:g6} cpu={want[j],12:g6}");
                        }
                    int badCnt = 0;
                    for (int j = 0; j < c; j++)
                        if (Math.Abs(got[j] - want[j]) > 1e-3) badCnt++;
                    log.WriteLine($"      badElems={badCnt}/{c}");
                    for (int j = Math.Max(0, at - 2); j < Math.Min(c, at + 3); j++)
                        log.WriteLine($"      [{j}] gpu={got[j],12:g6} cpu={want[j],12:g6}");
                }
            }
            log.WriteLine($"node[{i,3}] {n.Op,-14} {n.Name,-24} out={string.Join('x', v.Shape)} maxd={maxd:e3} @{at}{status}");
        }
        log.Close();
        Assert.True(bad == 0, $"{bad} nodes diverged — see %TEMP%/vkdbg_{which}_{tag}.txt");
    }
}
