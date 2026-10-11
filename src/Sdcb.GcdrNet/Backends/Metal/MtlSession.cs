using Sdcb.GcdrNet.Engine;

namespace Sdcb.GcdrNet.Backends.Metal;

/// <summary>
/// GPU counterpart of <see cref="InferenceSession"/>: one activation arena in
/// shared (CPU+GPU-visible) memory; Run() writes the input window straight
/// into the arena, records the graph's dispatches into one command buffer and
/// waits once — no staging copies (unified memory). Not thread-safe.
/// </summary>
public sealed unsafe class MtlInferenceSession : IDisposable
{
    private readonly MtlDevice _dev;
    private readonly MtlGraphModel _graph;
    private readonly ExecPlan _plan;

    internal readonly MtlBuffer Arena;    // shared-memory activation arena
    private readonly float[] _inHost, _outHost;
    private readonly ulong _inOff, _outOff;
    private readonly int _inFloats, _outFloats;
    internal int Es => _graph.Es;         // bytes per stored element (2 = fp16)

    internal MtlInferenceSession(MtlDevice dev, ExecPlan plan, MtlGraphModel graph)
    {
        _dev = dev;
        _plan = plan;
        _graph = graph;
        _graph.Acquire();

        Arena = dev.NewBuffer((nuint)((ulong)plan.ArenaFloats * (ulong)Es + 64));
        _inFloats = checked((int)plan.Values[plan.InputValue].ElementCount);
        _outFloats = checked((int)plan.Values[plan.OutputValue].ElementCount);
        _inOff = (ulong)plan.BufOffset[plan.Values[plan.InputValue].Buf] * (ulong)Es;
        _outOff = (ulong)plan.BufOffset[plan.Values[plan.OutputValue].Buf] * (ulong)Es;
        _inHost = new float[_inFloats];
        _outHost = new float[_outFloats];
    }

    internal MtlBuffer ResolveBuf(MtlBind b) => Resolve(b).buf;
    internal ulong ResolveOff(MtlBind b) => b.Offset;
    internal (MtlBuffer buf, ulong off, ulong size) Resolve(MtlBind b) => b.Slot switch
    {
        MtlSlot.Arena => (Arena, b.Offset, b.Size),
        MtlSlot.Consts => (_graph.Consts, b.Offset, b.Size),
        MtlSlot.Direct => (b.Buf!, b.Offset, b.Size),
        _ => (_graph.Dummy, b.Offset, b.Size),
    };

    /// <summary>Input tensor shape (e.g. [1,3,H,W]).</summary>
    public int[] InputShape => _plan.Values[_plan.InputValue].Shape;
    public int[] OutputShape => _plan.Values[_plan.OutputValue].Shape;

    /// <summary>The writable input window (staged into the arena on <see cref="Run"/>).</summary>
    public Span<float> Input => _inHost;
    /// <summary>The output tensor after <see cref="Run"/>.</summary>
    public ReadOnlySpan<float> Output => _outHost;

    internal ulong InputOffset => _inOff;
    internal ulong OutputOffset => _outOff;
    internal ulong InputBytes => (ulong)_inFloats * (ulong)Es;
    internal ulong OutputBytes => (ulong)_outFloats * (ulong)Es;

    /// <summary>Host float -> stored elements (fp16 narrowing).</summary>
    internal static void ToStorage(ReadOnlySpan<float> src, void* dst)
    {
        var d = new Span<Half>(dst, src.Length);
        for (int i = 0; i < src.Length; i++) d[i] = (Half)src[i];
    }

    /// <summary>Stored elements -> host float (fp16 widening).</summary>
    internal static void FromStorage(void* src, Span<float> dst)
    {
        var sp = new ReadOnlySpan<Half>(src, dst.Length);
        for (int i = 0; i < dst.Length; i++) dst[i] = (float)sp[i];
    }

    /// <summary>Downloads dst.Length stored elements at byte offset <paramref name="off"/>.</summary>
    internal void DownloadElems(MtlBuffer buf, ulong off, Span<float> dst)
        => FromStorage((byte*)buf.Contents + off, dst);

    /// <summary>Record one dispatch on an open encoder.</summary>
    internal static void RecordOne(MtlEncoder enc, MtlGraphModel g,
        MtlDispatch d, MtlBuffer arena, ulong arenaBase = 0)
    {
        enc.SetPipeline(d.Pipe);
        for (int b = 0; b < d.Binds.Length; b++)
        {
            MtlBind mb = d.Binds[b];
            (MtlBuffer buf, ulong off, _) = mb.Slot switch
            {
                MtlSlot.Arena => (arena, arenaBase + mb.Offset, mb.Size),
                MtlSlot.Consts => (g.Consts, mb.Offset, mb.Size),
                MtlSlot.Direct => (mb.Buf!, mb.Offset, mb.Size),
                _ => (g.Dummy, mb.Offset, mb.Size),
            };
            enc.SetBuffer(buf, off, b);
        }
        if (d.Pc.Length > 0)
            fixed (byte* pc = d.Pc)
                enc.SetPc(pc, (nuint)d.Pc.Length, d.Binds.Length);
        enc.Dispatch(d.Gx, d.Gy, d.Gz, 256, 1, 1);
    }

    /// <summary>Record all dispatches on an open encoder (serial dispatch order).</summary>
    internal void RecordCompute(MtlEncoder enc)
    {
        foreach (var d in _graph.Dispatches)
            RecordOne(enc, _graph, d, Arena);
    }

    /// <summary>Debug/test: run the dispatches of ExecNode i (sequential state persists).</summary>
    internal void DebugRunNode(int i)
    {
        var (start, count) = _graph.NodeRanges[i];
        if (count > 0) DebugRunRange(start, count);
    }

    /// <summary>Debug/test: a value's arena window as host floats.</summary>
    internal float[] DebugValue(int valueId)
    {
        var dst = new float[checked((int)_plan.Values[valueId].ElementCount)];
        DebugDownload(valueId, dst);
        return dst;
    }

    /// <summary>Debug/test: run a single dispatch-list slice on its own command buffer.</summary>
    internal void DebugRunRange(int dStart, int dCount)
    {
        using var pool = AutoReleasePool.Create();
        var enc = _dev.BeginCommands();
        for (int i = dStart; i < dStart + dCount; i++)
            RecordOne(enc, _graph, _graph.Dispatches[i], Arena);
        enc.End();
        enc.CommitAndWait();
    }

    /// <summary>
    /// Debug/test: time each dispatch by running it alone (commit + wait) and
    /// wall-timing; includes ~45us launch overhead per dispatch — rank only.
    /// Returns name→best-of-2 ms.
    /// </summary>
    internal Dictionary<string, double> DebugProfile()
    {
        int n = _graph.Dispatches.Length;
        var best = new double[n];
        Array.Fill(best, double.MaxValue);
        // warm pass to keep state identical to production (input already in arena)
        using (var pool = AutoReleasePool.Create())
        {
            var enc = _dev.BeginCommands();
            foreach (var d in _graph.Dispatches) RecordOne(enc, _graph, d, Arena);
            enc.End();
            enc.CommitAndWait();
        }
        var sw = new System.Diagnostics.Stopwatch();
        for (int rep = 0; rep < 2; rep++)
            for (int i = 0; i < n; i++)
            {
                var d = _graph.Dispatches[i];
                using var pool = AutoReleasePool.Create();
                var enc = _dev.BeginCommands();
                RecordOne(enc, _graph, d, Arena);
                enc.End();
                sw.Restart();
                enc.CommitAndWait();
                double ms = sw.Elapsed.TotalMilliseconds;
                if (ms < best[i]) best[i] = ms;
            }
        var agg = new Dictionary<string, double>();
        PerDispatch = new (string, double)[n];
        for (int i = 0; i < n; i++)
        {
            string name = _graph.Dispatches[i].Name ?? $"d{i}";
            PerDispatch[i] = (name, best[i]);
            name = name[..(name.IndexOf('(') is int cut && cut > 0 ? cut : name.Length)];
            agg[name] = agg.TryGetValue(name, out double v) ? v + best[i] : best[i];
        }
        return agg;
    }

    /// <summary>Per-dispatch timings from the last <see cref="DebugProfile"/> call.</summary>
    internal (string Name, double Ms)[]? PerDispatch;

    /// <summary>Debug: the compiled plan backing this session.</summary>
    internal ExecPlan DebugPlan => _plan;

    /// <summary>Debug/test: const offset (ulong.MaxValue when not packed).</summary>
    internal ulong DebugConstOff(int valueId, int kind) => _graph.DebugConst(valueId, kind).off;

    /// <summary>Debug/test: const storage of a value (kind 0 raw,1 f32,2 convw,3 dww,4 convwt,5 gemmnk).</summary>
    internal float[] DebugConst(int valueId, int kind, int len)
    {
        var (buf, off, half) = _graph.DebugConst(valueId, kind);
        var dst = new float[len];
        if (half)
        {
            var h = new ReadOnlySpan<Half>((byte*)buf.Contents + off, len);
            for (int i = 0; i < len; i++) dst[i] = (float)h[i];
        }
        else
            new ReadOnlySpan<float>((byte*)buf.Contents + off, len).CopyTo(dst);
        return dst;
    }

    /// <summary>Debug/test: dispatch names for ExecNode i.</summary>
    internal string[] DebugNodeDispatches(int i)
    {
        var (st, cnt) = _graph.NodeRanges[i];
        return _graph.Dispatches.Skip(st).Take(cnt).Select(d => d.Name ?? "").ToArray();
    }

    /// <summary>Debug: read the whole arena (ArenaFloats floats) into dst.</summary>
    internal void DebugArena(Span<float> dst)
        => FromStorage(Arena.Contents.ToPointer(), dst.Slice(0, _plan.ArenaFloats));

    /// <summary>Debug: graph consts buffer + the session's arena buffer.</summary>
    internal (MtlBuffer consts, MtlBuffer arena) DebugBuffers => (_graph.Consts, Arena);

    /// <summary>Debug/test: download an activation value's region into dst.</summary>
    internal void DebugDownload(int valueId, float[] dst)
    {
        CValue v = _plan.Values[valueId];
        FromStorage((byte*)Arena.Contents + (ulong)_plan.BufOffset[v.Buf] * (ulong)Es,
            dst.AsSpan(0, checked((int)v.ElementCount)));
    }

    /// <summary>Debug/test: copy host input into the arena's input region.</summary>
    internal void DebugUploadInput()
        => ToStorage(_inHost, (byte*)Arena.Contents + _inOff);

    /// <summary>Write input into the arena and run all dispatches.</summary>
    public void Run()
    {
        ToStorage(_inHost, (byte*)Arena.Contents + _inOff);
        using var pool = AutoReleasePool.Create();
        var enc = _dev.BeginCommands();
        RecordCompute(enc);
        enc.End();
        enc.CommitAndWait();
        FromStorage((byte*)Arena.Contents + _outOff, _outHost);
    }

    public void Dispose()
    {
        Arena.Dispose();
        _graph.Release();
    }
}
