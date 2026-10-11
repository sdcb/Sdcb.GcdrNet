using System.Runtime.InteropServices;

namespace Sdcb.GcdrNet.Backends.Metal;

// Thin Metal wrappers over ObjC msgSend. +1-owned objects are released on
// Dispose; autoreleased objects (commandBuffer, computeCommandEncoder, …) are
// only valid inside the current NSAutoreleasePool — all public entry points
// that create them take their own pool.

// MTLResourceOptions: StorageModeShared = 0 << 4 = 0 (the zero value), which
// is what the whole backend uses — unified memory means the same allocation
// is CPU- and GPU-visible without didModifyRange/coherence calls.
internal static class MtlStorageMode { public const nuint Shared = 0; }

// MTLDataType values (MTLDataType.h) — only what NewPipeline's intConstants
// path uses.
internal static class MtlDataType
{
    public const nuint Int = 29;
}

internal sealed class MtlDevice : IDisposable
{
    public readonly IntPtr H;
    private readonly IntPtr _queue;
    // shader library + PSO cache, shared by every MetalGraphModel on this
    // device (each PaddleOcrAll carries det/cls/rec models — per-model
    // libraries would recompile ~2.5k lines of MSL three times and leak the
    // whole set on every release). Disposed with the device.
    private MtlLibrary? _lib;
    private readonly Dictionary<string, MtlPipeline> _pipes = new();

    private MtlDevice(IntPtr h)
    {
        H = h;
        // newCommandQueue is +1 — owned, released in Dispose.
        _queue = ObjC.Send0(h, ObjC.S.NewCommandQueue);
        if (_queue == 0) throw new InvalidOperationException("Metal: newCommandQueue returned nil");
    }

    public IntPtr Queue => _queue;

    public string Name => ObjC.NsToString(ObjC.Send0(H, ObjC.S.Name)) ?? "?";

    // MTLGPUFamily enum values (MTLGPUFamily.h)
    public bool SupportsFamily(int family) => ObjC.SendBool1L(H, ObjC.S.SupportsFamily, family) != 0;
    public bool SupportsApple1 => SupportsFamily(1001);
    public bool SupportsApple2 => SupportsFamily(1002);
    public bool SupportsApple3 => SupportsFamily(1003);
    public bool SupportsApple4 => SupportsFamily(1004);
    public bool SupportsApple5 => SupportsFamily(1005);
    public bool SupportsApple6 => SupportsFamily(1006);
    public bool SupportsApple7 => SupportsFamily(1007);
    public bool SupportsApple8 => SupportsFamily(1008);
    public bool SupportsApple9 => SupportsFamily(1009);
    public bool SupportsMetal3 => SupportsFamily(5001);
    public bool SupportsMetal4 => SupportsFamily(5002);

    public nuint MaxThreadgroupMemoryLength => ObjC.SendNuint(H, ObjC.S.MaxThreadgroupMemoryLength);

    /// <summary>Defensive check for optional selectors — an unrecognized
    /// selector on this paravirt driver raises an ObjC exception that kills
    /// the process, so never call a maybe-absent method unconditionally.</summary>
    public bool RespondsTo(IntPtr sel) => ObjC.SendBool1P(H, ObjC.S.RespondsToSelector, sel) != 0;
    public bool HasUnifiedMemory => ObjC.SendBool(H, ObjC.S.HasUnifiedMemory) != 0;
    public ulong RecommendedMaxWorkingSetSize => (ulong)ObjC.SendNuint(H, ObjC.S.RecommendedMaxWorkingSetSize);
    public nuint CurrentAllocatedSize => ObjC.SendNuint(H, ObjC.S.CurrentAllocatedSize);
    public ulong RegistryID => (ulong)ObjC.SendNuint(H, ObjC.S.RegistryID);

    public static MtlDevice? Probe()
    {
        if (!OperatingSystem.IsMacOS()) return null;
        using var pool = AutoReleasePool.Create();
        // MTLCreateSystemDefaultDevice follows the Create rule: already +1.
        IntPtr h = ObjC.CreateSystemDefaultDevice();
        return h == 0 ? null : new MtlDevice(h);
    }

    /// <summary>Pipeline for a kernel in the embedded shader library,
    /// compiled once and cached per device.</summary>
    internal MtlPipeline Pipe(string name)
    {
        lock (_pipes)
        {
            if (!_pipes.TryGetValue(name, out MtlPipeline? p))
                _pipes[name] = p = NewPipeline(ShaderLibrary(), name);
            return p;
        }
    }

    private MtlLibrary ShaderLibrary()
    {
        if (_lib is not null) return _lib;
        var asm = typeof(MtlDevice).Assembly;
        var names = asm.GetManifestResourceNames()
            .Where(n => n.StartsWith("Sdcb.GcdrNet.Backends.Metal.Shaders.")
                        && n.EndsWith(".metal"))
            .OrderBy(n => n).ToArray();
        if (names.Length == 0)
            throw new FileNotFoundException("Metal: no embedded .metal shaders");
        var sb = new System.Text.StringBuilder();
        foreach (var n in names)
        {
            using Stream s = asm.GetManifestResourceStream(n)!;
            using var r = new StreamReader(s);
            sb.AppendLine(r.ReadToEnd());
        }
        return _lib = NewLibrary(sb.ToString());
    }

    public MtlBuffer NewBuffer(nuint bytes)
    {
        // MTLResourceStorageModeShared = 0 options.
        IntPtr b = ObjC.Send1N1N(H, ObjC.S.NewBufferWithLengthOptions, bytes, 0);
        if (b == 0) throw new InvalidOperationException($"Metal: newBufferWithLength({bytes}) failed");
        return new MtlBuffer(b, bytes);
    }

    public unsafe MtlBuffer NewBuffer(void* src, nuint bytes)
    {
        IntPtr b = ObjC.Send1P2N(H, ObjC.S.NewBufferWithBytesLengthOptions, (IntPtr)src, bytes, 0);
        if (b == 0) throw new InvalidOperationException($"Metal: newBufferWithBytes({bytes}) failed");
        return new MtlBuffer(b, bytes);
    }

    /// <summary>Compile MSL source into a library. Throws with the NSError
    /// localizedDescription on failure — MSL compile errors must never be
    /// silently swallowed.</summary>
    public unsafe MtlLibrary NewLibrary(string source, bool fastMath = false)
    {
        using var pool = AutoReleasePool.Create();
        IntPtr src = ObjC.NsStr(source);
        IntPtr opts = MakeCompileOptions(fastMath);
        IntPtr err = 0;
        IntPtr lib;
        lib = ObjC.Send3P(H, ObjC.S.NewLibraryWithSourceOptionsError, src, opts, (IntPtr)(&err));
        ObjC.Release(src);
        ObjC.Release(opts);
        if (lib == 0) throw new InvalidOperationException("MSL compile failed: " + DescribeError(err));
        return new MtlLibrary(lib);
    }

    private static IntPtr MakeCompileOptions(bool fastMath)
    {
        // MTLCompileOptions: mathMode is a private-ish key; the public knob is
        // fastMathEnabled (deprecated in MSL 3.x but still honored) plus
        // languageVersion. MSL 3.1 = 0x30001 << 16? No — MTLLanguageVersion is
        // (major<<16)+minor: 2.4=0x20004, 3.1=0x30001. We request 3.1 when
        // supported for simdgroup_matrix; harmless if clamped.
        IntPtr o = ObjC.Send0(ObjC.Send0(ObjC.ClassMTLCompileOptions, ObjC.S.Alloc), ObjC.S.Init);
        ObjC.SendV1N(o, ObjC.S.SetLanguageVersion, (nuint)0x00030001); // MSL 3.1
        ObjC.SendV1B(o, ObjC.S.SetFastMathEnabled, (byte)(fastMath ? 1 : 0));
        return o;
    }

    public unsafe MtlPipeline NewPipeline(MtlLibrary lib, string fnName, int[]? intConstants = null)
    {
        using var pool = AutoReleasePool.Create();
        IntPtr name = ObjC.NsStr(fnName);
        IntPtr fn;
        if (intConstants is { Length: > 0 })
        {
            IntPtr fc = ObjC.Send0(ObjC.Send0(ObjC.ClassMTLFunctionConstantValues, ObjC.S.Alloc), ObjC.S.Init);
            for (int i = 0; i < intConstants.Length; ++i)
            {
                int v = intConstants[i];
                // setConstantValue:type:atIndex: — (const void*, MTLDataType, NSUInteger).
                ObjC.Send3P(fc, ObjC.S.SetConstantValueTypeAtIndex, (IntPtr)(&v), (IntPtr)(nint)(long)MtlDataType.Int, (IntPtr)(nint)i);
            }
            IntPtr err = 0;
            fn = ObjC.Send3P(lib.H, ObjC.S.NewFunctionWithNameConstantValuesError, name, fc, (IntPtr)(&err));
            ObjC.Release(fc);
            if (fn == 0) { ObjC.Release(name); throw new InvalidOperationException($"Metal: function '{fnName}' failed: {DescribeError(err)}"); }
        }
        else
        {
            fn = ObjC.Send1P(lib.H, ObjC.S.NewFunctionWithName, name);
        }
        ObjC.Release(name);
        if (fn == 0) throw new InvalidOperationException($"Metal: no function '{fnName}' in library");

        IntPtr err2 = 0;
        IntPtr pso = ObjC.Send2P(H, ObjC.S.NewComputePipelineStateWithFunctionError, fn, (IntPtr)(&err2));
        ObjC.Release(fn);
        if (pso == 0) throw new InvalidOperationException($"Metal: PSO '{fnName}' failed: {DescribeError(err2)}");
        return new MtlPipeline(pso, fnName);
    }

    internal static unsafe string DescribeError(IntPtr err)
    {
        if (err == 0) return "(no NSError)";
        IntPtr desc = ObjC.Send0(err, ObjC.S.LocalizedDescription);
        return ObjC.NsToString(desc) ?? "(unreadable NSError)";
    }

    /// <summary>Begin a command buffer + compute encoder pair. Both are
    /// autoreleased — caller must hold an AutoReleasePool open.</summary>
    public MtlEncoder BeginCommands()
    {
        IntPtr cmd = ObjC.Send0(_queue, ObjC.S.CommandBuffer);
        IntPtr enc = ObjC.Send0(cmd, ObjC.S.ComputeCommandEncoder);
        return new MtlEncoder(cmd, enc);
    }

    /// <summary>Begin encoder with a specific dispatch type
    /// (MTLDispatchType: 0=Serial, 1=Concurrent — concurrent allows the GPU to
    /// overlap dispatches that don't depend on each other; serial inserts an
    /// implicit barrier between dispatches which is what the schedule assumes).</summary>
    public MtlEncoder BeginCommands(bool concurrent)
    {
        IntPtr cmd = ObjC.Send0(_queue, ObjC.S.CommandBuffer);
        IntPtr enc = concurrent
            ? ObjC.Send1N(cmd, ObjC.S.ComputeCommandEncoderWithDispatchType, 1)
            : ObjC.Send0(cmd, ObjC.S.ComputeCommandEncoder);
        return new MtlEncoder(cmd, enc);
    }

    public void Dispose()
    {
        lock (_pipes)
        {
            foreach (MtlPipeline p in _pipes.Values) p.Dispose();
            _pipes.Clear();
        }
        _lib?.Dispose();
        ObjC.Release(_queue);
        ObjC.Release(H);
    }
}

internal sealed class MtlBuffer : IDisposable
{
    public IntPtr H;
    public readonly nuint Length;
    public MtlBuffer(IntPtr h, nuint length) { H = h; Length = length; }
    public IntPtr Contents => ObjC.Send0(H, ObjC.S.Contents);
    public void Dispose() { ObjC.Release(H); H = 0; }
}

internal sealed class MtlLibrary : IDisposable
{
    public readonly IntPtr H;
    public MtlLibrary(IntPtr h) => H = h;
    public void Dispose() => ObjC.Release(H);
}

internal sealed class MtlPipeline : IDisposable
{
    public readonly IntPtr Pso;
    public readonly string Name;
    public readonly nuint MaxTotalThreadsPerThreadgroup;
    public readonly nuint ThreadExecutionWidth;
    public MtlPipeline(IntPtr pso, string name)
    {
        Pso = pso;
        Name = name;
        MaxTotalThreadsPerThreadgroup = ObjC.SendNuint(pso, ObjC.S.MaxTotalThreadsPerThreadgroup);
        ThreadExecutionWidth = ObjC.SendNuint(pso, ObjC.S.ThreadExecutionWidth);
    }
    public void Dispose() => ObjC.Release(Pso);
}

/// <summary>One-shot command-buffer + compute-encoder pair. The objects are
/// autoreleased; keep the enclosing AutoReleasePool alive until Commit has
/// been issued (and waitUntilCompleted returned if you read results).</summary>
internal readonly struct MtlEncoder
{
    public readonly IntPtr Cmd;
    public readonly IntPtr Enc;

    public MtlEncoder(IntPtr cmd, IntPtr enc) { Cmd = cmd; Enc = enc; }

    public void SetPipeline(MtlPipeline p) => ObjC.Send1P(Enc, ObjC.S.SetComputePipelineState, p.Pso);

    public void SetBuffer(MtlBuffer buf, ulong byteOffset, int index) =>
        ObjC.SendV1P2N(Enc, ObjC.S.SetBufferOffsetAtIndex, buf.H, (nuint)byteOffset, (nuint)index);

    public unsafe void SetPc<T>(in T value, int index) where T : unmanaged
    {
        fixed (T* p = &value)
            ObjC.SendV1P2N(Enc, ObjC.S.SetBytesLengthAtIndex, (IntPtr)p, (nuint)sizeof(T), (nuint)index);
    }

    public unsafe void SetPc(void* data, nuint bytes, int index) =>
        ObjC.SendV1P2N(Enc, ObjC.S.SetBytesLengthAtIndex, (IntPtr)data, bytes, (nuint)index);

    public void SetThreadgroupMemory(nuint bytes, int index) =>
        ObjC.SendV1N1N(Enc, ObjC.S.SetThreadgroupMemoryLengthAtIndex, bytes, (nuint)index);

    public void Dispatch(nuint gx, nuint gy, nuint gz, nuint tx, nuint ty, nuint tz) =>
        ObjC.SendV2Size(Enc, ObjC.S.DispatchThreadgroupsThreadsPerThreadgroup,
            new ObjC.MTLSize(gx, gy, gz), new ObjC.MTLSize(tx, ty, tz));

    /// <summary>MTLBarrierScope buffers(=1): visibility barrier between
    /// dispatches — needed when a later dispatch consumes a buffer written by
    /// an earlier one and we used a Concurrent encoder. Serial encoders imply
    /// this already.</summary>
    public void BarrierBuffers() => ObjC.SendV1N(Enc, ObjC.S.MemoryBarrierWithScope, 1);

    public void End() => ObjC.Send0(Enc, ObjC.S.EndEncoding);

    public void Commit() => ObjC.Send0(Cmd, ObjC.S.Commit);

    /// <summary>Wait for completion and check status; throws with NSError on
    /// any non-completed terminal state (incl. GPU watchdog kills).</summary>
    public void CommitAndWait()
    {
        Commit();
        ObjC.Send0(Cmd, ObjC.S.WaitUntilCompleted);
        long status = (long)ObjC.SendLong(Cmd, ObjC.S.Status);
        // MTLCommandBufferStatus: 4=Completed; 5=Error; others shouldn't occur post-wait.
        if (status != 4)
        {
            IntPtr err = ObjC.Send0(Cmd, ObjC.S.Error);
            throw new InvalidOperationException($"Metal command buffer failed (status={status}): {MtlDevice.DescribeError(err)}");
        }
    }

    /// <summary>GPU-side wall time of the buffer (seconds). 0 when timestamps
    /// are unavailable (paravirt drivers often return 0).</summary>
    public double GpuDurationSeconds()
    {
        double s = ObjC.SendDouble(Cmd, ObjC.S.GpuStartTime);
        double e = ObjC.SendDouble(Cmd, ObjC.S.GpuEndTime);
        return e > s ? e - s : 0;
    }
}
