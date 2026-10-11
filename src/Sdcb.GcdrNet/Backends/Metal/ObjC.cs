using System.Runtime.InteropServices;

namespace Sdcb.GcdrNet.Backends.Metal;

// Minimal libobjc P/Invoke layer — pure net10.0 C#, no macOS workload.
// Ownership follows Cocoa rules: alloc/init/newXxx/copy return +1 (caller
// releases); everything else is autoreleased within the enclosing
// NSAutoreleasePool. Selectors and class handles are cached in static
// readonly fields — a dispatch already costs enough objc_msgSend calls that
// re-registering selector strings per call is wasted marshaling.
internal static unsafe partial class ObjC
{
    private const string LibObjC = "libobjc.dylib";
    private const string LibMetal = "/System/Library/Frameworks/Metal.framework/Metal";

    [LibraryImport(LibObjC, EntryPoint = "objc_getClass")]
    private static partial IntPtr GetClassRaw(IntPtr name);

    [LibraryImport(LibObjC, EntryPoint = "sel_registerName")]
    private static partial IntPtr RegSelRaw(IntPtr name);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial IntPtr Send0(IntPtr recv, IntPtr sel);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial IntPtr Send1P(IntPtr recv, IntPtr sel, IntPtr a);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial IntPtr Send2P(IntPtr recv, IntPtr sel, IntPtr a, IntPtr b);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial IntPtr Send3P(IntPtr recv, IntPtr sel, IntPtr a, IntPtr b, IntPtr c);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial IntPtr Send1N(IntPtr recv, IntPtr sel, nuint a);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial IntPtr Send1P1N(IntPtr recv, IntPtr sel, IntPtr a, nuint b);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial IntPtr Send2P1N(IntPtr recv, IntPtr sel, IntPtr a, IntPtr b, nuint c);

    // newBufferWithBytes:length:options: / newBufferWithLength:options:
    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial IntPtr Send1P2N(IntPtr recv, IntPtr sel, IntPtr a, nuint b, nuint c);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial IntPtr Send1N1N(IntPtr recv, IntPtr sel, nuint a, nuint b);

    // setBuffer:offset:atIndex: / setBytes:length:atIndex:
    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial void SendV1P2N(IntPtr recv, IntPtr sel, IntPtr a, nuint b, nuint c);

    // dispatchThreadgroups:threadsPerThreadgroup: — MTLSize by value (24B)
    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial void SendV2Size(IntPtr recv, IntPtr sel, MTLSize a, MTLSize b);

    // setThreadgroupMemoryLength:atIndex:
    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial void SendV1N1N(IntPtr recv, IntPtr sel, nuint a, nuint b);

    // single-NSUInteger void calls (setLanguageVersion:, memoryBarrierWithScope:, …)
    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial void SendV1N(IntPtr recv, IntPtr sel, nuint a);

    // single-BOOL void calls (setFastMathEnabled:…) — BOOL is 1 byte on arm64.
    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial void SendV1B(IntPtr recv, IntPtr sel, byte a);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial byte SendBool(IntPtr recv, IntPtr sel);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial byte SendBool1P(IntPtr recv, IntPtr sel, IntPtr a);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial byte SendBool1N(IntPtr recv, IntPtr sel, nuint a);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial nuint SendNuint(IntPtr recv, IntPtr sel);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial double SendDouble(IntPtr recv, IntPtr sel);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial long SendLong(IntPtr recv, IntPtr sel);

    // supportsFamily: — (id,SEL,NSInteger)
    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial byte SendBool1L(IntPtr recv, IntPtr sel, nint a);

    [LibraryImport(LibMetal, EntryPoint = "MTLCreateSystemDefaultDevice")]
    public static partial IntPtr CreateSystemDefaultDevice();

    [StructLayout(LayoutKind.Sequential)]
    public readonly struct MTLSize
    {
        public readonly nuint X, Y, Z;
        public MTLSize(nuint x, nuint y, nuint z) { X = x; Y = y; Z = z; }
    }

    [StructLayout(LayoutKind.Sequential)]
    public readonly struct NSRange
    {
        public readonly nuint Location, Length;
        public NSRange(nuint location, nuint length) { Location = location; Length = length; }
    }

    // ---- cached class/selector tables ------------------------------------
    // sel_registerName returns the same SEL for the same string, so register
    // once. Classes likewise are process-wide stable.

    private static readonly IntPtr s_classNsString = Cls("NSString");
    private static readonly IntPtr s_classNsAutoreleasePool = Cls("NSAutoreleasePool");
    private static readonly IntPtr s_classMtlCompileOptions = Cls("MTLCompileOptions");
    private static readonly IntPtr s_classMtlFunctionConstantValues = Cls("MTLFunctionConstantValues");

    public static IntPtr ClassNSString => s_classNsString;
    public static IntPtr ClassNSAutoreleasePool => s_classNsAutoreleasePool;
    public static IntPtr ClassMTLCompileOptions => s_classMtlCompileOptions;
    public static IntPtr ClassMTLFunctionConstantValues => s_classMtlFunctionConstantValues;

    public static IntPtr Cls(string name)
    {
        IntPtr n = Marshal.StringToHGlobalAnsi(name);
        try { return GetClassRaw(n); } finally { Marshal.FreeHGlobal(n); }
    }

    public static IntPtr Sel(string name)
    {
        IntPtr n = Marshal.StringToHGlobalAnsi(name);
        try { return RegSelRaw(n); } finally { Marshal.FreeHGlobal(n); }
    }

    public static class S
    {
        public static readonly IntPtr Alloc = Sel("alloc");
        public static readonly IntPtr Init = Sel("init");
        public static readonly IntPtr Release = Sel("release");
        public static readonly IntPtr Retain = Sel("retain");
        public static readonly IntPtr Drain = Sel("drain");
        public static readonly IntPtr InitWithUTF8String = Sel("initWithUTF8String:");
        public static readonly IntPtr Utf8String = Sel("UTF8String");
        public static readonly IntPtr LocalizedDescription = Sel("localizedDescription");
        public static readonly IntPtr Length = Sel("length");

        public static readonly IntPtr Name = Sel("name");
        public static readonly IntPtr RespondsToSelector = Sel("respondsToSelector:");
        public static readonly IntPtr SupportsFamily = Sel("supportsFamily:");
        public static readonly IntPtr MaxThreadgroupMemoryLength = Sel("maxThreadgroupMemoryLength");
        public static readonly IntPtr MaxThreadsPerThreadgroup = Sel("maxThreadsPerThreadgroup");
        public static readonly IntPtr MaxTotalThreadsPerThreadgroup = Sel("maxTotalThreadsPerThreadgroup");
        public static readonly IntPtr ThreadExecutionWidth = Sel("threadExecutionWidth");
        public static readonly IntPtr HasUnifiedMemory = Sel("hasUnifiedMemory");
        public static readonly IntPtr RecommendedMaxWorkingSetSize = Sel("recommendedMaxWorkingSetSize");
        public static readonly IntPtr CurrentAllocatedSize = Sel("currentAllocatedSize");
        public static readonly IntPtr RegistryID = Sel("registryID");

        public static readonly IntPtr NewCommandQueue = Sel("newCommandQueue");
        public static readonly IntPtr NewBufferWithBytesLengthOptions = Sel("newBufferWithBytes:length:options:");
        public static readonly IntPtr NewBufferWithLengthOptions = Sel("newBufferWithLength:options:");
        public static readonly IntPtr Contents = Sel("contents");
        public static readonly IntPtr NewLibraryWithSourceOptionsError = Sel("newLibraryWithSource:options:error:");
        public static readonly IntPtr NewFunctionWithName = Sel("newFunctionWithName:");
        public static readonly IntPtr NewFunctionWithNameConstantValuesError = Sel("newFunctionWithName:constantValues:error:");
        public static readonly IntPtr NewComputePipelineStateWithFunctionError = Sel("newComputePipelineStateWithFunction:error:");
        public static readonly IntPtr FunctionName = Sel("functionName");
        public static readonly IntPtr FunctionConstantsDictionary = Sel("functionConstantsDictionary");
        public static readonly IntPtr SetConstantValueTypeAtIndex = Sel("setConstantValue:type:atIndex:");

        public static readonly IntPtr SetFastMathEnabled = Sel("setFastMathEnabled:");
        public static readonly IntPtr SetMathMode = Sel("setMathMode:");
        public static readonly IntPtr SetLanguageVersion = Sel("setLanguageVersion:");

        public static readonly IntPtr CommandBuffer = Sel("commandBuffer");
        public static readonly IntPtr ComputeCommandEncoder = Sel("computeCommandEncoder");
        public static readonly IntPtr ComputeCommandEncoderWithDispatchType = Sel("computeCommandEncoderWithDispatchType:");
        public static readonly IntPtr SetComputePipelineState = Sel("setComputePipelineState:");
        public static readonly IntPtr SetBufferOffsetAtIndex = Sel("setBuffer:offset:atIndex:");
        public static readonly IntPtr SetBytesLengthAtIndex = Sel("setBytes:length:atIndex:");
        public static readonly IntPtr DispatchThreadgroupsThreadsPerThreadgroup = Sel("dispatchThreadgroups:threadsPerThreadgroup:");
        public static readonly IntPtr DispatchThreadsThreadsPerThreadgroup = Sel("dispatchThreads:threadsPerThreadgroup:");
        public static readonly IntPtr SetThreadgroupMemoryLengthAtIndex = Sel("setThreadgroupMemoryLength:atIndex:");
        public static readonly IntPtr MemoryBarrierWithScope = Sel("memoryBarrierWithScope:");
        public static readonly IntPtr EndEncoding = Sel("endEncoding");
        public static readonly IntPtr Commit = Sel("commit");
        public static readonly IntPtr WaitUntilCompleted = Sel("waitUntilCompleted");
        public static readonly IntPtr Status = Sel("status");
        public static readonly IntPtr Error = Sel("error");
        public static readonly IntPtr GpuStartTime = Sel("GPUStartTime");
        public static readonly IntPtr GpuEndTime = Sel("GPUEndTime");
        public static readonly IntPtr Label = Sel("label");
        public static readonly IntPtr SetLabel = Sel("setLabel:");
    }

    // NSString* with UTF-8 — caller owns (+1 via alloc/init); pass owns:false
    // to consume within the current autorelease pool.
    public static IntPtr NsStr(string s)
    {
        IntPtr obj = Send0(s_classNsString, S.Alloc);
        IntPtr cstr = Marshal.StringToHGlobalAnsi(s);
        try { return Send1P(obj, S.InitWithUTF8String, cstr); }
        finally { Marshal.FreeHGlobal(cstr); }
    }

    public static string? NsToString(IntPtr nsString)
    {
        if (nsString == 0) return null;
        IntPtr utf8 = Send0(nsString, S.Utf8String);
        return utf8 == 0 ? null : Marshal.PtrToStringUTF8(utf8);
    }

    public static void Release(IntPtr obj) { if (obj != 0) Send0(obj, S.Release); }
    public static IntPtr Retain(IntPtr obj) => Send0(obj, S.Retain);
}

/// <summary>Scoped NSAutoreleasePool — .NET worker threads have none, so every
/// public entry that allocates autoreleased ObjC objects wraps itself in one.
/// Drain at Dispose releases the pool itself.</summary>
internal readonly struct AutoReleasePool : IDisposable
{
    private readonly IntPtr _pool;
    private AutoReleasePool(IntPtr pool) => _pool = pool;

    public static AutoReleasePool Create() =>
        new(ObjC.Send0(ObjC.Send0(ObjC.ClassNSAutoreleasePool, ObjC.S.Alloc), ObjC.S.Init));

    public void Dispose() => ObjC.Send0(_pool, ObjC.S.Drain);
}
