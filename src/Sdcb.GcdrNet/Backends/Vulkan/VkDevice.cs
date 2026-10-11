// Thin object wrappers over Vk.cs: instance/device/queue, buffers with
// memory-type selection (ReBAR fast path vs host-visible staging), shader
// modules from embedded SPIR-V, compute pipelines, descriptor sets, and
// one-shot command submission. Adapted from Sdcb.SimdPaddleOCR.
using System.Runtime.InteropServices;

namespace Sdcb.GcdrNet.Backends.Vulkan;

internal unsafe sealed class VkDevice : IDisposable
{
    public IntPtr Instance, PhysDevice, Device, Queue;
    public string DeviceName = "";
    public uint VendorId;
    public uint SubgroupSize;
    public uint QueueFamily;
    public Vk.VkPhysicalDeviceMemoryProperties MemProps;
    public bool CoherentDeviceLocal;   // DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT exists (ReBAR)
    public bool StorageBuffer16;       // VK_KHR_16bit_storage: activations may be stored as float16
    public uint MaxSharedMemory;
    public ulong MaxStorageRange = uint.MaxValue;
    public bool PushDescriptors;
    /// <summary>Tensor-core convs (conv_cm.comp; GCDRNET_VK_COOPMAT=0 disables): VK_KHR_cooperative_matrix
    /// plus shaderFloat16, in one of two shapes. Default: a 16x16x16 fp16 x fp16 -> fp32 subgroup config
    /// on 32-lane subgroups; devices whose default compute subgroup is not 32 lanes (RDNA wave64) qualify
    /// only when VK_EXT_subgroup_size_control can pin it to 32 (<see cref="CoopSubgroupSize"/>).
    /// Otherwise <see cref="CoopM8"/>: the 8x16x16 config on subgroups pinned to 16 lanes (Intel Xe2).
    /// A probe dispatch must confirm the pinned width; failing that the convs stay on conv_tile.</summary>
    public bool CoopMat16;
    /// <summary>The coopmat convs use the SG16 build (8x16x16 fragments, 16-lane subgroups; Intel Arc
    /// XMX): taken only when the 16x16x16 / 32-lane path is unavailable (GCDRNET_VK_SG16=0 disables).</summary>
    public bool CoopM8;
    /// <summary>requiredSubgroupSize for the coopmat pipelines: 0 = device default (already 32),
    /// 32 / 16 = pinned through VK_EXT_subgroup_size_control.</summary>
    public uint CoopSubgroupSize;
    /// <summary>Coopmat convs use the LITE variants (single-buffered 28 KB staging, scalar epilogue,
    /// grouped raster): wave64-capable parts or less than the 48 KB shared memory the
    /// double-buffered kernels take.</summary>
    public bool CoopLite;
    /// <summary>VK_EXT_subgroup_size_control range (1..128 when the extension is absent).</summary>
    public uint SubgroupMin = 1, SubgroupMax = 128;
    /// <summary>Memory-bound NHWC shaders (maxpool, bilinear, depthwise, shift) take 4 channels per
    /// invocation when c % 4 == 0 — integrated GPUs, where they are bandwidth/issue bound, and Intel
    /// GPUs (Arc B580: 10.4 -> 6.7 ms for the four; per-element index math is costly on SIMD16)
    /// (GCDRNET_VK_VEC4=0 disables; other discrete GPUs keep the per-element shaders).</summary>
    public bool Vec4Eltwise;
    /// <summary>3x3 stem conv+pool uses convpool_k3 (input window rows in registers): Intel GPUs
    /// (Arc B580: 7.5 -> 4.9 ms for the two stems) (GCDRNET_VK_CONVPOOL3=0 disables).</summary>
    public bool ConvPoolK3;
    private const uint VendorIntel = 0x8086;
    /// <summary>Largest DEVICE_LOCAL heap (bytes).</summary>
    public ulong DeviceLocalHeap;
    /// <summary>Integrated GPU: "device-local" memory is system RAM shared with the CPU.</summary>
    public bool UnifiedMemory;
    private unsafe delegate* unmanaged[Cdecl]<IntPtr, uint, IntPtr, uint, uint, Vk.VkWriteDescriptorSet*, void> _pushDesc;
    public double TimestampPeriodNs = 1;
    private IntPtr _commandPool;
    private IntPtr _descPool;

    // live vkAllocateMemory bytes across the process (diagnostic — buffers are pooled)
    private static long _liveBytes, _peakBytes;
    public static long LiveDeviceBytes => Interlocked.Read(ref _liveBytes);
    public static long PeakDeviceBytes => Interlocked.Read(ref _peakBytes);
    public static void ResetPeakDeviceBytes() => Interlocked.Exchange(ref _peakBytes, _liveBytes);
    private static void TrackAlloc(long bytes)
    {
        long now = Interlocked.Add(ref _liveBytes, bytes), p;
        while ((p = Interlocked.Read(ref _peakBytes)) < now)
            Interlocked.CompareExchange(ref _peakBytes, now, p);
    }
    internal static void TrackFree(long bytes) => Interlocked.Add(ref _liveBytes, -bytes);
    /// <summary>Guards the device-level command pool / descriptor pool (uploads).</summary>
    public readonly object Sync = new();
    /// <summary>vkQueueSubmit on the single compute queue is externally synchronized.</summary>
    public readonly object QueueLock = new();
    private readonly Dictionary<(string name, int bindings, int pcBytes, uint sg), VkPipeline> _pipes = new();

    /// <summary>Process-wide pipeline cache: every graph/session shares one
    /// VkPipeline per (shader, binding-count, push-const-size). Set layouts carry
    /// PUSH_DESCRIPTOR_BIT when the extension is available.</summary>
    public VkPipeline GetPipeline(string name, int bindings, int pcBytes, uint requiredSubgroupSize = 0)
    {
        var key = (name, bindings, pcBytes, requiredSubgroupSize);
        lock (_pipes)
        {
            if (_pipes.TryGetValue(key, out VkPipeline? p)) return p;
            p = NewPipeline(NewShaderModule(VkShaders.Load(name)), bindings, pcBytes, PushDescriptors, requiredSubgroupSize);
            p.Name = name;
            p.Bindings = bindings;
            _pipes[key] = p;
            return p;
        }
    }

    /// <summary>Pipeline of a conv_cm variant ("conv_cm", "conv_cm_n64", ...; fp16 storage): the
    /// LITE build on <see cref="CoopLite"/> devices (or when <paramref name="lite"/> forces it),
    /// the SG16 build (always LITE-style) on <see cref="CoopM8"/> devices, with the subgroup pinned
    /// to <see cref="CoopSubgroupSize"/>. Every variant keeps BM = 128 rows per workgroup.</summary>
    public VkPipeline GetCoopPipeline(string variant, int bindings, int pcBytes, bool? lite = null)
    {
        if (!variant.StartsWith("conv_cm", StringComparison.Ordinal)) throw new ArgumentException(variant);
        string suffix = variant["conv_cm".Length..];
        string name = CoopM8 ? "conv_cm_sg16" + suffix : (lite ?? CoopLite) ? "conv_cm_l" + suffix : variant;
        return GetPipeline(name + "_h", bindings, pcBytes, CoopSubgroupSize);
    }

    public static VkDevice Create(uint deviceIndex = 0)
    {
        Vk.RegisterResolver();
        var d = new VkDevice();

        Vk.VkApplicationInfo app = new()
        {
            SType = VkConst.StApplicationInfo,
            AppVersion = 1, EngineVersion = 1, ApiVersion = (1u << 22) | (1u << 12), // Vulkan 1.1
        };
        Vk.VkInstanceCreateInfo ici = new() { SType = VkConst.StInstanceCreateInfo, PAppInfo = &app };
        Vk.Check(Vk.vkCreateInstance(&ici, null, out d.Instance), "vkCreateInstance");

        uint ndev = 0;
        Vk.Check(Vk.vkEnumeratePhysicalDevices(d.Instance, &ndev, null), "enum physical devices");
        if (ndev == 0) throw new PlatformNotSupportedException("Vulkan: no physical devices (no Vulkan-capable GPU/driver)");
        IntPtr* devs = stackalloc IntPtr[(int)ndev];
        Vk.Check(Vk.vkEnumeratePhysicalDevices(d.Instance, &ndev, devs), "enum physical devices");

        // Prefer discrete GPU; honor deviceIndex as "the nth discrete first, else nth overall".
        int chosen = -1;
        for (int pass = 0; pass < 2 && chosen < 0; pass++)
        {
            uint seen = 0;
            for (int i = 0; i < (int)ndev; i++)
            {
                Vk.VkPhysicalDeviceProperties p;
                Vk.vkGetPhysicalDeviceProperties(devs[i], &p);
                bool want = pass == 0 ? p.DeviceType == VkConst.PhysDeviceDiscrete : p.DeviceType != VkConst.PhysDeviceCpu;
                if (!want) continue;
                if (seen++ == deviceIndex) { chosen = i; break; }
            }
        }
        if (chosen < 0) throw new PlatformNotSupportedException($"Vulkan: device index {deviceIndex} out of range ({ndev} devices)");
        d.PhysDevice = devs[chosen];

        Vk.VkPhysicalDeviceProperties props;
        Vk.vkGetPhysicalDeviceProperties(d.PhysDevice, &props);
        d.DeviceName = Marshal.PtrToStringAnsi((IntPtr)props.DeviceName) ?? "?";
        d.VendorId = props.VendorID;
        d.TimestampPeriodNs = props.TimestampPeriodNs;
        d.MaxSharedMemory = props.MaxComputeSharedMemorySize;
        d.MaxStorageRange = props.MaxStorageBufferRange;

        uint nqf = 0;
        Vk.vkGetPhysicalDeviceQueueFamilyProperties(d.PhysDevice, &nqf, null);
        Vk.VkQueueFamilyProperties* qf = stackalloc Vk.VkQueueFamilyProperties[(int)nqf];
        Vk.vkGetPhysicalDeviceQueueFamilyProperties(d.PhysDevice, &nqf, qf);
        if (Environment.GetEnvironmentVariable("GCDRNET_VK_DBG") == "1")
            for (uint i = 0; i < nqf; i++)
                Console.Error.WriteLine($"[vk] qf{i}: flags={qf[i].QueueFlags:x} count={qf[i].QueueCount} tsbits={qf[i].TimestampValidBits}");
        // Prefer a compute-only family (no GRAPHICS bit) for async-compute —
        // but only if it supports timestamps; otherwise fall back to graphics+compute.
        d.QueueFamily = uint.MaxValue;
        for (uint i = 0; i < nqf; i++)
            if ((qf[i].QueueFlags & VkConst.QueueCompute) != 0 && (qf[i].QueueFlags & 1u) == 0
                && qf[i].TimestampValidBits > 0)
            { d.QueueFamily = i; break; }
        for (uint i = 0; i < nqf && d.QueueFamily == uint.MaxValue; i++)
            if ((qf[i].QueueFlags & VkConst.QueueCompute) != 0 && qf[i].TimestampValidBits > 0)
            { d.QueueFamily = i; break; }
        for (uint i = 0; i < nqf && d.QueueFamily == uint.MaxValue; i++)
            if ((qf[i].QueueFlags & VkConst.QueueCompute) != 0) { d.QueueFamily = i; break; }
        if (d.QueueFamily == uint.MaxValue)
            throw new PlatformNotSupportedException("Vulkan: no compute queue family");

        // Extension probe: VK_KHR_push_descriptor (inline descriptor writes,
        // skips per-dispatch descriptor-set binds).
        uint next = 0;
        Vk.vkEnumerateDeviceExtensionProperties(d.PhysDevice, null, &next, null);
        Vk.VkExtensionProperties* exts = stackalloc Vk.VkExtensionProperties[(int)next];
        Vk.vkEnumerateDeviceExtensionProperties(d.PhysDevice, null, &next, exts);
        byte* wantPush = stackalloc byte[] { (byte)'V', (byte)'K', (byte)'_', (byte)'K', (byte)'H', (byte)'R',
            (byte)'_', (byte)'p', (byte)'u', (byte)'s', (byte)'h', (byte)'_', (byte)'d', (byte)'e', (byte)'s', (byte)'c',
            (byte)'r', (byte)'i', (byte)'p', (byte)'t', (byte)'o', (byte)'r', 0 };
        bool hasPush = false, hasCoop = false, hasF16Int8 = false, hasVmm = false, hasSgc = false;
        for (int i = 0; i < (int)next; i++)
        {
            string en = new((sbyte*)exts[i].ExtensionName);
            if (en == "VK_KHR_push_descriptor") hasPush = true;
            else if (en == "VK_KHR_cooperative_matrix") hasCoop = true;
            else if (en == "VK_KHR_shader_float16_int8") hasF16Int8 = true;
            else if (en == "VK_KHR_vulkan_memory_model") hasVmm = true;
            else if (en == "VK_EXT_subgroup_size_control") hasSgc = true;
        }
        if (Environment.GetEnvironmentVariable("GCDRNET_VK_NOPUSH") == "1") hasPush = false;

        float prio = 1f;
        Vk.VkDeviceQueueCreateInfo qci = new()
        {
            SType = VkConst.StDeviceQueueCreateInfo,
            QueueFamilyIndex = d.QueueFamily, QueueCount = 1, PQueuePriorities = &prio,
        };
        Vk.VkPhysicalDeviceSubgroupProperties sgP = new() { SType = VkConst.StPhysicalDeviceSubgroupProperties };
        Vk.VkPhysicalDeviceSubgroupSizeControlProperties sgcP = new() { SType = VkConst.StPhysicalDeviceSubgroupSizeControlPropertiesEXT };
        if (hasSgc) sgP.PNext = &sgcP;
        Vk.VkPhysicalDeviceProperties2 p2 = new() { SType = VkConst.StPhysicalDeviceProperties2, PNext = &sgP };
        Vk.vkGetPhysicalDeviceProperties2(d.PhysDevice, &p2);
        d.SubgroupSize = sgP.SubgroupSize;
        bool sgcCompute = false;
        if (hasSgc)
        {
            d.SubgroupMin = sgcP.MinSubgroupSize;
            d.SubgroupMax = sgcP.MaxSubgroupSize;
            sgcCompute = (sgcP.RequiredSubgroupSizeStages & VkConst.StageComputeShader) != 0;
        }

        // Query storageBuffer16BitAccess (VK 1.1 core) for fp16 activation storage.
        Vk.VkPhysicalDevice16BitStorageFeatures feat16 = new()
        {
            SType = VkConst.StPhysicalDevice16BitStorageFeatures,
        };
        Vk.VkPhysicalDeviceFeatures2 feats2 = new()
        {
            SType = VkConst.StPhysicalDeviceFeatures2,
            PNext = &feat16,
        };
        Vk.vkGetPhysicalDeviceFeatures2(d.PhysDevice, &feats2);
        d.StorageBuffer16 = feat16.StorageBuffer16BitAccess != 0;

        // tensor-core path: coopmat + shaderFloat16 on top of 16-bit storage, 32-lane subgroups;
        // coopmat SPIR-V uses the Vulkan memory model, so that feature is required as well
        Vk.VkPhysicalDeviceShaderFloat16Int8Features f16q = new() { SType = VkConst.StPhysicalDeviceShaderFloat16Int8Features };
        Vk.VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopQ = new() { SType = VkConst.StPhysicalDeviceCooperativeMatrixFeaturesKHR };
        Vk.VkPhysicalDeviceVulkanMemoryModelFeatures vmmQ = new() { SType = VkConst.StPhysicalDeviceVulkanMemoryModelFeatures };
        Vk.VkPhysicalDeviceSubgroupSizeControlFeaturesEXT sgcQ = new() { SType = VkConst.StPhysicalDeviceSubgroupSizeControlFeaturesEXT };
        // a default other than 32 lanes (RDNA: wave64) needs the compute subgroup pinned to 32
        bool pinSg32 = d.SubgroupSize != 32 && hasSgc && sgcCompute && d.SubgroupMin <= 32 && d.SubgroupMax >= 32
            && Environment.GetEnvironmentVariable("GCDRNET_VK_SG32") != "0";
        // the SG16 build (8x16x16, Intel Xe2 XMX) needs 16 lanes in the pinnable compute range
        bool canSg16 = hasSgc && sgcCompute && d.SubgroupMin <= 16 && d.SubgroupMax >= 16
            && Environment.GetEnvironmentVariable("GCDRNET_VK_SG16") != "0";
        bool wantCm = hasCoop && hasF16Int8 && hasVmm && d.StorageBuffer16 && (d.SubgroupSize == 32 || pinSg32 || canSg16)
            && Environment.GetEnvironmentVariable("GCDRNET_VK_COOPMAT") != "0";
        bool pinSg16 = false;
        if (wantCm)
        {
            f16q.PNext = &coopQ;
            coopQ.PNext = &vmmQ;
            if (pinSg32 || canSg16) vmmQ.PNext = &sgcQ;
            Vk.VkPhysicalDeviceFeatures2 fq = new() { SType = VkConst.StPhysicalDeviceFeatures2, PNext = &f16q };
            Vk.vkGetPhysicalDeviceFeatures2(d.PhysDevice, &fq);
            bool feats = f16q.ShaderFloat16 != 0 && coopQ.CooperativeMatrix != 0 && vmmQ.VulkanMemoryModel != 0;
            CoopShapes(d, out bool m16, out bool m8);
            // 16x16x16 on 32 lanes first (NVIDIA, RDNA pinned to 32); 8x16x16 on 16 lanes only without it
            bool cm32 = feats && (d.SubgroupSize == 32 || pinSg32) && (!pinSg32 || sgcQ.SubgroupSizeControl != 0) && m16;
            pinSg16 = !cm32 && feats && canSg16 && sgcQ.SubgroupSizeControl != 0 && m8;
            wantCm = cm32 || pinSg16;
            pinSg32 &= cm32;
        }
        else pinSg32 = false;
        bool pinSg = pinSg32 || pinSg16;

        byte** extsToEnable = stackalloc byte*[5];
        uint nExt = 0;
        if (hasPush) extsToEnable[nExt++] = wantPush;
        byte* pCm = stackalloc byte[32], pF16 = stackalloc byte[32], pVmm = stackalloc byte[32], pSgc = stackalloc byte[32];
        System.Text.Encoding.ASCII.GetBytes("VK_KHR_cooperative_matrix\0", new Span<byte>(pCm, 32));
        System.Text.Encoding.ASCII.GetBytes("VK_KHR_shader_float16_int8\0", new Span<byte>(pF16, 32));
        System.Text.Encoding.ASCII.GetBytes("VK_KHR_vulkan_memory_model\0", new Span<byte>(pVmm, 32));
        System.Text.Encoding.ASCII.GetBytes("VK_EXT_subgroup_size_control\0", new Span<byte>(pSgc, 32));
        if (wantCm) { extsToEnable[nExt++] = pCm; extsToEnable[nExt++] = pF16; extsToEnable[nExt++] = pVmm; }
        if (pinSg) extsToEnable[nExt++] = pSgc;
        Vk.VkPhysicalDeviceShaderFloat16Int8Features f16en = new()
        {
            SType = VkConst.StPhysicalDeviceShaderFloat16Int8Features, ShaderFloat16 = 1,
        };
        Vk.VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopEn = new()
        {
            SType = VkConst.StPhysicalDeviceCooperativeMatrixFeaturesKHR, CooperativeMatrix = 1,
        };
        Vk.VkPhysicalDeviceVulkanMemoryModelFeatures vmmEn = new()
        {
            SType = VkConst.StPhysicalDeviceVulkanMemoryModelFeatures, VulkanMemoryModel = 1,
        };
        Vk.VkPhysicalDeviceSubgroupSizeControlFeaturesEXT sgcEn = new()
        {
            SType = VkConst.StPhysicalDeviceSubgroupSizeControlFeaturesEXT, SubgroupSizeControl = 1,
            ComputeFullSubgroups = sgcQ.ComputeFullSubgroups,
        };
        if (pinSg) vmmEn.PNext = &sgcEn;
        coopEn.PNext = &vmmEn;
        f16en.PNext = &coopEn;
        feat16.PNext = wantCm ? &f16en : null;

        // To enable 16-bit storage we chain feat16 into VkDeviceCreateInfo.pNext
        // (VK 1.1 core: no extension needed). PEnabledFeatures must be null when
        // using Features2 pNext; we inline a zero VkPhysicalDeviceFeatures in feat2.
        Vk.VkPhysicalDeviceFeatures2 feat2enable = new()
        {
            SType = VkConst.StPhysicalDeviceFeatures2,
            PNext = d.StorageBuffer16 ? &feat16 : null,
        };
        Vk.VkDeviceCreateInfo dci = new()
        {
            SType = VkConst.StDeviceCreateInfo,
            QueueCreateInfoCount = 1, PQueueCreateInfos = &qci,
            PEnabledFeatures = null,
            PNext = &feat2enable,
            EnabledExtensionCount = nExt,
            PpEnabledExtensionNames = nExt > 0 ? extsToEnable : null,
        };
        Vk.Check(Vk.vkCreateDevice(d.PhysDevice, &dci, null, out d.Device), "vkCreateDevice");
        d.CoopMat16 = wantCm;
        d.CoopM8 = pinSg16;
        d.CoopSubgroupSize = pinSg32 ? 32u : pinSg16 ? 16u : 0u;
        d._fullSubgroups = pinSg && sgcQ.ComputeFullSubgroups != 0;
        d.CoopLite = wantCm && !pinSg16 && ((hasSgc ? d.SubgroupMax : d.SubgroupSize) >= 64 || d.MaxSharedMemory < 48 * 1024)
            && Environment.GetEnvironmentVariable("GCDRNET_VK_LITE") != "0";
        Vk.vkGetDeviceQueue(d.Device, d.QueueFamily, 0, out d.Queue);

        if (hasPush)
        {
            byte* fnName = stackalloc byte[] { (byte)'v', (byte)'k', (byte)'C', (byte)'m', (byte)'d',
                (byte)'P', (byte)'u', (byte)'s', (byte)'h', (byte)'D', (byte)'e', (byte)'s', (byte)'c',
                (byte)'r', (byte)'i', (byte)'p', (byte)'t', (byte)'o', (byte)'r', (byte)'S', (byte)'e',
                (byte)'t', (byte)'K', (byte)'H', (byte)'R', 0 };
            IntPtr fn = Vk.vkGetDeviceProcAddr(d.Device, fnName);
            if (fn != IntPtr.Zero)
            {
                d.PushDescriptors = true;
                d._pushDesc = (delegate* unmanaged[Cdecl]<IntPtr, uint, IntPtr, uint, uint, Vk.VkWriteDescriptorSet*, void>)fn;
            }
        }

        fixed (Vk.VkPhysicalDeviceMemoryProperties* mp = &d.MemProps)
            Vk.vkGetPhysicalDeviceMemoryProperties(d.PhysDevice, mp);
        for (int i = 0; i < (int)d.MemProps.MemoryTypeCount; i++)
        {
            uint f = d.MemProps.TypeAt(i).PropertyFlags;
            if ((f & (VkConst.MemDeviceLocal | VkConst.MemHostVisible | VkConst.MemHostCoherent)) == (VkConst.MemDeviceLocal | VkConst.MemHostVisible | VkConst.MemHostCoherent))
            {
                d.CoherentDeviceLocal = true;
                break;
            }
        }
        d.UnifiedMemory = props.DeviceType == VkConst.PhysDeviceIntegrated;
        d.Vec4Eltwise = (d.UnifiedMemory || d.VendorId == VendorIntel) && Environment.GetEnvironmentVariable("GCDRNET_VK_VEC4") != "0";
        d.ConvPoolK3 = d.VendorId == VendorIntel && Environment.GetEnvironmentVariable("GCDRNET_VK_CONVPOOL3") != "0";
        for (int i = 0; i < (int)d.MemProps.MemoryHeapCount; i++)
        {
            var hp = d.MemProps.HeapAt(i);
            if ((hp.Flags & 1u) != 0) d.DeviceLocalHeap = Math.Max(d.DeviceLocalHeap, hp.Size);   // HEAP_DEVICE_LOCAL
        }

        Vk.VkCommandPoolCreateInfo cpci = new()
        {
            SType = VkConst.StCommandPoolCreateInfo,
            Flags = VkConst.CmdPoolResetCommandBuffer,
            QueueFamilyIndex = d.QueueFamily,
        };
        Vk.Check(Vk.vkCreateCommandPool(d.Device, &cpci, null, out d._commandPool), "vkCreateCommandPool");

        Vk.VkDescriptorPoolSize psz = new() { Type = VkConst.DescStorageBuffer, DescriptorCount = 1 << 20 };
        Vk.VkDescriptorPoolCreateInfo dpci = new()
        {
            SType = VkConst.StDescriptorPoolCreateInfo,
            Flags = 1, MaxSets = 1 << 16, PoolSizeCount = 1, PPoolSizes = &psz,
        };
        Vk.Check(Vk.vkCreateDescriptorPool(d.Device, &dpci, null, out d._descPool), "vkCreateDescriptorPool");

        // never trust a pinned subgroup size blindly: the coopmat lane mapping is hard-wired to
        // the pinned width (32, or 16 for SG16), so anything else would silently compute garbage
        if (pinSg && !d.ProbeSubgroup(d.CoopSubgroupSize))
        {
            d.CoopMat16 = false;
            d.CoopM8 = false;
            d.CoopLite = false;
            d.CoopSubgroupSize = 0;
        }
        if (Environment.GetEnvironmentVariable("GCDRNET_VK_VERBOSE") == "1")
        {
            Console.Error.WriteLine($"[vk] device={d.DeviceName} type={props.DeviceType} vendor=0x{d.VendorId:x4} qfamily={d.QueueFamily} " +
                $"sg={d.SubgroupSize} sgRange={d.SubgroupMin}-{d.SubgroupMax} sgcCompute={sgcCompute} push={d.PushDescriptors} " +
                $"coopmat={d.CoopMat16} coopSg={(d.CoopSubgroupSize == 0 ? "default" : d.CoopSubgroupSize.ToString())} " +
                $"m8={d.CoopM8} fullSg={d._fullSubgroups} lite={d.CoopLite} vec4={d.Vec4Eltwise} cp3={d.ConvPoolK3} lds={d.MaxSharedMemory} sb16={d.StorageBuffer16}");
            for (uint i = 0; i < nqf; i++)
                Console.Error.WriteLine($"[vk] qf{i}: flags=0x{qf[i].QueueFlags:x} count={qf[i].QueueCount} tsbits={qf[i].TimestampValidBits}"
                    + (i == d.QueueFamily ? " <- selected" : ""));
            if (hasCoop)
                foreach (var cp in CoopProps(d))
                    Console.Error.WriteLine($"[vk] coopmat {cp.MSize}x{cp.NSize}x{cp.KSize} a={cp.AType} b={cp.BType} c={cp.CType} r={cp.ResultType} scope={cp.Scope}");
            for (int i = 0; i < (int)d.MemProps.MemoryHeapCount; i++)
            {
                var hp = d.MemProps.HeapAt(i);
                Console.Error.WriteLine($"[vk] heap{i}: {hp.Size >> 20} MB flags={hp.Flags:x}");
            }
            for (int i = 0; i < (int)d.MemProps.MemoryTypeCount; i++)
                Console.Error.WriteLine($"[vk] type{i}: heap{d.MemProps.TypeAt(i).HeapIndex} flags={d.MemProps.TypeAt(i).PropertyFlags:x}");
        }
        return d;
    }

    private bool _fullSubgroups;

    /// <summary>Runs sgprobe.comp (256 threads) with requiredSubgroupSize = <paramref name="lanes"/> and
    /// checks every invocation reports <paramref name="lanes"/>-lane subgroups, 256 / lanes per workgroup.</summary>
    private bool ProbeSubgroup(uint lanes)
    {
        VkPipeline? pipe = null;
        VkBuffer? buf = null;
        uint count = 256 / lanes;
        try
        {
            pipe = NewPipeline(NewShaderModule(VkShaders.Load("sgprobe")), 1, 0, PushDescriptors, lanes);
            buf = NewStorageBuffer(256 * 4, hostVisible: true, preferHost: true);
            IntPtr set = IntPtr.Zero, pool = IntPtr.Zero;
            lock (Sync)
            {
                IntPtr cmd = NewCommandBuffer();
                IntPtr fence = NewFence();
                var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
                Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
                Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, pipe.Pipeline);
                Vk.VkDescriptorBufferInfo bi = new() { Buffer = buf.Buffer, Offset = 0, Range = 256 * 4 };
                Vk.VkWriteDescriptorSet w = new()
                {
                    SType = VkConst.StWriteDescriptorSet, DstBinding = 0,
                    DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer, PBufferInfo = &bi,
                };
                if (PushDescriptors) CmdPushDescriptors(cmd, pipe.Layout, &w, 1);
                else
                {
                    pool = NewDescriptorPool(1);
                    set = AllocateDescriptorSet(pool, pipe.SetLayout);
                    BindBuffer(set, 0, buf, 0, 256 * 4);
                    Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, pipe.Layout, 0, 1, &set, 0, null);
                }
                Vk.vkCmdDispatch(cmd, 1, 1, 1);
                Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
                Submit(cmd, fence);
                WaitFence(fence);
                Vk.vkDestroyFence(Device, fence, null);
                FreeCommandBuffer(cmd);
                if (pool != IntPtr.Zero) Vk.vkDestroyDescriptorPool(Device, pool, null);
            }
            uint* r = (uint*)buf.Map();
            buf.Invalidate(0, 256 * 4);
            bool ok = true;
            for (int i = 0; i < 256; i++)
                ok &= (r[i] & 0xff) == lanes && ((r[i] >> 8) & 0xff) == count && ((r[i] >> 16) & 0xff) < count && (r[i] >> 24) < lanes;
            buf.Unmap();
            return ok;
        }
        catch (InvalidOperationException) { return false; }
        finally
        {
            buf?.Free();
            if (pipe is not null)
            {
                Vk.vkDestroyPipeline(Device, pipe.Pipeline, null);
                Vk.vkDestroyPipelineLayout(Device, pipe.Layout, null);
                Vk.vkDestroyDescriptorSetLayout(Device, pipe.SetLayout, null);
            }
        }
    }

    /// <summary>Which fp16 x fp16 -> fp32 (C and result) subgroup coopmat shapes the device offers:
    /// 16x16x16 (the default conv_cm builds) and 8x16x16 (SG16).</summary>
    private static void CoopShapes(VkDevice d, out bool m16, out bool m8)
    {
        m16 = m8 = false;
        // component types: 0 = float16, 1 = float32; scope 3 = subgroup
        foreach (var p in CoopProps(d))
            if (p.NSize == 16 && p.KSize == 16 && p.AType == 0 && p.BType == 0
                && p.CType == 1 && p.ResultType == 1 && p.Scope == 3)
            {
                m16 |= p.MSize == 16;
                m8 |= p.MSize == 8;
            }
    }

    /// <summary>vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR (empty when unavailable).</summary>
    private static Vk.VkCooperativeMatrixPropertiesKHR[] CoopProps(VkDevice d)
    {
        byte[] name = System.Text.Encoding.ASCII.GetBytes("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR\0");
        IntPtr fn;
        fixed (byte* pn = name) fn = Vk.vkGetInstanceProcAddr(d.Instance, pn);
        if (fn == IntPtr.Zero) return [];
        var query = (delegate* unmanaged[Cdecl]<IntPtr, uint*, Vk.VkCooperativeMatrixPropertiesKHR*, VkResult>)fn;
        uint np = 0;
        if (query(d.PhysDevice, &np, null) != VkResult.Success || np == 0) return [];
        var props = new Vk.VkCooperativeMatrixPropertiesKHR[np];
        for (int i = 0; i < props.Length; i++) props[i].SType = VkConst.StCooperativeMatrixPropertiesKHR;
        fixed (Vk.VkCooperativeMatrixPropertiesKHR* pp = props)
            if (query(d.PhysDevice, &np, pp) != VkResult.Success) return [];
        return props;
    }

    /// <summary>Allocate a storage buffer. hostVisible=false prefers DEVICE_LOCAL;
    /// hostVisible=true prefers DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT (ReBAR zero-copy), falling
    /// back to HOST_VISIBLE|HOST_COHERENT. preferHost skips DEVICE_LOCAL types — for buffers the
    /// CPU reads (uncached WC reads from VRAM are ~15MB/s, catastrophic).</summary>
    public VkBuffer NewStorageBuffer(ulong bytes, bool hostVisible, bool preferHost = false)
    {
        var b = new VkBuffer { Dev = Device, Size = bytes };
        Vk.VkBufferCreateInfo bci = new()
        {
            SType = VkConst.StBufferCreateInfo,
            Size = bytes,
            Usage = VkConst.BufferUsageStorageBuffer | VkConst.BufferUsageTransferSrc | VkConst.BufferUsageTransferDst,
            SharingMode = VkConst.SharingExclusive,
        };
        Vk.Check(Vk.vkCreateBuffer(Device, &bci, null, out b.Buffer), "vkCreateBuffer");
        Vk.VkMemoryRequirements req;
        Vk.vkGetBufferMemoryRequirements(Device, b.Buffer, &req);

        uint memType = uint.MaxValue;
        for (int i = 0; i < (int)MemProps.MemoryTypeCount; i++)
        {
            if ((req.MemoryTypeBits & (1u << i)) == 0) continue;
            uint f = MemProps.TypeAt(i).PropertyFlags;
            bool ok;
            if (!hostVisible)
                ok = (f & VkConst.MemDeviceLocal) != 0;
            else if (preferHost)
                ok = (f & (VkConst.MemHostVisible | VkConst.MemHostCoherent)) == (VkConst.MemHostVisible | VkConst.MemHostCoherent)
                     && (f & VkConst.MemDeviceLocal) == 0
                     && (f & VkConst.MemHostCached) != 0;
            else
                ok = (f & VkConst.MemHostVisible) != 0 && (f & VkConst.MemHostCoherent) != 0;
            if (ok && (memType == uint.MaxValue || (!preferHost && (f & VkConst.MemDeviceLocal) != 0)))
            {
                memType = (uint)i;
                if (preferHost || (f & VkConst.MemDeviceLocal) != 0) break;
            }
        }
        if (memType == uint.MaxValue && preferHost)
            for (int i = 0; i < (int)MemProps.MemoryTypeCount && memType == uint.MaxValue; i++)
            {
                if ((req.MemoryTypeBits & (1u << i)) == 0) continue;
                uint f = MemProps.TypeAt(i).PropertyFlags;
                if ((f & (VkConst.MemHostVisible | VkConst.MemHostCoherent))
                    == (VkConst.MemHostVisible | VkConst.MemHostCoherent)
                    && (f & VkConst.MemDeviceLocal) == 0)
                    memType = (uint)i;
            }
        if (memType == uint.MaxValue && preferHost)
        {
            for (int i = 0; i < (int)MemProps.MemoryTypeCount && memType == uint.MaxValue; i++)
            {
                if ((req.MemoryTypeBits & (1u << i)) == 0) continue;
                uint f = MemProps.TypeAt(i).PropertyFlags;
                if ((f & (VkConst.MemHostVisible | VkConst.MemHostCoherent | VkConst.MemHostCached))
                    == (VkConst.MemHostVisible | VkConst.MemHostCoherent | VkConst.MemHostCached))
                    memType = (uint)i;
            }
            for (int i = 0; i < (int)MemProps.MemoryTypeCount && memType == uint.MaxValue; i++)
            {
                if ((req.MemoryTypeBits & (1u << i)) == 0) continue;
                uint f = MemProps.TypeAt(i).PropertyFlags;
                if ((f & (VkConst.MemHostVisible | VkConst.MemHostCoherent))
                    == (VkConst.MemHostVisible | VkConst.MemHostCoherent))
                    memType = (uint)i;
            }
        }
        if (memType == uint.MaxValue)
            throw new PlatformNotSupportedException($"Vulkan: no suitable memory type for {bytes}-byte buffer (hostVisible={hostVisible})");

        Vk.VkMemoryAllocateInfo mai = new()
        {
            SType = VkConst.StMemoryAllocateInfo,
            AllocationSize = req.Size, MemoryTypeIndex = memType,
        };
        Vk.Check(Vk.vkAllocateMemory(Device, &mai, null, out b.Memory), "vkAllocateMemory");
        b.AllocSize = req.Size;
        TrackAlloc((long)req.Size);
        Vk.Check(Vk.vkBindBufferMemory(Device, b.Buffer, b.Memory, 0), "vkBindBufferMemory");
        b.Flags = MemProps.TypeAt((int)memType).PropertyFlags;
        return b;
    }

    public IntPtr NewShaderModule(byte[] spirv)
    {
        if ((spirv.Length & 3) != 0) throw new ArgumentException("SPIR-V size must be a multiple of 4");
        fixed (byte* p = spirv)
        {
            Vk.VkShaderModuleCreateInfo ci = new()
            {
                SType = VkConst.StShaderModuleCreateInfo,
                CodeSize = (nuint)spirv.Length, PCode = (uint*)p,
            };
            Vk.Check(Vk.vkCreateShaderModule(Device, &ci, null, out IntPtr m), "vkCreateShaderModule");
            return m;
        }
    }

    /// <summary>One set layout with `bindings` storage buffers (0..bindings-1) + push constants.
    /// requiredSubgroupSize != 0 pins the compute subgroup width (VK_EXT_subgroup_size_control).</summary>
    public VkPipeline NewPipeline(IntPtr shaderModule, int bindings, int pushConstBytes, bool pushLayout,
        uint requiredSubgroupSize = 0)
    {
        var dslb = stackalloc Vk.VkDescriptorSetLayoutBinding[bindings];
        for (int i = 0; i < bindings; i++)
            dslb[i] = new Vk.VkDescriptorSetLayoutBinding
            {
                Binding = (uint)i, DescriptorType = VkConst.DescStorageBuffer,
                DescriptorCount = 1, StageFlags = VkConst.StageComputeShader,
            };
        IntPtr setLayout;
        Vk.VkDescriptorSetLayoutCreateInfo dslci = new()
        {
            SType = VkConst.StDescriptorSetLayoutCreateInfo,
            BindingCount = (uint)bindings, PBindings = dslb,
            Flags = pushLayout ? 1u : 0u,   // VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR
        };
        Vk.Check(Vk.vkCreateDescriptorSetLayout(Device, &dslci, null, out setLayout), "vkCreateDescriptorSetLayout");

        Vk.VkPushConstantRange pcr = new()
        {
            StageFlags = VkConst.StageComputeShader, Offset = 0, Size = (uint)pushConstBytes,
        };
        IntPtr layout;
        Vk.VkPipelineLayoutCreateInfo plci = new()
        {
            SType = VkConst.StPipelineLayoutCreateInfo,
            SetLayoutCount = 1, PSetLayouts = &setLayout,
            PushConstantRangeCount = pushConstBytes > 0 ? 1u : 0u,
            PPushConstantRanges = pushConstBytes > 0 ? &pcr : null,
        };
        Vk.Check(Vk.vkCreatePipelineLayout(Device, &plci, null, out layout), "vkCreatePipelineLayout");

        byte* name = stackalloc byte[5] { (byte)'m', (byte)'a', (byte)'i', (byte)'n', 0 };
        Vk.VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sgc = new()
        {
            SType = VkConst.StPipelineShaderStageRequiredSubgroupSizeCreateInfo,
            RequiredSubgroupSize = requiredSubgroupSize,
        };
        Vk.VkComputePipelineCreateInfo cpci = new()
        {
            SType = VkConst.StComputePipelineCreateInfo,
            Stage = new Vk.VkPipelineShaderStageCreateInfo
            {
                SType = VkConst.StPipelineShaderStageCreateInfo,
                Stage = VkConst.StageComputeShader, Module = shaderModule, PName = name,
                PNext = requiredSubgroupSize != 0 ? &sgc : null,
                // REQUIRE_FULL_SUBGROUPS: every subgroup of a pinned pipeline has all its lanes
                Flags = requiredSubgroupSize != 0 && _fullSubgroups ? 2u : 0u,
            },
            Layout = layout,
        };
        IntPtr pipeline;
        Vk.Check(Vk.vkCreateComputePipelines(Device, IntPtr.Zero, 1, &cpci, null, &pipeline), "vkCreateComputePipelines");
        return new VkPipeline { Pipeline = pipeline, Layout = layout, SetLayout = setLayout, PushLayout = pushLayout };
    }

    public IntPtr NewDescriptorPool(uint maxSets)
    {
        Vk.VkDescriptorPoolSize psz = new() { Type = VkConst.DescStorageBuffer, DescriptorCount = maxSets * 16 };
        Vk.VkDescriptorPoolCreateInfo dpci = new()
        {
            SType = VkConst.StDescriptorPoolCreateInfo,
            Flags = 0, MaxSets = maxSets, PoolSizeCount = 1, PPoolSizes = &psz,
        };
        Vk.Check(Vk.vkCreateDescriptorPool(Device, &dpci, null, out IntPtr pool), "vkCreateDescriptorPool");
        return pool;
    }

    public IntPtr AllocateDescriptorSet(IntPtr pool, IntPtr setLayout)
    {
        IntPtr set;
        Vk.VkDescriptorSetAllocateInfo ai = new()
        {
            SType = VkConst.StDescriptorSetAllocateInfo,
            DescriptorPool = pool, DescriptorSetCount = 1, PSetLayouts = &setLayout,
        };
        Vk.Check(Vk.vkAllocateDescriptorSets(Device, &ai, &set), "vkAllocateDescriptorSets");
        return set;
    }

    public IntPtr NewDescriptorSet(IntPtr setLayout)
    {
        IntPtr set;
        Vk.VkDescriptorSetAllocateInfo ai = new()
        {
            SType = VkConst.StDescriptorSetAllocateInfo,
            DescriptorPool = _descPool, DescriptorSetCount = 1, PSetLayouts = &setLayout,
        };
        Vk.Check(Vk.vkAllocateDescriptorSets(Device, &ai, &set), "vkAllocateDescriptorSets");
        return set;
    }

    /// <summary>VK_KHR_push_descriptor: write bindings inline into the command buffer
    /// (DstSet in each write is ignored).</summary>
    public void CmdPushDescriptors(IntPtr cmd, IntPtr layout, Vk.VkWriteDescriptorSet* writes, uint count)
        => _pushDesc(cmd, VkConst.BindPointCompute, layout, 0, count, writes);

    /// <summary>Record the standard write->read barrier between two compute dispatches.</summary>
    public void CmdComputeBarrier(IntPtr cmd)
    {
        Vk.VkMemoryBarrier mb = new()
        {
            SType = VkConst.StMemoryBarrier,
            SrcAccessMask = VkConst.AccessShaderWrite, DstAccessMask = VkConst.AccessShaderRead,
        };
        Vk.vkCmdPipelineBarrier(cmd, VkConst.PipelineStageComputeShader, VkConst.PipelineStageComputeShader, 0,
            1, &mb, 0, null, 0, null);
    }

    /// <summary>Upload raw bytes into a buffer (host memcpy when mappable, else staging + copy cmd).</summary>
    public void Upload(VkBuffer buf, void* src, ulong bytes)
    {
        if ((buf.Flags & VkConst.MemHostVisible) != 0)
        {
            void* p = buf.Map();
            Buffer.MemoryCopy(src, p, bytes, bytes);
            buf.Flush(0, bytes);
            buf.Unmap();
            return;
        }
        VkBuffer staging = NewStorageBuffer(bytes, hostVisible: true);
        void* sp = staging.Map();
        Buffer.MemoryCopy(src, sp, bytes, bytes);
        staging.Flush(0, bytes);
        staging.Unmap();
        lock (Sync)
        {
            IntPtr cmd = NewCommandBuffer();
            IntPtr fence = NewFence();
            var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
            Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
            Vk.VkBufferCopy r = new() { Size = bytes };
            Vk.vkCmdCopyBuffer(cmd, staging.Buffer, buf.Buffer, 1, &r);
            Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
            Submit(cmd, fence);
            WaitFence(fence);
            Vk.vkDestroyFence(Device, fence, null);
            FreeCommandBuffer(cmd);
        }
        staging.Free();
    }

    /// <summary>Download raw bytes from a buffer into dst (host memcpy or staging).</summary>
    public void Download(VkBuffer buf, void* dst, ulong bytes, ulong srcOffset = 0)
    {
        if ((buf.Flags & VkConst.MemHostVisible) != 0)
        {
            void* p = buf.Map();
            buf.Invalidate(srcOffset, bytes);
            Buffer.MemoryCopy((byte*)p + srcOffset, dst, bytes, bytes);
            buf.Unmap();
            return;
        }
        VkBuffer staging = NewStorageBuffer(bytes, hostVisible: true, preferHost: true);
        lock (Sync)
        {
            IntPtr cmd = NewCommandBuffer();
            IntPtr fence = NewFence();
            var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
            Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
            Vk.VkBufferCopy r = new() { Size = bytes, SrcOffset = srcOffset };
            Vk.vkCmdCopyBuffer(cmd, buf.Buffer, staging.Buffer, 1, &r);
            Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
            Submit(cmd, fence);
            WaitFence(fence);
            Vk.vkDestroyFence(Device, fence, null);
            FreeCommandBuffer(cmd);
        }
        void* sp = staging.Map();
        staging.Invalidate(0, bytes);
        Buffer.MemoryCopy(sp, dst, bytes, bytes);
        staging.Unmap();
        staging.Free();
    }

    public void BindBuffer(IntPtr set, uint binding, VkBuffer buf, ulong offset, ulong size)
    {
        Vk.VkDescriptorBufferInfo bi = new() { Buffer = buf.Buffer, Offset = offset, Range = size };
        Vk.VkWriteDescriptorSet w = new()
        {
            SType = VkConst.StWriteDescriptorSet, DstSet = set, DstBinding = binding,
            DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer, PBufferInfo = &bi,
        };
        Vk.vkUpdateDescriptorSets(Device, 1, &w, 0, null);
    }

    public IntPtr NewCommandBuffer()
    {
        Vk.VkCommandBufferAllocateInfo ai = new()
        {
            SType = VkConst.StCommandBufferAllocateInfo,
            CommandPool = _commandPool, Level = VkConst.CmdBufferLevelPrimary, CommandBufferCount = 1,
        };
        IntPtr cmd;
        Vk.Check(Vk.vkAllocateCommandBuffers(Device, &ai, &cmd), "vkAllocateCommandBuffers");
        return cmd;
    }

    public void FreeCommandBuffer(IntPtr cmd)
    {
        if (cmd != IntPtr.Zero) Vk.vkFreeCommandBuffers(Device, _commandPool, 1, &cmd);
    }

    /// <summary>Private resettable command pool (command pools are externally synchronized:
    /// a worker that re-records on its own thread must not share the device pool).</summary>
    public IntPtr NewCommandPool()
    {
        Vk.VkCommandPoolCreateInfo ci = new()
        {
            SType = VkConst.StCommandPoolCreateInfo,
            Flags = VkConst.CmdPoolResetCommandBuffer,
            QueueFamilyIndex = QueueFamily,
        };
        Vk.Check(Vk.vkCreateCommandPool(Device, &ci, null, out IntPtr pool), "vkCreateCommandPool");
        return pool;
    }

    public IntPtr AllocateCommandBuffer(IntPtr pool)
    {
        Vk.VkCommandBufferAllocateInfo ai = new()
        {
            SType = VkConst.StCommandBufferAllocateInfo,
            CommandPool = pool, Level = VkConst.CmdBufferLevelPrimary, CommandBufferCount = 1,
        };
        IntPtr cmd;
        Vk.Check(Vk.vkAllocateCommandBuffers(Device, &ai, &cmd), "vkAllocateCommandBuffers");
        return cmd;
    }

    /// <summary>Destroys a private command pool (frees its command buffers).</summary>
    public void DestroyCommandPool(IntPtr pool)
    {
        if (pool != IntPtr.Zero) Vk.vkDestroyCommandPool(Device, pool, null);
    }

    public IntPtr NewFence(bool signaled = false)
    {
        Vk.VkFenceCreateInfo ci = new() { SType = VkConst.StFenceCreateInfo, Flags = signaled ? 1u : 0u };
        Vk.Check(Vk.vkCreateFence(Device, &ci, null, out IntPtr f), "vkCreateFence");
        return f;
    }

    public void Submit(IntPtr cmd, IntPtr fence)
    {
        Vk.VkSubmitInfo si = new()
        {
            SType = VkConst.StSubmitInfo,
            CommandBufferCount = 1, PCommandBuffers = &cmd,
        };
        VkResult r;
        lock (QueueLock) r = Vk.vkQueueSubmit(Queue, 1, &si, fence);
        Vk.Check(r, "vkQueueSubmit");
    }

    public void DestroyFence(IntPtr fence)
    {
        if (fence != IntPtr.Zero) Vk.vkDestroyFence(Device, fence, null);
    }

    public void WaitFence(IntPtr fence)
    {
        Vk.Check(Vk.vkWaitForFences(Device, 1, &fence, 1, ulong.MaxValue), "vkWaitForFences");
        Vk.Check(Vk.vkResetFences(Device, 1, &fence), "vkResetFences");
    }

    public void Dispose()
    {
        if (Device != IntPtr.Zero)
        {
            Vk.vkDeviceWaitIdle(Device);
            if (_descPool != IntPtr.Zero) { Vk.vkDestroyDescriptorPool(Device, _descPool, null); _descPool = IntPtr.Zero; }
            if (_commandPool != IntPtr.Zero) { Vk.vkDestroyCommandPool(Device, _commandPool, null); _commandPool = IntPtr.Zero; }
            lock (_pipes)
                foreach (VkPipeline p in _pipes.Values)
                {
                    Vk.vkDestroyPipeline(Device, p.Pipeline, null);
                    Vk.vkDestroyPipelineLayout(Device, p.Layout, null);
                    Vk.vkDestroyDescriptorSetLayout(Device, p.SetLayout, null);
                }
            Vk.vkDestroyDevice(Device, null); Device = IntPtr.Zero;
        }
        if (Instance != IntPtr.Zero) { Vk.vkDestroyInstance(Instance, null); Instance = IntPtr.Zero; }
    }
}

internal sealed class VkPipeline
{
    public IntPtr Pipeline, Layout, SetLayout;
    public string Name = "";
    public int Bindings;
    public bool PushLayout;
}

internal unsafe sealed class VkBuffer
{
    public IntPtr Dev, Buffer, Memory;
    public ulong Size, AllocSize;
    public uint Flags;

    public bool HostCoherent => (Flags & VkConst.MemHostCoherent) != 0;

    public void* Map()
    {
        void* p;
        Vk.Check(Vk.vkMapMemory(Dev, Memory, 0, Size, 0, &p), "vkMapMemory");
        return p;
    }
    public void Unmap() => Vk.vkUnmapMemory(Dev, Memory);

    public void Flush(ulong offset, ulong size)
    {
        if (HostCoherent) return;
        Vk.VkMappedMemoryRange r = new() { SType = VkConst.StMappedMemoryRange, Memory = Memory, Offset = offset, Size = size };
        Vk.vkFlushMappedMemoryRanges(Dev, 1, &r);
    }
    public void Invalidate(ulong offset, ulong size)
    {
        if (HostCoherent) return;
        Vk.VkMappedMemoryRange r = new() { SType = VkConst.StMappedMemoryRange, Memory = Memory, Offset = offset, Size = size };
        Vk.vkInvalidateMappedMemoryRanges(Dev, 1, &r);
    }

    public void Free()
    {
        if (Buffer != IntPtr.Zero) { Vk.vkDestroyBuffer(Dev, Buffer, null); Buffer = IntPtr.Zero; }
        if (Memory != IntPtr.Zero)
        {
            Vk.vkFreeMemory(Dev, Memory, null); Memory = IntPtr.Zero;
            VkDevice.TrackFree((long)AllocSize);
            AllocSize = 0;
        }
    }
}
