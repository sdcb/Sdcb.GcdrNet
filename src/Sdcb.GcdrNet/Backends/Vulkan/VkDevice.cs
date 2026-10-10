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
    public uint MaxSharedMemory;
    public ulong MaxStorageRange = uint.MaxValue;
    public bool PushDescriptors;
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
    private readonly Dictionary<(string name, int bindings, int pcBytes), VkPipeline> _pipes = new();

    /// <summary>Process-wide pipeline cache: every graph/session shares one
    /// VkPipeline per (shader, binding-count, push-const-size). Set layouts carry
    /// PUSH_DESCRIPTOR_BIT when the extension is available.</summary>
    public VkPipeline GetPipeline(string name, int bindings, int pcBytes)
    {
        var key = (name, bindings, pcBytes);
        lock (_pipes)
        {
            if (_pipes.TryGetValue(key, out VkPipeline? p)) return p;
            p = NewPipeline(NewShaderModule(VkShaders.Load(name)), bindings, pcBytes, PushDescriptors);
            p.Name = name;
            p.Bindings = bindings;
            _pipes[key] = p;
            return p;
        }
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
        bool hasPush = false;
        for (int i = 0; i < (int)next; i++)
        {
            string en = new((sbyte*)exts[i].ExtensionName);
            if (en == "VK_KHR_push_descriptor") hasPush = true;
        }
        if (Environment.GetEnvironmentVariable("GCDRNET_VK_NOPUSH") == "1") hasPush = false;

        float prio = 1f;
        Vk.VkDeviceQueueCreateInfo qci = new()
        {
            SType = VkConst.StDeviceQueueCreateInfo,
            QueueFamilyIndex = d.QueueFamily, QueueCount = 1, PQueuePriorities = &prio,
        };
        Vk.VkPhysicalDeviceSubgroupProperties sgP = new() { SType = VkConst.StPhysicalDeviceSubgroupProperties };
        Vk.VkPhysicalDeviceProperties2 p2 = new() { SType = VkConst.StPhysicalDeviceProperties2, PNext = &sgP };
        Vk.vkGetPhysicalDeviceProperties2(d.PhysDevice, &p2);
        d.SubgroupSize = sgP.SubgroupSize;

        Vk.VkPhysicalDeviceFeatures feats = new();
        byte** extsToEnable = stackalloc byte*[1];
        uint nExt = 0;
        if (hasPush) extsToEnable[nExt++] = wantPush;
        Vk.VkDeviceCreateInfo dci = new()
        {
            SType = VkConst.StDeviceCreateInfo,
            QueueCreateInfoCount = 1, PQueueCreateInfos = &qci,
            PEnabledFeatures = &feats,
            EnabledExtensionCount = nExt,
            PpEnabledExtensionNames = nExt > 0 ? extsToEnable : null,
        };
        Vk.Check(Vk.vkCreateDevice(d.PhysDevice, &dci, null, out d.Device), "vkCreateDevice");
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
        if (Environment.GetEnvironmentVariable("GCDRNET_VK_VERBOSE") == "1")
            Console.Error.WriteLine($"[vk] device={d.DeviceName} qfamily={d.QueueFamily} sg={d.SubgroupSize} push={d.PushDescriptors} lds={d.MaxSharedMemory}");

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
        return d;
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

    /// <summary>One set layout with `bindings` storage buffers (0..bindings-1) + push constants.</summary>
    public VkPipeline NewPipeline(IntPtr shaderModule, int bindings, int pushConstBytes, bool pushLayout)
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
        Vk.VkComputePipelineCreateInfo cpci = new()
        {
            SType = VkConst.StComputePipelineCreateInfo,
            Stage = new Vk.VkPipelineShaderStageCreateInfo
            {
                SType = VkConst.StPipelineShaderStageCreateInfo,
                Stage = VkConst.StageComputeShader, Module = shaderModule, PName = name,
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
