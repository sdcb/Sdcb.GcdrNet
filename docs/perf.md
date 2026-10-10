# 本机性能（`--corpus` harness，49 张）

口径与 [corpus.md](./corpus.md) 一致：`--corpus <backend>`，manifest NN 序，
01_color 作 warmup 不计时，其余 **49 张**逐张计 `Enhance` 端到端（解码在计时外）；
pass1 含每尺寸的冷启动开销，pass2 为同进程全量复跑（同时校验输出 hash 确定性）。

## 环境

| 项      | 值                                                   |
| ------- | ---------------------------------------------------- |
| CPU     | AMD Ryzen 7 5800X（8C/16T，AVX2+FMA，无 AVX-512）     |
| GPU     | NVIDIA GeForce RTX 3080 Ti（驱动 32.0.15.8180）       |
| 内存    | 64 GB                                                |
| OS      | Windows 11 专业工作站版                               |
| .NET    | SDK 11.0.100-rc.1，`net10.0`，Release                 |
| Commit  | CPU：`c2a19c3`（+ arena 修复）；Vulkan：跨尺寸复用之后      |

## 命令

```powershell
dotnet build bench/Sdcb.GcdrNet.Benchmark -c Release
$exe = 'bench/Sdcb.GcdrNet.Benchmark/bin/Release/net10.0/Sdcb.GcdrNet.Benchmark.exe'

& $exe --corpus vulkan                                    # Vulkan
& $exe --corpus cpu                                       # CPU AVX2+FMA（默认）
$env:DOTNET_EnableAVX2 = '0';        & $exe --corpus cpu  # CPU Vector<float>（128 位）
$env:DOTNET_EnableHWIntrinsic = '0'; & $exe --corpus cpu  # CPU Scalar
```

ISA 档位通过 JIT 开关强制降级，与 `test-all.ps1` 的矩阵一致。

## 结果汇总

轮数：Vulkan / AVX2 / Vector 各 **3 轮**取均值；Scalar 单轮 ~18 分钟，只跑 **1 轮**。
49 张合计 183.4 MP。CPU 内存列来自单独的 1 轮（Scalar 未测），Vulkan 内存列取自 R1，口径见下。

| 后端          | 轮数 | pass1 总耗时 | pass1 均值/张  | pass2 均值/张 | 相对 AVX2 | PSNR     | ΔWS          | peak WS     |
| ------------- | ---- | ------------ | -------------- | ------------- | --------- | -------- | ------------ | ----------- |
| Vulkan        | 3    | 2.82 s       | **57.5 ms**    | 52.3 ms       | 25.6× 快  | 24.68 dB | **+969 MB**  | **1939 MB** |
| CPU AVX2+FMA  | 3    | 72.18 s      | **1473.1 ms**  | 1472.1 ms     | 1.00×     | 24.65 dB | **+4380 MB** | **4918 MB** |
| CPU Vector    | 3    | 127.06 s     | **2593.1 ms**  | 2662.6 ms     | 1.76× 慢  | 24.65 dB | **+4391 MB** | **4932 MB** |
| CPU Scalar    | 1    | 559.23 s     | **11412.9 ms** | 11085.0 ms    | 7.75× 慢  | 24.65 dB | —            | —           |

CPU 三档逐级比：AVX2 → Vector **1.76×**，Vector → Scalar **4.40×**。

内存口径（working set，参考 SimdPaddleOCR）：harness 在**引擎初始化前后**各采一次进程
working set（`Process.Refresh()` 后读 `WorkingSet64`），之后**每张图** `Enhance` 返回后采一次
（不在计时内、不触发 GC）。

- **ΔWS**：loaded（`GcdrEnhancer.FromFiles` 之后）→ pass1 最后一张（50_shadow）。
- **peak WS**：warmup + pass1 + pass2 所有采样点的最大值。
- 三个后端 init → loaded 都是 26 → 49 MB（+22 MB）；managed Δ（pass1 前后各一次
  `GC.GetTotalMemory(true)`，仅这两次强制 GC）Vulkan +330 MB（缓存的每尺寸 plan），CPU −1 MB。
- Vulkan 的 working set 不含设备本地显存，显存看 `vkDevice`（`vkAllocateMemory` 计数）：
  live 2930 MB / peak 2930 MB（单个 worker 的 gc+dr arena 约 450 B/px，加 25% 增长余量）。
- CPU 的 ws 基本等于"当前尺寸 gc+dr 两个 session 的 arena"（5 MP 约 4.4 GB，小图降到
  ~1 GB），随图片尺寸起落，不再累积。
- CPU 内存这一轮的耗时（AVX2 1513.6 / Vector 2674.1 ms/张）没有计入耗时列，
  与 3 轮均值相差约 +2–3%。

## 逐轮明细（pass1 / pass2 总耗时，ms）

| 后端         | R1                  | R2                  | R3                  | 波动（pass1） |
| ------------ | ------------------- | ------------------- | ------------------- | ------------- |
| Vulkan       | 2845.4 / 2547.1     | 2803.7 / 2539.5     | 2798.5 / 2596.3     | ±0.8%         |
| Vulkan（复用前） | 5558.0 / 4986.6 | 5725.1 / 5068.7     | 5706.7 / 5237.8     | ±1.5%         |
| CPU AVX2     | 72591.4 / 70651.4   | 72044.1 / 72042.3   | 71905.5 / 73706.3   | ±0.5%         |
| CPU Vector   | 126100.2 / 129648.5 | 129072.2 / 130914.9 | 126013.0 / 130840.0 | ±1.2%         |
| CPU Scalar   | 559233.0 / 543165.4 | —                   | —                   | —             |

## CPU arena 内存修复（前 → 后）

修复前 CPU 的 ws 会一路涨到 20 GB 以上：

| 后端       | ΔWS 修复前    | peak 修复前   | ΔWS 修复后   | peak 修复后  |
| ---------- | ------------- | ------------- | ------------ | ------------ |
| CPU AVX2   | +21216 MB     | 28369 MB      | +4380 MB     | 4918 MB      |
| CPU Vector | +20444 MB     | 25957 MB      | +4391 MB     | 4932 MB      |

- **原因**：`InferenceSession` 的 arena 是 `new float[ArenaFloats]`（LOH，5 MP 下 gc+dr 合计约
  4.4 GB），池驱逐后 `Dispose()` 是空实现，死 arena 要等 gen2 GC。本机 64 GB、内存负载
  远低于 GC 的高水位阈值，gen2 迟迟不触发，于是 49 张交错尺寸下垃圾堆到 20+ GB。
- **修复**：arena 改为 `NativeMemory.AllocZeroed`，`Dispose()` 确定性释放（另有 finalizer
  兜底 + `GC.AddMemoryPressure`）；`SizePool.Return` 与 `Dispose` 的竞态补了一次排空。
- **代价**：AVX2 1473 → 1514 ms/张、Vector 2593 → 2674 ms/张（约 +3%，新尺寸 arena 走 OS 新页
  缺页）；输出 hash 与修复前一致，PSNR 不变，测试在 AVX2 和 Vector 两档下都通过。
- 汇总表 CPU 行的 3 轮耗时是修复前测的。Scalar 本轮没有重测。

## Vulkan 跨尺寸复用（前 → 后）

稳态下同尺寸连跑只要 ~67 ms/张（5 MP），但复用前语料均值 115.6 ms：每换一个尺寸都要重建整条
GPU 管线（池按原始尺寸做 key），实测新尺寸多花 50–150 ms，比推理本身还贵：

| 新尺寸开销              | 复用前       | 复用后                              |
| ----------------------- | ------------ | ----------------------------------- |
| 权重打包 + 上传 ×2       | ~35 ms       | 0（按模型共享，按内容校验）          |
| 显存 arena 分配 ×2       | 14–16 ms     | 0（worker 持有，预算内只增不减）      |
| 新显存首次使用           | ~30–80 ms    | 只在 arena 变大时出现                 |
| plan 编译 + 调度表 ×2    | 6–8 ms       | 首次见到该补齐尺寸时 ~11–18 ms，之后命中缓存 |
| 补齐后同尺寸（1945 vs 1950） | 重建一次     | 0（只重录命令缓冲）                   |

做法参考 SimdPaddleOCR 的 `GpuGraphModel`：

- **常量按模型共享**：`VkConstStore` 按模型 + fp16 做 key，新尺寸的 const 列表逐项内容一致就
  复用已上传的 buffer，否则退回私有 store（行为不变）。
- **worker 与尺寸无关**：`VkGcdrPipeline` 持有 gc/dr arena 和 BGR staging，在 3 GB 预算内只增不减
  （变大时留 25% 余量；单张超预算的大图之后，下一张能放进预算的图会把 arena 缩回实际所需大小），
  尺寸变化时只重录命令缓冲（私有 command pool）；`GcdrEnhancer` 的 Vulkan 侧只剩一个
  worker 池，不再按尺寸建池、驱逐；空闲 worker 除第一个外，按显存合计 3 GB 的预算保留。

| 指标（49 张）      | 复用前              | 复用后              |
| ------------------ | ------------------- | ------------------- |
| pass1 均值/张      | 115.6 ms            | **57.5 ms**（2.01×）|
| pass2 均值/张      | 104.0 ms            | **52.3 ms**（1.99×）|
| pass1 − pass2      | 470–660 ms          | 202–298 ms          |
| ΔWS / peak WS      | +2128 / 2796 MB     | **+969 / 1939 MB**  |
| vkDevice peak      | 3011 MB             | 2930 MB             |
| 池 / worker        | 69 个尺寸池，驱逐 67 | 1 个池，1 个 worker  |

49 张输出 hash 与复用前**逐张一致**；新增测试 `Pipeline_SizeSwitching_MatchesFresh_Vulkan`
验证同一 worker 在多尺寸间切换（脏的、更大的 arena）与新建 worker 逐字节相同。

## 观察

- **Vulkan**：pass2 比 pass1 快 ~7–10%（首次见到的补齐尺寸要编译 plan、arena 变大时重新分配）；
  `vkDevice` live/peak ~2.93 GB；PSNR 24.68 dB（fp16 存储，与 CPU 差
  0.03 dB，在预期内）。稳态 5 MP ~67 ms ≈ 3.4 TFLOPS（模型约 45 kFLOP/px），
  约为 3080 Ti FP32 峰值的 10%，3×3 卷积占 GPU 时间 ~60%。
- **CPU**：每尺寸 session 编译几乎免费，pass1/pass2 基本持平（±3% 噪声）。同尺寸下
  ONNX Runtime 1.21 CPU 1637 ms，本实现 1327 ms（1600×2304）。
- 所有档位、所有轮次 pass1/pass2 输出 hash 全部一致；CPU 三档 PSNR 完全相同。