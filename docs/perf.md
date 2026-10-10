# 本机性能（`--corpus` harness，49 张）

口径与 [corpus.md](./corpus.md) 一致：`--corpus <backend>`，manifest NN 序，
01_color 作 warmup 不计时，其余 **49 张**逐张计 `Enhance` 端到端（解码在计时外）；
pass1 含每尺寸冷管线创建，pass2 为同进程全量复跑（同时校验输出 hash 确定性）。

## 环境

| 项      | 值                                                   |
| ------- | ---------------------------------------------------- |
| CPU     | AMD Ryzen 7 5800X（8C/16T，AVX2+FMA，无 AVX-512）     |
| GPU     | NVIDIA GeForce RTX 3080 Ti（驱动 32.0.15.8180）       |
| 内存    | 64 GB                                                |
| OS      | Windows 11 专业工作站版                               |
| .NET    | SDK 11.0.100-rc.1，`net10.0`，Release                 |
| Commit  | `c2a19c3`                                            |

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
49 张合计 183.4 MP。内存列来自单独的 1 轮（Scalar 未测），口径见下。

| 后端          | 轮数 | pass1 总耗时 | pass1 均值/张  | pass2 均值/张 | 相对 AVX2 | PSNR     | ΔWS          | peak WS     |
| ------------- | ---- | ------------ | -------------- | ------------- | --------- | -------- | ------------ | ----------- |
| Vulkan        | 3    | 5.66 s       | **115.6 ms**   | 104.0 ms      | 12.7× 快  | 24.68 dB | **+2128 MB** | **2796 MB** |
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
  `GC.GetTotalMemory(true)`，仅这两次强制 GC）Vulkan +381 MB，CPU −1 MB。
- Vulkan 的 working set 不含设备本地显存，显存看 `vkDevice`（`vkAllocateMemory` 计数）：
  live 2791 MB / peak 3011 MB。
- CPU 的 ws 基本等于"当前尺寸 gc+dr 两个 session 的 arena"（5 MP 约 4.4 GB，小图降到
  ~1 GB），随图片尺寸起落，不再累积。
- 内存这一轮的耗时（Vulkan 117.4 / AVX2 1513.6 / Vector 2674.1 ms/张）没有计入耗时列，
  与 3 轮均值相差约 +2–3%。

## 逐轮明细（pass1 / pass2 总耗时，ms）

| 后端         | R1                  | R2                  | R3                  | 波动（pass1） |
| ------------ | ------------------- | ------------------- | ------------------- | ------------- |
| Vulkan       | 5558.0 / 4986.6     | 5725.1 / 5068.7     | 5706.7 / 5237.8     | ±1.5%         |
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
- 汇总表的 3 轮耗时是修复前测的。Scalar 本轮没有重测。

## 观察

- **Vulkan**：pass2 比 pass1 快 ~10%（冷管线创建 ~470–660 ms/49 张）；
  `vkDevice` live ~2.78 GB、peak ~3.0 GB；PSNR 24.68 dB（fp16 存储，与 CPU 差
  0.03 dB，在预期内）。
- **CPU**：每尺寸 session 编译几乎免费，pass1/pass2 基本持平（±3% 噪声）。
- 所有档位、所有轮次 pass1/pass2 输出 hash 全部一致；CPU 三档 PSNR 完全相同。
