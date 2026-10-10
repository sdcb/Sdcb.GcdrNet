# 本机性能（`--corpus` harness，49 张）

口径与 [corpus.md](./corpus.md) 一致：`--corpus <backend>`，manifest NN 序，
01_color 作 warmup 不计时，其余 **49 张**逐张计 `Enhance` 端到端（解码在计时外）；
pass1 含每尺寸的冷启动开销（plan 编译、arena 增长），pass2 为同进程全量复跑（同时校验输出 hash 确定性）。

## 环境

| 项      | 值                                                   |
| ------- | ---------------------------------------------------- |
| CPU     | AMD Ryzen 7 5800X（8C/16T，AVX2+FMA，无 AVX-512）     |
| GPU     | NVIDIA GeForce RTX 3080 Ti（驱动 32.0.15.8180）       |
| 内存    | 64 GB                                                |
| OS      | Windows 11 专业工作站版                               |
| .NET    | SDK 11.0.100-rc.1，`net10.0`，Release                 |
| Commit  | `3b7881d`                                            |

## 命令

```powershell
dotnet build bench/Sdcb.GcdrNet.Benchmark -c Release
$exe = 'bench/Sdcb.GcdrNet.Benchmark/bin/Release/net10.0/Sdcb.GcdrNet.Benchmark.exe'

& $exe --corpus vulkan                                    # Vulkan
& $exe --corpus cpu                                       # CPU AVX2+FMA（默认）
$env:DOTNET_EnableAVX2 = '0';        & $exe --corpus cpu  # CPU Vector<float>（128 位）
$env:DOTNET_EnableHWIntrinsic = '0'; & $exe --corpus cpu  # CPU Scalar

& $exe --bench 2304 1600 3      # CPU 单尺寸稳态（GCDRNET_PROF=1 打开逐算子 profile）
& $exe --benchvk 2304 1600 10   # Vulkan 单尺寸稳态（--vkprof 2304 1600 逐 shader）
```

ISA 档位通过 JIT 开关强制降级，与 `test-all.ps1` 的矩阵一致。

## 结果汇总

Vulkan / AVX2 / Vector 各 **3 轮**取均值；Scalar 单轮 ~18 分钟、本轮优化前测的，未重测。
49 张合计 183.4 MP。内存列取自各后端 R1，口径见下。

| 后端          | 轮数 | pass1 总耗时 | pass1 均值/张  | pass2 均值/张 | 相对 AVX2 | PSNR     | ΔWS          | peak WS     |
| ------------- | ---- | ------------ | -------------- | ------------- | --------- | -------- | ------------ | ----------- |
| Vulkan        | 3    | 1.86 s       | **37.9 ms**    | 44.8 ms¹      | 20.2× 快  | 24.68 dB | **+971 MB**  | **1959 MB** |
| CPU AVX2+FMA  | 3    | 37.58 s      | **767.0 ms**   | 748.9 ms      | 1.00×     | 24.65 dB | **+3219 MB** | **3715 MB** |
| CPU Vector    | 3    | 80.30 s      | **1638.8 ms**  | 1639.6 ms     | 2.14× 慢  | 24.65 dB | **+3143 MB** | **3692 MB** |
| CPU Scalar²   | 1    | 559.23 s     | 11412.9 ms     | 11085.0 ms    | —         | 24.65 dB | —            | —           |

单尺寸稳态（1600×2304 ≈ 3.7 MP，连续跑同一尺寸）：

| 后端         | 优化前    | 现在        | 倍数  |
| ------------ | --------- | ----------- | ----- |
| Vulkan e2e   | 47.8 ms   | **~18 ms**  | 2.7×  |
| CPU AVX2     | 1345 ms   | **671 ms**  | 2.0×  |
| CPU Vector   | 2187 ms³  | **1429 ms** | 1.5×  |

1. **Vulkan 的语料数字受 GPU 调频支配。** harness 在两张图之间解码 webp（CPU 约 100 ms），
   GPU 每次只忙 ~20 ms；NVIDIA 驱动按平均利用率把 3080 Ti 从 P0（1695 MHz）降到 P3，
   核心频率一路掉到 ~420 MHz，单张随之从 ~25 ms 涨到 ~100 ms（`nvidia-smi` 实测；同尺寸
   背靠背连续跑则一直是 ~23 ms/5 MP）。pass2 比 pass1 慢就是这个原因，轮间波动 ±5%。
   连续供图的服务场景应看单尺寸稳态；要压低延迟可在 NVIDIA 控制面板设"最高性能优先"。
2. Scalar 档本轮未重测（单轮 ~18 分钟）；表中是优化前的数字，Winograd 只在有 SIMD
   的档位启用，Scalar 只吃到了融合部分的收益。
3. Vector 的"优化前"是 tail 融合之后、Winograd 之前的数字；三轮语料里 R2 明显偏快
   （1468 vs ~1725 ms/张，桌面有其他负载），表中取三轮均值。

内存口径（working set，参考 SimdPaddleOCR）：harness 在**引擎初始化前后**各采一次进程
working set（`Process.Refresh()` 后读 `WorkingSet64`），之后**每张图** `Enhance` 返回后采一次
（不在计时内、不触发 GC）。

- **ΔWS**：loaded（`GcdrEnhancer.FromFiles` 之后）→ pass1 最后一张（50_shadow）。
- **peak WS**：warmup + pass1 + pass2 所有采样点的最大值。
- managed Δ（pass1 前后各一次 `GC.GetTotalMemory(true)`）：Vulkan +384 MB、CPU +168 MB，
  都是缓存的每尺寸 plan（plan 各自持有一份常量拷贝）。
- Vulkan 的 working set 不含设备本地显存，显存看 `vkDevice`（`vkAllocateMemory` 计数）：
  live/peak **1289 MB**（单个 worker 的 gc+dr arena，fp16 约 220 B/px，加 1/8 余量）。
- CPU 的 ws 主要是 worker 的两块 arena：在 4 GB 预算内只增不减（首次之后增长留 25% 余量），
  不再因尺寸切换反复分配和缺页。

## 逐轮明细（pass1 / pass2 总耗时，ms）

| 后端         | R1                  | R2                  | R3                  | 波动（pass1） |
| ------------ | ------------------- | ------------------- | ------------------- | ------------- |
| Vulkan       | 1959.6 / 2186.6     | 1783.1 / 2204.5     | 1830.9 / 2192.1     | ±4.7%         |
| CPU AVX2     | 37605.6 / 36991.6   | 37458.9 / 36720.3   | 37683.1 / 36373.8   | ±0.3%         |
| CPU Vector   | 84220.3 / 85003.3   | 71914.5 / 72209.0   | 84773.1 / 83803.4   | ±8.0%         |

## 优化历程（语料 pass1 均值/张）

| 步骤 | 提交 | Vulkan | AVX2 | Vector |
| ---- | ---- | ------ | ---- | ------ |
| 基线（3 轮） | `c2a19c3` | 115.6 ms | 1473.1 ms | 2593.1 ms |
| CPU arena 改 native（修内存失控，见下） | `b503ead` | — | 1513.6 ms | 2674.1 ms |
| Vulkan worker 跨尺寸复用 | `c1b866d` | 57.5 ms | — | — |
| 原分辨率尾部融合（UpHead） | `3b258a8` | 49.7 ms | 1219.7 ms | — |
| Winograd F(4×4,3×3) + CPU worker | `d324b96` | — | 765.0 ms | — |
| 张量核心卷积（KHR coopmat） | `5581c83` | ~34 ms | — | — |
| Gemm 上张量核心 | `3b7881d` | 37.9 ms¹ | 767.0 ms | 1638.8 ms |

（单轮数字；"—" 表示该步不影响或未单独测。最后一行是上面的 3 轮均值。）

### 原分辨率尾部融合

两个网络都以 `resize → relu → add → 1×1 conv（C→3）→ sigmoid` 收尾，全在原分辨率上，
每一步都把 C 通道的大张量读写一遍，而 oc=3 的 1×1 卷积让通用卷积 tile 空转 ~95%。
`ExecOp.UpHead` 每像素一次算完，中间值只在寄存器/栈里；中层的 `resize → add` 也并进
resize。CPU 内核保持与未融合版逐元素相同的运算顺序：**AVX2 语料输出 hash 逐张不变**。
Vulkan 显存 2930 → 1495 MB（arena 峰值落在尾部）。

### Winograd + CPU worker

3×3 步长 1、ic ≥ 16 的卷积（占 CPU 卷积时间的大头）改 F(4×4,3×3)：每个 4×4 输出 tile
的乘法从 144 降到 36。按 tile 块处理，变换后的输入/输出只在 L2 里，36 个小 GEMM 复用
6×16 微内核。相对直接卷积误差 < 1e-4（golden 容差内），PSNR 不变。

CPU 侧同时改成和 Vulkan 一样的结构：worker 持有可复用的 native arena，每尺寸 plan 缓存
（8 个）并通过 `CpuKernels.Cache` 共享打包好的权重（含 Winograd 变换），新尺寸只需编译 plan。

### 张量核心卷积

`conv_cm.comp`（参照 SimdPaddleOCR 的 `convk_cm_sg32`）：16×16×16 fp16 coopmat、fp32 累加，
A 直接从 NHWC 输入 gather（无 im2col），双缓冲 shared staging；支持零/反射填充、bias、
residual、relu/gelu。按 oc 选 BN=128/64/32，3/6 通道的首层用逐元素 gather 的变体。
transformer 块里的 Gemm 当作 1×M 图上的 1×1 卷积走同一个 shader。
需要 `VK_KHR_cooperative_matrix` 的 16×16×16 fp16→fp32 配置、32 lane subgroup、
shaderFloat16 和 Vulkan memory model，缺任何一项退回 `conv_tile`（`GCDRNET_VK_COOPMAT=0`
可强制关闭）；只在 fp16 存储下启用。

Vulkan 单尺寸稳态 GPU 时间（1600×2304，`--vkprof`）从 ~50 ms 降到 ~20 ms，
3×3 卷积从 ~30 ms 降到 ~8 ms。

### Vulkan 跨尺寸复用

复用前每换一个尺寸都重建整条 GPU 管线（权重重新打包上传 ~35 ms、arena 分配 14–16 ms、
新显存首次使用 30–80 ms），比推理本身还贵。现在权重按模型共享（`VkConstStore`，按内容校验），
worker 持有预算内只增不减的 arena 和 staging，换尺寸只重录命令缓冲；新尺寸的 gc/dr 两个
graph 并行编译。

### CPU arena 内存修复

修复前 CPU 的 ws 会一路涨到 20 GB 以上（AVX2 peak 28369 MB、Vector 25957 MB）：
`InferenceSession` 的 arena 是 LOH 上的 `new float[]`，池驱逐后要等 gen2 GC，而 64 GB 机器上
gen2 迟迟不触发。改为 `NativeMemory.AllocZeroed` + `Dispose()` 确定性释放；之后又被 worker
复用 arena 取代（见上）。

## 观察

- **Vulkan**：稳态 ~18 ms/3.7 MP ≈ 9 TFLOPS 等效（模型约 45 kFLOP/px），比优化前的
  ~3.4 TFLOPS 高 2.7 倍；剩余时间主要是 layernorm、maxpool、shift、bilinear 等小算子和每尺寸编译。
  语料场景先受 GPU 调频影响（见注 1）。
- **CPU**：AVX2 单尺寸 671 ms，是 ONNX Runtime 1.21 CPU（1637 ms，同尺寸）的 2.4 倍；
  Vector 档 Winograd 收益约 1.5×，与 AVX2 的差距 2.1×（128 位宽且无 FMA）。
- 所有档位、所有轮次 pass1/pass2 输出 hash 全部一致；CPU 两档 PSNR 完全相同，
  Vulkan（fp16 存储）高 0.03 dB。