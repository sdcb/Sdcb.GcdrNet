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
| Commit  | `80ff057`                                            |

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

Vulkan / AVX2 / Vector 各 **3 轮**取均值，Scalar 单轮（~17 分钟）。49 张合计 183.4 MP。
内存列取自各后端 R1，口径见下。

| 后端          | 轮数 | pass1 总耗时 | pass1 均值/张  | pass2 均值/张 | 相对 AVX2 | PSNR     | ΔWS         | peak WS    | 显存       |
| ------------- | ---- | ------------ | -------------- | ------------- | --------- | -------- | ----------- | ---------- | ---------- |
| Vulkan        | 3    | 1.38 s       | **28.1 ms**¹   | 23.9 ms       | 25.7× 快  | 24.68 dB | **+283 MB** | **332 MB** | **420 MB** |
| CPU AVX2+FMA  | 3    | 35.37 s      | **721.8 ms**   | 715.0 ms      | 1.00×     | 24.65 dB | **+734 MB** | **794 MB** | —          |
| CPU Vector    | 3    | 47.03 s      | **959.8 ms**²  | 956.5 ms      | 1.33× 慢  | 24.65 dB | **+729 MB** | **784 MB** | —          |
| CPU Scalar    | 1    | 499.26 s     | 10188.9 ms     | 10191.0 ms    | 14.1× 慢  | 24.65 dB | +667 MB     | 721 MB     | —          |

单尺寸稳态（1600×2304 ≈ 3.7 MP，连续跑同一尺寸）：

| 后端         | 优化前    | 现在        | 倍数  |
| ------------ | --------- | ----------- | ----- |
| Vulkan e2e   | 47.8 ms   | **~17 ms**  | 2.8×  |
| CPU AVX2     | 1345 ms   | **673 ms**  | 2.0×  |
| CPU Vector   | 2187 ms³  | **~930 ms** | 2.4×  |

1. **Vulkan 的语料数字受 GPU 调频影响。** harness 在两张图之间解码 webp（CPU 约 100 ms），
   GPU 每次只忙 ~20 ms；NVIDIA 驱动按平均利用率把 3080 Ti 从 P0（1695 MHz）往下调，
   最低到 P3 的 ~420 MHz，单张随之从 ~25 ms 涨到最多 ~100 ms（`nvidia-smi` 实测；同尺寸
   背靠背连续跑则一直是 ~23 ms/5 MP）。轮间波动 ±2–5%，pass1 比 pass2 多出的部分是新尺寸的
   plan/graph 编译。连续供图的服务场景应看单尺寸稳态；要压低延迟可在 NVIDIA 控制面板设
   "最高性能优先"。
2. Vector 档原先的三轮双峰（一轮 ~1790、两轮 ~1530 ms/张）来自 `TileVec` 把累加器放在栈上、
   每次乘加都走一遍内存，快慢取决于 buffer 落在哪里；改成寄存器累加（`80ff057`）后
   1619 → 960 ms/张，三轮差 ±0.4%。
3. Vector 的"优化前"是 tail 融合之后、Winograd 之前的数字。

内存口径（working set，参考 SimdPaddleOCR）：harness 在**引擎初始化前后**各采一次进程
working set（`Process.Refresh()` 后读 `WorkingSet64`），之后**每张图** `Enhance` 返回后采一次
（不在计时内、不触发 GC）。源图/输出/真值三块 BGR 缓冲只增不减地复用，ws 里没有 harness 的
逐图垃圾。

- **ΔWS**：loaded（`GcdrEnhancer.FromFiles` 之后）→ pass1 最后一张（50_shadow）。
- **peak WS**：warmup + pass1 + pass2 所有采样点的最大值。
- **显存**：Vulkan 的 working set 不含设备本地显存，显存看 `vkDevice`（`vkAllocateMemory`
  计数）的 live/peak：单个 worker 的 arena（gc+dr 共用，fp16）。
- managed Δ（pass1 前后各一次 `GC.GetTotalMemory(true)`）：Vulkan +19 MB、CPU +26 MB——
  每尺寸 plan 只剩元数据，权重数组所有尺寸共享。
- CPU 的 ws 主要是 worker 的 arena（5.1 MP 一张 492 MB，约 96 B/px）：在 4 GB 预算内只增不减
  （首次之后增长留 25% 余量，余量页不触碰就不进 ws），不再因尺寸切换反复分配和缺页。

## 逐轮明细（pass1 / pass2 总耗时，ms）

| 后端         | R1                  | R2                  | R3                  | 波动（pass1） |
| ------------ | ------------------- | ------------------- | ------------------- | ------------- |
| Vulkan       | 1378.7 / 1127.8     | 1409.3 / 1185.0     | 1345.6 / 1204.1     | ±2.3%         |
| CPU AVX2     | 35381.6 / 35263.8   | 35582.4 / 35069.6   | 35143.1 / 34795.5   | ±0.6%         |
| CPU Vector   | 46800.1 / 46762.6   | 47067.0 / 46871.9   | 47217.2 / 47006.5   | ±0.4%         |
| CPU Scalar   | 499255.4 / 499357.1 | —                   | —                   | —             |

## 内存优化

同一 harness（缓冲复用版）、同一语料，单轮对比优化前 `1af112a` 和现在：

| 后端   | 指标        | `1af112a`   | 现在       | 降幅  |
| ------ | ----------- | ----------- | ---------- | ----- |
| AVX2   | peak WS     | 3293 MB     | **794 MB** | −76%  |
| AVX2   | ΔWS         | +2865 MB    | +734 MB    | −74%  |
| AVX2   | managed Δ   | +185 MB     | +26 MB     | −86%  |
| Vulkan | peak WS     | 933 MB      | **332 MB** | −64%  |
| Vulkan | ΔWS         | +826 MB     | +283 MB    | −66%  |
| Vulkan | 显存        | 1289 MB     | **420 MB** | −67%  |
| Vulkan | managed Δ   | +366 MB     | +19 MB     | −95%  |

速度不变：AVX2 708.8 → 721.8 ms/张（+1.8%，重算 stem 卷积的代价），Vulkan 31.8 → 28.1 ms/张；
CPU 输出 hash 逐张不变，Vulkan PSNR 逐张不变。

CPU 一张 1952×2624（5.1 MP）的 arena：

| 步骤                                  | gc         | dr          | worker 合计                 |
| ------------------------------------- | ---------- | ----------- | --------------------------- |
| 优化前（def 序 free list，两块 arena） | 799 MB     | 1516 MB     | 2315 MB（452 B/px）         |
| 紧凑排布（greedy by size）             | 717 MB     | 1352 MB     | 2069 MB                     |
| gc 叠在 dr 输入之后，共用一块          | —          | —           | 1352 MB                     |
| conv+maxpool 融合、UpHead 重算 skip    | 307 MB     | 492 MB      | **492 MB（96 B/px）**       |

做法：

1. **紧凑排布**（`d989e67`）：`PlanArena` 增加 greedy-by-size（按大小降序，每个 buffer 放进与它
   生存期重叠的已放置 buffer 之间最省的空隙），比 def 序 free list 紧，直接达到"同时存活字节
   之和"的下界；输入固定放在 offset 0。`GCDRNET_DUMP=3` 打印 arena、下界和峰值时刻的存活 buffer。
2. **gc 与 dr 共用一块 arena**（`d989e67`）：gc 跑完 dr 才开始，所以 gc 的 arena 叠在 dr 输入窗口
   之后（`GcdrEnhancer.StackArenas`）：除法校正读 gc 的输入/输出、写 dr 输入，三者不重叠；dr 再
   复用上面的全部空间。合计从 gc + dr 变成 max(dr, dr 输入 + gc)。CPU 和 Vulkan worker 都是一块。
3. **权重只存一份**（`d989e67`）：`OnnxTensorData.AsFloats()` 解码一次并丢掉原始字节，所有尺寸的
   plan 引用同一组数组（之前每个缓存的 plan 各拷一份，Vulkan 侧 32 个 graph 就是 ~380 MB）。
4. **conv + maxpool 融合**（CPU `6b77a95`）：编码器三层 `conv → maxpool(2×2/2)` 的卷积输出只被
   pool 读，于是卷积按行带（直接卷积 2 行、Winograd 一个 tile 行 = 4 行）算进线程局部缓冲后立刻
   pool，原分辨率的卷积输出不再落 arena。直接卷积和 Winograd 拆出 `DenseRow` / `WinoTiles`，
   与非融合路径共用逐 tile 的计算，输出逐位一致。
5. **UpHead 重算 skip**（CPU `6b77a95`，Vulkan `5b57738`）：尾部 UpHead 的 skip 是 stem 卷积
   （3/6 → 16/32 通道）的原分辨率输出，要从第一层活到最后一层（dr 32 通道 = 128 B/px）。现在
   UpHead 改读 stem 卷积的输入（反正一直存活），逐行（Vulkan 逐像素）重算 skip。代价是 stem
   卷积多算一遍（约占模型 MAC 的 10%，但通道少、访存小），AVX2 语料 +1.8%，3.7 MP 单尺寸稳态持平。
6. **Vulkan 只融合 stem**（`5b57738`）：`convpool.comp`（ic < 8 的卷积 + 2×2 max pool，权重放
   shared memory）和 `uphead_sc.comp`；更深层的 conv+pool 走张量核心，没有融合。

`GCDRNET_NOFUSE_MEM=1` 关掉 4–6，便于 A/B。剩下的 CPU arena 主要是 dr 的输入（6 通道）、
两个编码器 skip（1/2 分辨率 32 通道、1/4 分辨率 64 通道）和 1/8 分辨率 transformer 块里同时
存活的三个张量；再往下压就要重算第一层的 pool 输出，估计耗时 +5–10%，没做。

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
| 内存优化（见上） | `d989e67`…`5b57738` | 28.1 ms | 721.8 ms | 1619.0 ms |
| Vector 寄存器累加 | `80ff057` | — | — | 959.8 ms |

（单轮数字；"—" 表示该步不影响或未单独测。最后三行是 3 轮均值；最后两行换了缓冲复用的
harness，"内存优化"一行 Vulkan 的变化主要来自 harness 和调频，不是内存优化本身——同 harness 下
`1af112a` 是 31.8 ms/张。）

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

- **Vulkan**：稳态 ~17 ms/3.7 MP ≈ 10 TFLOPS 等效（模型约 45 kFLOP/px），比优化前的
  ~3.4 TFLOPS 高 2.8 倍；剩余时间主要是 layernorm、maxpool、shift、bilinear 等小算子和每尺寸编译。
  语料场景先受 GPU 调频影响（见注 1）。
- **CPU**：AVX2 单尺寸 673 ms，是 ONNX Runtime 1.21 CPU（1637 ms，同尺寸）的 2.4 倍；
  Vector 档与 AVX2 的差距 1.33×（128 位宽且无 FMA）。
- **内存**：CPU 各档 peak WS 都在 ~800 MB（Scalar 没有 Winograd 的线程缓冲，略低），其中 arena
  约 500 MB；Vulkan 进程 ws 332 MB、显存 420 MB。
- 所有档位、所有轮次 pass1/pass2 输出 hash 全部一致；CPU 各档 PSNR 完全相同，
  Vulkan（fp16 存储）高 0.03 dB。