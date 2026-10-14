# Rockchip RK3399（OrangePi 4 LTS）CPU NEON 性能（`--corpus` harness，49 张）

口径与 [perf.md](./perf.md)、[corpus.md](./corpus.md) 相同：

- 命令是 `--corpus cpu`，按 manifest 的 NN 序跑。
- 01_color 作 warmup，不计时；其余 **49 张**逐张计 `Enhance` 端到端，解码不计时。
- pass1 含每个尺寸的冷启动开销（plan 编译、arena 增长）；pass2 是同一进程全量复跑，同时校验输出 hash 确定性。

这块板子**只有 CPU 可用**：Mali-T860（Midgard）没有 Vulkan 驱动，系统里唯一的 Vulkan 设备是 llvmpipe（`PHYSICAL_DEVICE_TYPE_CPU`），`VkDevice.Create` 会跳过它，`GcdrBackend.Auto` 回退到 CPU。

## 环境

| 项     | 值                                                                                                           |
| ------ | ------------------------------------------------------------------------------------------------------------ |
| SoC    | Rockchip RK3399：2× Cortex-A72 1.8 GHz（CPU4–5，共享 1 MB L2）+ 4× Cortex-A53 1.416 GHz（CPU0–3，共享 512 KB L2） |
| SIMD   | AArch64 AdvSimd（NEON），没有 dotprod / fp16 算术 / SVE                                                       |
| 内存   | 4 GB LPDDR4                                                                                                  |
| OS     | Ubuntu 22.04.1，内核 5.18.5-rk3399，cpufreq `ondemand`                                                       |
| 散热   | 被动散热，无风扇；`cpu-thermal` trip 点 70 / 75 / 95 °C                                                       |
| .NET   | 运行时 10.0.12（`/root/.dotnet`），`net10.0` Release，在 Windows 上编译后拷贝到板子（IL 与平台无关）           |
| 线程数 | 默认 6（`Environment.ProcessorCount`），A72 和 A53 混用                                                      |

散热是这块板子最大的性能变量。满负载下大约十几秒 A72 就会到 70 °C 并开始降频，语料跑下来 A72 的平均频率大约只有 1.0 GHz（标称 1.8 GHz）。因此 A/B 对比分两种口径：

- **默认频率**：反映真实使用。
- **锁 1008 MHz**：把两簇的 `scaling_max_freq` 都设为 1008000，温度保持在 55 °C 左右，不触发降频，run-to-run 抖动约 2%，用于干净的 A/B 对比。

每次测量前先等 SoC 降到 50 °C 以下。

"优化前"指 main（`d613bb7`）。它在 ARM64 上没有专门的内核，走通用 `Vector<float>`（128 位）档位。它的 bench 有一个计时上限 bug，下文对优化前单独打了 harness 补丁，库代码不变，见文末"Harness"一节。

## 命令

```bash
# Windows 上编译，拷贝 bin/Release/net10.0 和 tests/testdata（含 tools/fetch-fixture.ps1 拉下来的 realdae/）到板子
dotnet bench/Sdcb.SimdGcdrNet.Benchmark.dll --corpus cpu                 # NEON（默认）
DOTNET_EnableHWIntrinsic=0 dotnet ... --corpus cpu                   # Scalar
dotnet bench/Sdcb.SimdGcdrNet.Benchmark.dll --benchcpu 1152 1600 3       # 单尺寸 e2e（真实 Enhance 路径）
dotnet bench/Sdcb.SimdGcdrNet.Benchmark.dll --bench 512 512 3            # gc+dr 推理（GCDRNET_PROF=1 GCDRNET_PROF_NODES=1 逐节点）
dotnet bench/Sdcb.SimdGcdrNet.Benchmark.dll --node dr 1152 1600 49 3     # 单节点隔离计时（新增）

# 锁频与温度 / 频率采样
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo 1008000 > $p/scaling_max_freq; done
cat /sys/class/thermal/thermal_zone0/temp /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq
```

注意：.NET 10 在 arm64 上 `DOTNET_EnableAdvSimd=0` 关不掉 `AdvSimd.Arm64.IsSupported`。降到 Scalar 要用 `DOTNET_EnableHWIntrinsic=0`。

## 结果汇总

语料（49 张，合计 183.4 MP，默认频率）：

| 后端                   | pass1 总耗时 | pass1 均值/张 | pass2 均值/张 | 相对 NEON | PSNR     | 逐张 WS 最大值 | A72 平均频率 / 平均温度 |
| ---------------------- | ------------ | ------------- | ------------- | --------- | -------- | -------------- | ----------------------- |
| NEON（本轮）           | 439.0 s      | **8959 ms**   | 10295 ms      | 1.00×     | 24.65 dB | 802 MB¹        | 1024 MHz / 69.4 °C      |
| Vector（优化前，main） | 1180.0 s     | **24081 ms**  | —²            | 2.69× 慢  | 24.65 dB | 839 MB         | 1067 MHz / 69.5 °C      |

1. 进程的 peak WS 为 813 MB，ΔWS 为 +747 MB（相对引擎加载后），与 x86 各档同量级。
2. 优化前单轮 pass1 就要 20 分钟，跑完 pass1 后就停了。

- 两轮都从 50 °C 冷机开始，跑几张图后 A72 就贴着 70 °C 降频。两轮的平均频率与温度几乎相同，所以 2.69× 可以直接比。
- NEON 的 pass2 比 pass1 慢，是因为板子已经热透，不是 harness 的开销。
- NEON 的 pass1 / pass2 输出 hash 逐张相同。
- 49 张里有 43 张的 u8 输出与优化前逐位相同。NEON 用 FMA，Vector 档用乘加分开，float 结果有细微差别。PSNR 相同。

单尺寸稳态（`--benchcpu 1152 1600 3`，真实 `Enhance` 路径，取最好一次）：

| 频率          | 优化前（Vector） | NEON          | 倍数      |
| ------------- | ---------------- | ------------- | --------- |
| 默认（热降频） | 9206.7 ms        | **3743.5 ms**（复测 3814.8） | **2.46×** |
| 锁 1008 MHz   | 14150.0 ms       | **5461.0 ms** | **2.59×** |

默认频率那一行，NEON 两次 A72 平均 1729 / 1718 MHz，优化前平均 1774 MHz，所以 2.46× 略偏保守。锁频那一行是干净的对比。

CPU 档位（`--bench 512 512`，gc+dr 推理，锁 1008 MHz）：

| 档位                            | gc       | dr       | 合计          | 相对 NEON |
| ------------------------------- | -------- | -------- | ------------- | --------- |
| NEON（本轮）                    | 293.6 ms | 597.0 ms | **890.6 ms**  | 1.00×     |
| Vector（优化前，main）          | 684.7 ms | 1483.5 ms | 2168.3 ms    | 2.43× 慢  |
| Scalar（`DOTNET_EnableHWIntrinsic=0`） | 3853.2 ms | 9582.0 ms | 13435.1 ms | 15.1× 慢 |

## 优化内容

NEON 档位（`Nhwc.UseNeon = AdvSimd.Arm64.IsSupported`）沿用 x86 各档的算子结构，以下是只在 NEON 上走的部分：

1. **6×16 micro-kernel `TileNeonK`**（`Kernels/Nhwc.Neon.cs`）
   - 每像素一次 128 位读入 4 个输入通道，`fmla` by element 累加到 24 个寄存器累加器（共 32 个向量寄存器）。
   - 下一组权重向量提前一组读入，照顾按序执行的 A53。`ic % 4` 的余数用 `ld1r` 广播。
   - 单核微基准（`taps × ic` 次累加，L1 内）：A72 6.3–7.1 GFLOP/s，约为峰值 7.5 的 85–95%；A53 3.5–3.8 GFLOP/s，约为峰值的 55%。
   - 热点方法标 `AggressiveOptimization`，不依赖分层 JIT 升级。绑单核测时后台 JIT 线程抢不到 CPU，epilogue 一直停在 Tier0（gc stem 慢了 2.6 倍）。
2. **直接卷积按核行 run 读入**（`DenseRowNeon`）
   - NHWC 下每个 ky 的 kw·ic 个输入是连续的，权重 `[ky][kx][ic][16]` 本身就是 `[ky][kx·ic][16]`。所以 k×k 卷积变成 kh 个 tap、每 tap 长度 kw·ic。3 / 6 通道的 stem 也因此能走 4 通道一组的主循环（run = 9 / 18）。
   - 跨越 reflect / 常数 padding 边界的像素会先 gather 到线程局部缓冲。
   - 满 tile、无残差、激活为 None 或 Relu 时，直接从寄存器写到输出，不经过 scratch tile。
   - dr stem（6→32，1600×1152，锁 1008 MHz）从 464 ms 降到 395 ms。
3. **Winograd**
   - GEMM 走 `TileNeonK`，输出 epilogue 改为向量化。
   - V 块预算降到 16 KB（ic ≥ 32 时 T = 6）。4 个 A53 共享 512 KB L2，块大了会互相挤出。
   - U（36·oc·ic）超过 2 MB 的深层改走三阶段分组调度 `WinogradGrouped`（`Kernels/NhwcWinograd.Neon.cs`）：输入变换 → 每个 (xi, oc 块) 一个 GEMM 任务，每块 U 每组只读一次 → 输出变换。原来是每 6 个 tile 就把整块 U 从 DRAM 流一遍。
   - 分组调度与原调度逐位相同：变换、k 序都一样。512→256 层（1600×1152，锁频）从 105 ms 降到 83 ms，256→128 从 120 ms 降到 93 ms。
4. **向量化激活**：`SimdMath` 增加 Vector128 版 Exp / Sigmoid / Erf / GELU，公式与 AVX2 版一致，用 FMA。
   - 深度卷积 + GELU 原来在非 AVX 档逐元素走标量 `ErfScalar`，现在向量化：gc 128 通道那一层从 21.6 ms 降到 8.5 ms（512²，锁频）。
   - 卷积 / Gemm / Winograd 的 epilogue 和 UpHead 的 sigmoid 也改为向量化。
5. **UpHead 整行融合**（`UpHeadRowNeon`，oc ≤ 4）：双线性插值用 FMA，加上 skip，再用 `fmla` by element 与打包好的 1×1 权重做点积。上采样后的像素不落地。
6. **LayerNorm**：非 AVX 档原来是单线程标量，现在按行组并行并向量化。dr 从 25 ms 降到 7 ms（512²，锁频）。

内存融合（conv+maxpool、UpHead 重算 skip）在 NEON 档同样逐位成立，由 `NeonKernelTests` 和 `MemFusions_*` 测试覆盖。peak WS 与 x86 同量级（813 MB）。

## 尝试过但没采用

- **A53 拆分读入**（`ldr d` + `ldr x` + `ins` 代替 `ldr q`）：不交错时 A53 慢 19%、A72 慢 17%；手工与 fmla 交错后 A53 仍慢 2%、A72 慢 8%。推测 A72 上 `ins` 占用了 FP 管线。
- **大权重直接卷积按输入通道分块**（每个任务固定一个 oc 块和一段行，权重分块常驻 L1）：A72 和 A53 单核都更慢（s2 层 127 ms 变成 173 ms）。
- **更大的 Winograd 块**（V 预算 32 KB–1 MB）：多线程更慢，L2 被挤出。

## 精度与测试

- 板子上 `--check small|pad` 对 PyTorch golden 的最大误差：gc ≤ 0.9e-6，dr ≤ 1.4e-6，与 x86 同级。
- 板子上 `dotnet test`：71 个通过，84 个跳过（Vulkan / AVX-512），排除了需要语料的冒烟测试。新增的 `NeonKernelTests`（36 个）都在其中。
  - 这组测试把 NEON 与通用 `Vector<float>` 实现在同进程内对比：容差 2e-5，Winograd 用 2e-4，相对输出 RMS。
  - 同时校验 NEON 档内融合与非融合逐位相同。
- 切换档位开关的测试（`NeonKernelTests`、`Avx512KernelTests`）放进不并行的 `TierToggle` collection。否则会与其他测试类并发跑，导致 `MemFusions_BitIdentical_AndShrinkArena` 偶发失败。
- x86 的路径没有改动（NEON 分支只在 `UseNeon` 为真时进入），`--check mid` 的误差与 main 完全相同，Windows 上默认档和 Vector 档测试全过。

## Harness

- `--bench` / `--benchcpu` 的"最好一次"原来初始化为 `1e9` tick。Linux 上 `Stopwatch.Frequency` = 1e9，这等于 1 秒上限，超过 1 秒的都被报成 1000.0 ms。现改为 `double.MaxValue`。优化前一列是给 main 的 bench 打了同样补丁后测的。
- `GCDRNET_PROF` 的逐算子统计不再计入 warmup，按每轮平均。
- 新增 `--node gc|dr h w idx reps`，在完整 warmup 后隔离计时单个节点。
- bench 和测试项目引用 `SkiaSharp.NativeAssets.Linux.NoDependencies`，语料解码和语料测试在 Linux 上可以跑。

## 后续可做

- A53 的 kernel 效率只有峰值的约 55%，它占全片算力的 2/3。在 RyuJIT 不做指令调度的前提下，C# 里还没找到更好的写法。
- 1600×1152 时 dr 的 UpHead 约占 10%，其中约 2/3 是重算 stem（6→32）。如果改为存储 skip 能省下这部分，但要多 236 MB 内存，在 4 GB 的板子上不划算，因此保留重算。
- Winograd 的变换仍是通用 `Vector<float>` 代码，没有 FMA，在 A72 上约占 Winograd 节点的 30%。
