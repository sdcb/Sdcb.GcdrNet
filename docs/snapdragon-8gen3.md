# 骁龙 8 Gen 3（真我 GT5 Pro）CPU NEON / Vulkan（Adreno 750）性能（`--corpus` harness，49 张）

口径与 [perf.md](./perf.md)、[corpus.md](./corpus.md) 相同：

- 命令是 `--corpus cpu|vulkan`，按 manifest 的 NN 序跑。
- 01_color 作 warmup，不计时；其余 **49 张**逐张计 `Enhance` 端到端，解码不计时。
- pass1 含每个尺寸的冷启动开销（plan 编译、arena 增长）；pass2 是同一进程全量复跑，同时校验输出 hash 确定性。

**结论：Adreno 750 没有协作矩阵，compute subgroup 只能是 64 或 128，原来走的是 `conv_tile` / `gemm` 这类共享内存分块内核，只到峰值的 8% 左右，比本机 CPU 还慢。新增一组不用共享内存的寄存器分块内核后，语料 Vulkan 从 1254 ms/张降到 486 ms/张（2.6 倍），是 CPU（1034 ms/张）的 2.1 倍。另外修掉了一个只在 Adreno 上出现的精度问题（depthwise 边界被算错）。CPU 没有改代码：这台手机不开放 SVE，几个 NEON 参数在手机上重扫都在噪声内。桌面路径不变。**

## 环境

| 项      | 值                                                                                                               |
| ------- | ---------------------------------------------------------------------------------------------------------------- |
| 手机    | 真我 GT5 Pro（RMX3888），Android 16                                                                               |
| SoC     | 骁龙 8 Gen 3（SM8650）：1× Cortex-X4 3.30 GHz + 3× A720 3.15 GHz + 2× A720 2.96 GHz + 2× A520 2.27 GHz           |
| CPU 特性 | `asimd asimddp asimdhp i8mm bf16 …`，**没有 `sve` / `sve2`**                                                     |
| GPU     | Adreno 750，Vulkan 1.3.128，driver `0x802fa029`，与 CPU 共用 16 GB LPDDR5X                                         |
| .NET    | 10.0.11 CoreCLR（`net10.0-android` 宿主，引用库的 `net10.0` 构建）                                                |
| 散热    | 被动散热；每次测量前等 `cpuss-1` 降到 42 °C 以下（`thermal_zone11`），CPU / Vulkan 同一时段交替跑                  |

### 宿主

新增 [`bench/Sdcb.GcdrNet.AndroidBench`](../bench/Sdcb.GcdrNet.AndroidBench)：`net10.0-android` 的 APK，链接桌面 `bench/Sdcb.GcdrNet.Benchmark/Program.cs`，所有模式（`--corpus`、`--bench*`、`--vkprof`、`--node` …）都一样。图片在手机上用 `BitmapFactory` 解码（SkiaSharp 2.88 的 android 库过不了 CoreCLR 的程序集改写，`XALNS7015`）。`run.ps1` 负责编译 → `adb install` → 推 `tests/testdata` → 启动 → 等结束 → 拉回日志：

```powershell
cd bench/Sdcb.GcdrNet.AndroidBench
.\run.ps1 -Build -Install -Push -Bench "--corpus vulkan"
.\run.ps1 -Bench "--vkprof 1152 1600" -Env "GCDRNET_VK_VERBOSE=1"
.\run.ps1 -Build -Install -GcdrEnv "DOTNET_EnableHWIntrinsic=0" -Bench "--bench 512 512 1"   # Scalar
```

`DOTNET_Enable*` 在托管代码运行前就被读取，`am start` 传进去的环境变量来不及，所以用 `-GcdrEnv` 编进 APK（`AndroidEnvironment`）。

## 结果汇总

语料（49 张，同一时段先 Vulkan 后 CPU）：

| 后端                   | pass1 总耗时 | pass1 均值/张 | pass2 均值/张 | PSNR     | 进程 WS 峰值¹ | 显存     |
| ---------------------- | ------------ | ------------- | ------------- | -------- | ------------- | -------- |
| Vulkan（本轮）         | 23.8 s       | **486 ms**    | 483 ms        | 24.66 dB | 661 MB        | 420 MB   |
| Vulkan（优化前，main） | 61.4 s       | 1254 ms       | 1252 ms       | 24.63 dB | 646 MB        | 420 MB   |
| CPU NEON               | 50.7 s       | **1034 ms**   | 1086 ms²      | 24.64 dB | 1158 MB       | —        |

1. 安卓应用进程在引擎加载前就有约 411 MB 的 WS（运行时 + 框架），表中是整个进程的峰值。
2. pass2 比 pass1 慢是持续负载后的降频；Vulkan 两轮几乎一样。

CPU 输出与优化前逐张 hash 相同（49/49）；Vulkan 两轮 hash 都确定。

单尺寸 1600×1152（`--benchcpu` / `--benchvk`，取最小值，含上下行和前后处理）：

| 路径         | 优化前（main） | 本轮         |
| ------------ | -------------- | ------------ |
| Vulkan e2e   | 615.3 ms¹      | **226.0 ms** |
| CPU NEON e2e | 493.4 ms       | 506.0 ms²    |

1. 本轮代码用 `GCDRNET_VK_DOT=0` 关掉新档是 619.6 ms。
2. CPU 代码没改，两次之间的差是降频。

512×512 gc+dr 推理（`--bench 512 512`）：NEON 83.2 ms，Scalar（`DOTNET_EnableHWIntrinsic=0`）1443.7 ms。

## Vulkan

### 设备能力

`GCDRNET_VK_VERBOSE=1`：`sg=64 sgRange=64-128`，没有 `VK_KHR_cooperative_matrix`，`maxComputeSharedMemorySize` 32 KB，fp16 storage 有，选中纯计算队列 family 1（时间戳 48 位可用）。所以 coopmat 三档（sg32 / sg32l / sg16）都不进，原来 3×3 卷积走 `conv_tile`（64 oc × 128 px 的共享内存分块，256 线程），Gemm 走 `gemm`（64×64 共享内存分块）。

同一台手机上 SimdPaddleOCR 测过纯寄存器 fp32 FMA 峰值约 2.09 TFLOPS（f16vec4 FMA 只有 0.91，没有翻倍），所以累加继续用 fp32。

### 新的一档：`VkDevice.ConvDot`

条件：没有协作矩阵、`VK_EXT_subgroup_size_control` 报告的最小 subgroup ≥ 64、有 fp16 storage（按能力判定，没有写死骁龙 / Adreno；`GCDRNET_VK_DOT=0` 关闭）。这一档换掉四类内核，都不用共享内存或少用共享内存：

| 内核 | 替换 | 做法 |
| --- | --- | --- |
| `conv_dot.comp` | 3×3 `conv_tile`（stride 1 / 2，ic、oc 是 4 的倍数） | 每个线程 4 个输出像素（stride 2：2 个）× 8 个输出通道，f16vec4 直接从全局读；一行卷积窗口的 TM·SW+2 个输入像素每 4 个输入通道读一次，三个 kx 抽头共用；n4 在 wave 内最快变化，输入读是广播、权重读是合并 |
| `gemm_dot.comp` | `gemm`（!transB） | 同样的寄存器分块，4 行 × 8 列 |
| `convpool_pi`（`convpool.comp -DPIXELS_INNER`） | stem 卷积 + 2×2 池化 | 4 个池化像素共用每次共享内存的权重读 |
| `uphead_sc2`（`uphead_sc.comp -DNP=2u`） | 尾部 UpHead（跳连卷积重算） | 每个线程 2 个水平相邻像素，跳连卷积和 1×1 头的权重读各供 2 个像素 |

`--vkprof 1152 1600`（逐 dispatch 单独提交取最小，ms；每个 dispatch 都带一次提交开销，合计比 e2e 大）：

| 着色器 | gc 优化前 | gc 本轮 | dr 优化前 | dr 本轮 |
| --- | ---: | ---: | ---: | ---: |
| 3×3 卷积 s1（`conv_k33s1` → `conv_dot_s1_4x2`） | 146.9 | 32.9 | 264.6 | 78.7 |
| 3×3 卷积 s2（`conv_k33s2` → `conv_dot_s2_2x2`） | 20.1 | 6.6 | 41.0 | 15.6 |
| UpHead（`uphead_sc` → `uphead_sc2`） | 22.4 | 16.0 | 49.2 | 33.6 |
| stem conv+pool（`convpool` → `convpool_pi`） | 8.1 | 5.7 | 27.8 | 16.5 |
| Gemm（`gemm` → `gemm_dot`） | 11.6 | 5.6 | 24.5 | 9.7 |
| 其余（layernorm、bilinear、maxpool …） | 26.8 | 23.2 | 45.0 | 27.1 |
| **合计** | **235.9** | **89.9** | **451.9** | **181.1** |

dr 的稠密卷积（约 25.6 GMAC）从约 168 GFLOPS 到约 545 GFLOPS，是 fp32 峰值的 26%。

### 扫过的形状（每项都在手机上交替测，`cpuss-1` 降温到 42 °C 以下）

`conv_dot` 每线程 TM 像素 × TN·4 通道，dr 的 s1 卷积合计（ms，最终版着色器）：

| TM×TN | 2×2  | 2×4  | 4×2       | 4×4   | 8×2  |
| ----- | ---: | ---: | --------: | ----: | ---: |
| s1    | 81.2 | 97.8 | **77–80** | 101.8 | 88.2 |

更早的一版（ky 循环里用 `continue`）里 1×2 / 1×4 / 2×1 / 3×2 / 4×1 / 8×1 都比 2×2 慢 15–120%，没有再测。stride 2 以 2×2（14.9–16.4）和 2×4（15.8–17.4）最好，选 2×2。工作组 64 / 128 / 256 没有差别（75.4 / 75.6 / 72.9），保留 64。

没有留下的：

| 尝试 | 结果 |
| --- | --- |
| 一个线程算 2 / 4 行输出（行间共用权重读） | 更慢（4×2×2：98.3，2×2×4：215.3），4×2×2 还会算错 |
| wave 内像素最快变化（权重读变成 wave 一致） | 慢 2.5–10 倍（输入读不再合并） |
| `convpool` 权重从全局读（不经共享内存） | 与共享内存版相同（17.5 对 16.0） |
| `uphead_sc` 每线程 4 像素 | 寄存器溢出，慢 9 倍（441.4） |
| `uphead_sc2` 权重从全局读（wave 一致下标） | 更慢（42.1 对 33.6） |
| 每行一个线程的 LayerNorm（f16vec4，无屏障） | gc 7.7 → 5.6，dr 不变；收益在逐 dispatch 计时的噪声里，没有留下 |

### Adreno 上的正确性问题

逐节点对拍（新增 `--vknodes gc|dr h w`：同一个 plan 在 CPU 和 GPU 上逐节点跑，打印每个节点输出的相对 RMS；`VKNODES_DUMP=1` 打印第一个出错节点的输入误差和出错位置分布）发现：

1. **`depthwise.comp` / `depthwise4.comp` 在零填充边界上算错。** 优化前就存在：dr 的第一个 depthwise（transformer 里的 3×3、GELU）输出相对误差从 7.7e-4 跳到 2.2e-1，错的都在边界行列。原写法是在抽头循环里 `if (!refl) { acc += wv * padVal; continue; }`，换成不带 `continue` 的写法（坐标 clamp / mirror 后读，再按是否越界选输入值或 padVal）后正确。整图 `--vkcheck 256 384`（对 CPU 的相对 RMS）：gc 8.0e-3 → 8.0e-4，dr 2.6e-2 → 8.1e-4；语料 PSNR 24.63 → 24.66 dB。
2. **`conv_dot` 的 4×2 形状**最初在 ky 循环里用 `continue` 跳过零填充行，结果每个线程第 3 个像素的第 2 组通道整组丢失（输出 0）；2×2、4×1、8×1 等形状不受影响。改成越界行用掩码读 0 后正确，所以 `conv_dot` 里不再有 `continue`。

两处数学上与原写法等价：3080 Ti 上 Vulkan 语料逐张 hash 与改动前 49/49 相同。仓库里其它带 `continue` 的着色器（`maxpool*`、`gemm`、`conv_tile` 的尾部）在这台手机上逐节点对拍都正确，没有改。

### 宿主侧

- `GcdrBackend.Auto` 在这台手机上选 Vulkan（有计算设备即用），现在 Vulkan 比 CPU 快 2.1 倍，保持不变。
- 加载的是 `/system/lib64/libvulkan.so`（`Vk.RegisterResolver` 早已包含 Android 的无版本号名字）。

## CPU

这台手机跑的就是 [rockchip-rk3399.md](./rockchip-rk3399.md) 里加的 NEON 档，没有改代码。

### SVE / SVE2

不会有优势，而且这台手机上用不了：

- 内核没有导出 `sve` / `sve2`（`/proc/cpuinfo` 的 Features 里没有，高通在 8 Gen 3 上关掉了），.NET 10 的 `Sve.IsSupported` 是 False。
- 即使开放，Cortex-X4 / A720 / A520 的 SVE 向量长度都是 128 位，和 NEON 一样宽；SVE 带来的只是谓词化尾部和 gather，对这里的 NHWC 卷积（通道是 4 的倍数、尾部已经很少）没有吞吐收益。
- 真正能提速的是 `bf16`（BFMMLA）和 fp16 算术（`asimdhp`），这台手机都有，但 .NET 10 的 `AdvSimd` 没有暴露这些指令。

### 在手机上重扫的参数（`--bench 1152 1600 3`，gc+dr 推理，ms）

| 参数 | 结果 | 结论 |
| --- | --- | --- |
| 线程数 8 / 6 / 4 | 443 / 446 / 560 | 两个 A520 几乎不贡献，但也不拖后腿（`Parallel.For` 动态分块），保持 8 |
| NEON Winograd V 预算 4K / 8K / 16K / 64K floats | 446–460 / 455 / 491 / 459 | 噪声内，保持 4K（RK3399 调出来的值） |
| 分组 Winograd 门槛 32K / 128K / 512K / 不分组 | 513 / 483 / 448 / 457 | 噪声内，保持 128K |

同一配置重复测的波动约 ±5%（降频）。逐节点看，最贵的是两个小输入通道的全分辨率卷积：dr 的 stem 6→32 conv+pool（约 136 GFLOPS）和 UpHead 里重算的同一个卷积（约 146 GFLOPS）；Winograd 层按直接卷积折算约 390–570 GFLOPS。单节点隔离测比整图里快约 1/3（例如 stem 32 ms 对 47 ms），差别主要是整图持续负载下的频率。

## 桌面

- 新内核只在 `ConvDot` 设备上进；`UseCoopConv` / `UseCoopGemm` 在 `ConvDot` 为真时让路（只有测试会在有协作矩阵的设备上强制打开）。
- 全量重编 SPIR-V 后，除 `depthwise*.spv`（上面的修复）外其余 `.spv` 逐字节不变；新增 `conv_dot_s1_4x2_h`、`conv_dot_s1_4x1_h`、`conv_dot_s2_2x2_h`、`conv_dot_s2_2x1_h`、`gemm_dot_h`、`convpool_pi_h`、`uphead_sc2_h`。
- 3080 Ti：Vulkan 语料 27.8 → 26.6 ms/张（噪声内），逐张 hash 49/49 相同；`dotnet test` 全过，新增 `VulkanConvDotTests` 在 3080 Ti 上强制走 `ConvDot` 档（这些着色器不用 subgroup 操作），gc / dr 各 3 个尺寸对 golden 通过。
- 880M、B580 用 `depthwise4`，改动在数学上等价，没有在这两台机器上复测。
