# WebAssembly（browser-wasm）CPU PackedSimd 性能（`--corpus` harness，49 张）

口径与 [perf.md](./perf.md)、[corpus.md](./corpus.md) 相同：

- 命令是 `--corpus cpu`，按 manifest 的 NN 序跑。
- 01_color 作 warmup，不计时；其余 **49 张**逐张计 `Enhance` 端到端，解码不计时。
- pass1 含每个尺寸的冷启动开销（plan 编译、arena 增长）；pass2 是同一进程全量复跑，同时校验输出 hash 确定性。
- 输出 hash 与桌面 `--corpus` 用同一个采样 FNV-1a。

浏览器里没有 GPU 后端：browser / WASI 上 `GcdrBackend.Auto`、`Vulkan`、`Metal` 都直接落到 CPU（见下文"运行要点"）。

## 环境

| 项       | 值 |
| -------- | -- |
| 宿主     | AMD Ryzen 9 5950X（16C/32T），128 GB，Windows Server 2025 |
| 浏览器   | Microsoft Edge 155.0.4283.45（Chromium / V8，`--headless=new`） |
| .NET     | 运行时 10.0.12，`browser-wasm`，`wasm-tools` 负载（Emscripten 3.1.56 + LLVM AOT，`RunAOTCompilation=true`） |
| SIMD     | WASM SIMD128：`PackedSimd.IsSupported = True`，`Vector<float>.Count = 4`；无 relaxed SIMD（无 FMA） |
| 多线程   | `WasmEnableThreads=true`：Web Workers + `SharedArrayBuffer`（server.mjs 下发 COOP `same-origin` / COEP `require-corp`） |
| 线程数   | 32（`Environment.ProcessorCount` = `navigator.hardwareConcurrency`），可用 `GCDRNET_THREADS` 覆盖 |

## 命令

```powershell
# 前置：dotnet workload install wasm-tools-net10；Node.js 18+ 在 PATH 上
$wb = ".\bench\Sdcb.SimdGcdrNet.WasmBench\run.ps1"

& $wb                                          # publish（LLVM AOT）+ --corpus cpu，49 张
& $wb -NoBuild -Bench "--corpus cpu --vector"  # 同一产物，Vector<float> 档（PackedSimd 关）做 A/B
& $wb -NoBuild -Bench "--corpus cpu" -Count 5  # 只跑前 5 张计时图
& $wb -NoBuild -Bench "--bench 1152 1600 3" -Env GCDRNET_PROF=1   # gc+dr 推理 + 逐算子耗时
& $wb -NoBuild -Bench "--benchcpu 1152 1600 3" # 真实 Enhance 路径单尺寸
& $wb -NoBuild -Bench "--tiers 384 512"        # PackedSimd vs Vector<float> 同进程对比
& $wb -NoBuild -Bench "--check small"          # 对 PyTorch golden
& $wb -NoBuild -Env GCDRNET_THREADS=4          # 限制线程数
& $wb -NoBuild -Open                           # 可见浏览器窗口（不自动退出）
& $wb -NoAot                                   # 跳过 AOT（解释 + jiterpreter，只用于调试，慢一个数量级）
```

结果 JSON 存到 `artifacts/wasm/`，控制台输出与桌面 `--corpus` 同格式。

## 结果汇总

语料（49 张）：

| 档位 | pass1 总耗时 | pass1 均值/张 | pass2 均值/张 | 相对 PackedSimd | PSNR | wasm 线性内存 |
| ---- | ------------ | ------------- | ------------- | --------------- | ---- | ------------- |
| PackedSimd（本轮） | 34.1 s | **695.9 ms** | 684.5 ms | 1.00× | 24.65 dB | 1623 MB |
| `Vector<float>`（`--vector`） | 53.0 s | 1081.3 ms | 1084.0 ms | 1.55× 慢 | 24.65 dB | 1623 MB |
| 桌面同机 128 位 `Vector<float>`（`DOTNET_EnableAVX2=0`，参考） | 46.8 s | 954.4 ms | 952.3 ms | 1.37× 慢 | 24.65 dB | — |
| 桌面同机 AVX2（参考） | 20.6 s | 419.8 ms | 418.0 ms | 1.66× 快 | 24.65 dB | — |

- 49 张合计 183.4 MP；PackedSimd 5.38 MP/s，`Vector<float>` 3.46 MP/s。
- 浏览器里的 PackedSimd 比同机桌面 RyuJIT 的 128 位 `Vector<float>` 档还快 1.37×，是桌面 AVX2 的 60%。
- pass1 只比 pass2 多 2%（每尺寸 plan 编译等冷启动）；01_color warmup 首张 2.1 s。
- 两档 pass1 / pass2 hash 逐张一致。
- wasm 线性内存（只增不减，即峰值）约 1.6 GB，托管堆峰值 289 MB，其余是 5 MP 图的 CPU arena。离默认 2 GB 上限不远：更大的图在浏览器里要先缩小。

单尺寸（`--bench 1152 1600 3`，gc+dr 推理，取最好一次）：

| 档位 | gc | dr | 合计 | 相对 PackedSimd |
| ---- | -- | -- | ---- | --------------- |
| PackedSimd | 130.6 ms | 263.1 ms | **393.7 ms** | 1.00× |
| `Vector<float>`（`--vector`） | 209.8 ms | 426.5 ms | 636.3 ms | 1.62× 慢 |
| 桌面 128 位 `Vector<float>`（参考） | 194.0 ms | 336.5 ms | 530.5 ms | 1.35× 慢 |
| 桌面 AVX2（参考） | 84.8 ms | 142.6 ms | 227.3 ms | 1.73× 快 |

逐算子（`GCDRNET_PROF=1`，ms/run，`Vector<float>` → PackedSimd）：

| 算子 | gc | dr |
| ---- | -- | -- |
| Conv | 108.3 → 73.1（1.48×） | 222.6 → 159.4（1.40×） |
| LayerNorm | 40.8 → 3.9（10.5×） | 77.0 → 5.5（14.0×） |
| UpHead | 22.8 → 10.9（2.09×） | 52.3 → 30.4（1.72×） |
| Gemm | 15.7 → 11.6（1.35×） | 28.2 → 27.2（1.04×） |
| Transpose | 20.7 → 22.9 | 28.8 → 31.1 |
| ResizeBilinear / ChannelShift / Elementwise | 13.6 → 13.0 | 23.8 → 25.2 |

Transpose 现在是 gc / dr 里仅次于 Conv 的算子（未优化，两档相同）。

正确性：

- `--check small`：gc 1.10e-6，dr 1.19e-6；`--check pad`：gc 0.66e-6，dr 1.55e-6（门限 2e-3，与 x86 / NEON 同级）。
- `--tiers 384 512`（同进程 PackedSimd vs `Vector<float>`）：gc rel-RMS 1.3e-7（max 1.3e-6），dr rel-RMS 3.1e-7（max 2.4e-6）；e2e u8 输出 589824 字节里 17 个差 1。
  - 整网输出不是逐位一致（gc 38%、dr 18% 的 float 位相同）：`Vector<float>` 档的 GELU / Sigmoid epilogue 和深度卷积 + GELU 走标量 `ErfScalar` / `MathF.Exp`，LayerNorm 是顺序标量求和，PackedSimd 档这些都是向量公式 / 4 lane 求和，最后几位的差别会沿网络传播。PSNR 相同。

## 优化内容

各平台的 CPU 档位：x64 上 AVX-512 / AVX2，arm64 上 NEON，browser-wasm 上 **PackedSimd**；没有专门内核的算子（Transpose、Resize、MaxPool 等）仍走 `Vector<float>`，再往下是标量。PackedSimd 档（`Nhwc.UsePacked = PackedSimd.IsSupported`）参照 SimdPaddleOCR 的 WASM 优化（`Nhwc.Conv.PackedSimd`），结构上对应本仓库 NEON 档已有的特化点：

1. **4 像素 × 8 输出通道 micro-kernel `TilePkK`**（Sdcb.SimdOnnx `src/Sdcb.SimdOnnx/Kernels/Nhwc.PackedSimd.cs`）
   - V8 在 x64 上最多给 v128 分 15 个 XMM 寄存器，6 行 tile 会在内层循环溢出。改为 8 个累加器 + 2 个权重 + 1 个 splat，每个 16 通道打包块跑两遍。
   - 输入直接 `v128.load32_splat` 从内存广播，乘、加分开（WASM SIMD 无 FMA），按 k 升序累加：与 `Vector<float>` 档逐 lane 运算顺序相同。
   - 用于 1×1 / Gemm（`GemmPk`）、直接卷积（`DenseRowPk`）和 Winograd 的 36 个小 GEMM（tile 行数随之改为 4）。
2. **直接卷积按核行 run 读入**（`DenseRowPk`，同 NEON 的 `DenseRowNeon`）：NHWC 下每个 ky 的 kw·ic 个输入连续，k×k 卷积变成 kh 个长度 kw·ic 的 tap；3 / 6 通道的 stem 不再是 3 / 6 次的短循环。k 序不变。
3. **向量化激活**：`SimdMath` 新增 `ExpPk` / `SigmoidPk` / `ErfPk` / `GeluPk`，逐运算照搬 `Vector<float>` 公式（不用 `Vector128.FusedMultiplyAdd`，它在 WASM 上会退化成逐元素标量 `fmaf`）。只对有限输入等价：`pmax` / `pmin` 把 NaN 变成常数（ReLU(NaN) = 0、Sigmoid(NaN) = 1），参考实现会传播 NaN。
   - 深度卷积 + GELU、卷积 / Gemm / Winograd 的 epilogue 原来在非 AVX 档逐元素走标量 `ErfScalar` / `MathF.Exp`，现在向量化。
   - ReLU 用 `f32x4.pmax`（一条 `maxps`；`f32x4.max` 要额外的 NaN / -0 修正序列）。
4. **深度卷积**（`DepthwisePixelPk`）：每像素 16 通道一组（4 个累加器），所有激活都走向量路径。
5. **UpHead 整行融合**（`UpHeadRowPk`，oc ≤ 4）：双线性插值、skip 相加和 1×1 点积都在寄存器里完成；WASM SIMD 没有按 lane 乘，4 个通道经一次 16 字节落栈后各自 splat。运算顺序与未融合的 `Vector<float>` 路径相同。
6. **LayerNorm**（`PackedKernels.LayerNorm`）：非 AVX / NEON 档原来是单线程标量，现在按行组并行并向量化。

收益见上文逐算子表：LayerNorm（10–14×）和 Conv（1.4–1.5×）贡献最大。

## 运行要点

### GPU 后端探测

browser-wasm 上未链接的 P/Invoke（`vulkan-1`、`libobjc`）不会抛异常，而是以 `RuntimeError: null function` 直接让 wasm 模块 trap，`VulkanOk()` / `MetalOk()` 的 `catch` 根本执行不到。`GcdrEnhancer` 在 `OperatingSystem.IsBrowser() || IsWasi()` 时不再探测，`Auto` / `Vulkan` / `Metal` 都走 CPU（与 SimdPaddleOCR #30 的修复相同）。

### 多线程与跨域隔离

- `WasmEnableThreads=true` 依赖 Web Workers + `SharedArrayBuffer`，页面必须 `crossOriginIsolated`，所以 server.mjs 下发 `Cross-Origin-Opener-Policy: same-origin` 与 `Cross-Origin-Embedder-Policy: require-corp`。
- 多线程版运行时把 `Main` 放在 deputy worker 上，`Enhance` 里的同步 `Parallel.For` / 等待可以阻塞。
- 多线程版不能在 Node / V8 shell 里跑，只能进浏览器。

### AOT 与内存

- 默认 `RunAOTCompilation=true`（LLVM AOT）；`-NoAot` 只用于跳过编译等待的调试，速度慢一个数量级。
- wasm32 线性内存默认上限 2 GB，只增不减。这套语料峰值约 1.6 GB（桌面 working set 约 0.8 GB；浏览器里多出的是不归还的堆和 32 个线程的栈 / 线程局部缓冲）。

## Harness

`bench/Sdcb.SimdGcdrNet.WasmBench`（不在 `.slnx` 里：构建需要 wasm 负载），结构参照 SimdPaddleOCR 的 `test/Sdcb.SimdPaddleOCR.WasmBench`：

- `Program.cs`：浏览器入口，模式与桌面 benchmark 对齐（`--corpus` / `--bench` / `--benchcpu` / `--check`），另有 `--tiers`（同进程切换 `Nhwc.UsePacked` 对比两档）和 `--vector`（整轮关掉 PackedSimd）。模型、语料、golden 都从 `/testdata/` 经 HTTP 拉取；webp 用 ImageSharp 3.1.x 解码（3.2 起构建期需要许可证）。
- 浏览器 fetch 偶尔在复用的 keep-alive 连接上报 `TypeError: Failed to fetch`：harness 对 `/testdata/` 请求重试 3 次，server.mjs 把 `keepAliveTimeout` 调到 120 s。
- `server.mjs`：本地 HTTP 服务（只绑 127.0.0.1），下发 COOP/COEP，提供 publish 后的 `wwwroot` 与 `tests/testdata`，转发浏览器控制台，把结果 JSON 存到 `artifacts/`（只接受其下的 `.json`，请求体上限 16 MB），并用每次新建的临时 profile 拉起 headless Edge / Chrome，跑完后杀掉整个浏览器进程树并以 bench 的退出码退出（`--timeout` 默认 7200 s，超时退出码 124）。`/api/*` 要求 `X-WasmBench` 请求头且 Host 为 `127.0.0.1:<port>`：本机其它网页的跨域请求写不了文件，也结束不了运行。
- `wwwroot/index.html`：启动 .NET 运行时（`withApplicationArguments` / `withEnvironmentVariable`），显示进度，结束时报告 wasm 线性内存大小。
- `run.ps1`：`dotnet publish` → `node server.mjs`。
