# Apple M4 Metal 性能（`--corpus` / `--benchmtl` harness）

口径与 [perf.md](./perf.md)、[corpus.md](./corpus.md) 一致：`--corpus <backend>`，
manifest NN 序，01_color 作 warmup 不计时，其余 **49 张**逐张计 `Enhance` 端到端（解码在计时外）；
pass1 含每尺寸冷启动（plan 编译、PSO 创建、arena 增长），pass2 为同进程全量复跑（同时校验输出 hash 确定性）。

## 环境

| 项     | 值                                                                                |
| ------ | --------------------------------------------------------------------------------- |
| CPU/GPU | Apple M4（Paravirtual 虚拟化切片，8 vCPU，16 GB 统一内存；`Apple Paravirtual device`） |
| OS     | macOS 26（arm64）                                                                  |
| .NET   | SDK 10.0.x，`net10.0`，Release                                                     |
| Commit | metal-backend 分支 HEAD                                                            |

> 虚拟化说明：本机是 Devin 的 Apple Virtualization 虚拟机（Metal 经 paravirt 透传），
> 调度开销比真机高（孤立 dispatch/commit 地板 ~0.8 ms），且宿主有小时级 ~1.5× 争抢抖动；
> **真机 M4 预期只会更快**。跨机器比绝对值没有意义，比的是同一台机器上 Metal vs CPU。

## 命令

```bash
dotnet build bench/Sdcb.GcdrNet.Benchmark -c Release
exe=bench/Sdcb.GcdrNet.Benchmark/bin/Release/net10.0/Sdcb.GcdrNet.Benchmark

$exe --corpus metal   # Metal（Apple7+ 默认 Auto 也走这里）
$exe --corpus cpu     # CPU（本机 arm64 → Vector<float> 走 AdvSimd 128 位）
$exe --benchmtl       # 单点：1600x1152 e2e 时延
$exe --benchcpu
$exe --checkgpu       # Metal vs CPU u8 输出偏差抽查
$exe --mtlprof        # 逐 shader 剖析（每条 dispatch 单独 CommitAndWait，含隔离开销，看比例别看绝对值）
```

## 结果（同一时段同机测）

### e2e 单点（1600×1152，pass2 中位数）

| backend | 时延    |
| ------- | ------- |
| Metal   | 42.3 ms |
| CPU     | 316.3 ms |

**Metal ≈ 7.5× CPU。**

### corpus 49 张

| backend | 总时间      | 吞吐        | 均值       | psnr mean | 峰值工作集 |
| ------- | ----------- | ----------- | ---------- | --------- | ---------- |
| Metal   | 4264.7 ms   | **42.99 MP/s** | 87.0 ms | 24.55 dB  | 336 MB     |
| CPU     | 33739.7 ms  | **5.43 MP/s**  | 688.6 ms| 24.54 dB  | 1318 MB    |

- **吞吐 7.9×**，PSNR 持平（24.55 vs 24.54），pass1/pass2 输出 hash 完全一致。
- 工作集约为 CPU 的 1/4（fp16 存储 + 单 arena 复用）。
- `--checkgpu`：Metal vs CPU u8 输出偏差 max 1 LSB（fp16 舍入边界，预期内）。

## 实现要点

- 纯 C#，零原生依赖：ObjC runtime 直接 `objc_msgSend` P/Invoke（`Backends/Metal/ObjC.cs`），
  不引 `Metal.NET`/Xamarin。嵌入式 `.metal` 源码在运行期经 `MTLLibrary` 编译为 PSO（~40 个，启动 ~0.5 s）。
- fp16 存储 + fp32 运算；统一内存（`StorageModeShared`），CPU 写 `MtlBuffer.Contents` 直传权重。
- 逐节点 emit（`MtlGraph.Emit`，18 个 ExecOp 全部覆盖），与 `ExecPlan` 同构：
  - `conv_mm`：simdgroup-MMA 隐式 GEMM（BM=64 × BN∈{128,64,32}），cin%4==0 走 half4 向量化读；
  - `convpool`/`convpool_k3`：ic<8 小通道 stride-1 卷积 + 融合 2×2/2 maxpool，线程=1 池化像素×8 oc；
  - `uphead`/`uphead_sc`：融合 resize+relu、skip conv、skip relu、act 的单次输出头；
  - `layernorm`：simdgroup-per-row（32 lanes/行，8 行/threadgroup，零 barrier）；
  - 其余 elementwise/gemm/resize/transpose/concat 均为逐元素或 tiled kernel。
- `GcdrBackend.Auto`：仅在 macOS arm64 且 `SupportsApple7`（simdgroup-MMA）时选 Metal；
  `GcdrBackend.Metal` 显式选择不做能力门控（缺 Apple7 会在 PSO 编译期报错）。

## 调试工具（bench 内）

- `--mtrace gc|dr <h> <w> [golden.f32]`：CPU plan 与 Metal plan 逐节点对拍（rmse/max/nan），可加载 golden 真输入。
- `--checkgpu`：随机输入整图 u8 偏差。
- `--mtlprof`：逐 shader 累计耗时。
- `MtlKernelTests`：conv_mm/convpool/layernorm/uphead 的随机输入单 kernel 对拍（独立于 e2e）。

## MSL 移植踩过的坑（备查）

1. `threadgroup float4` 的动态分量写 `v[i]=x` 在 MSL 里写的是临时副本——必须 `((threadgroup float*)v)[idx]=x`。
2. `device const half* p[i]` 下标返回单个 half，不会隐式广播成 half4——vec4 读需 `*(device const half4*)(p+i*4)`。
3. 一个 kernel 最多一个 `constant` PC 参数位于 `buffer(binds.Length)`——emit 侧只写一个 blob；多个 `constant` 形参要合并成一个 struct。
4. `simd_sum(v)` 只归约调用线程所在 simdgroup（32 lanes）的 32 个值，跨 simdgroup 要经 threadgroup 数组 + barrier。
