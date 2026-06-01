# PD Disaggregation 项目


## 1. 项目一句话概览

本项目研究 LLM serving 里的 prefill/decode disaggregation，也就是把请求的 prompt prefill 阶段和逐 token decode 阶段拆成两个资源池，并在两阶段之间做 KV cache handoff。

当前仓库的主线是一个 engine-level MVP：

- 不依赖 `nano-vLLM`。
- 不真正搬运 GPU 上的 KV cache。
- 用纯 Python 数据结构模拟请求、系统压力、拓扑选择、成本估计、传输计划和 transfer engine。
- 重点验证“两阶段动态路由”这个调度思路是否清楚、可测试、可后续接入真实 serving 系统。

历史实验脚本仍保留在 `scripts/` 中，主要用于 GPU/NCCL KV 传输 microbenchmark、并发仿真和结果分析。

## 2. 推荐的熟悉顺序

建议按下面顺序走：

1. 先运行 demo，看到一批请求如何选择 prefill route 和 decode route。
2. 看 `pd_disaggregation/core/request.py` 和 `system_state.py`，理解路由输入。
3. 看 `topology.py`，理解候选拓扑和 prefill/decode 组合方式。
4. 看 `cost_model.py`，理解为什么某条 route 会被选中。
5. 看 `router.py`，理解两阶段决策逻辑。
6. 看 `transfer_plan.py`、`handles.py`、`transfer_engine.py`，理解 route 如何变成 handoff 计划。
7. 跑测试，确认行为边界。
8. 如果要理解历史实验，再看 `scripts/` 和 `docs/old_README.md`。

## 3. 目录结构

```text
.
├── README.md
├── pd_disaggregation/
│   └── core/
│       ├── request.py          # Request 和 KVMetadata
│       ├── system_state.py     # 路由时可观察到的队列、worker、GPU 状态
│       ├── topology.py         # TopologyConfig、默认候选拓扑、两阶段 route 合成
│       ├── profiler.py         # TransferProfiler，默认 profile 和 CSV 加载
│       ├── cost_model.py       # SLOCostModel，估算 prefill/decode/queue/transfer/SLO
│       ├── router.py           # DynamicPDRouter，两阶段调度策略
│       ├── handles.py          # RouteHandleBundle，静态 rank 映射抽象
│       ├── transfer_plan.py    # TransferPlanner 和 TransferPlan
│       └── transfer_engine.py  # 模拟执行 transfer plan 的 facade
├── examples/
│   └── run_two_stage_demo.py   # 推荐的第一入口
├── tests/
│   ├── test_topology.py
│   ├── test_cost_model.py
│   ├── test_router.py
│   └── test_transfer_plan.py
├── scripts/
│   ├── pd_topology.py
│   ├── bench_hetero_kv_transfer.py
│   ├── bench_pd_concurrent.py
│   ├── simulate_pd_concurrent.py
│   ├── analyze_pd_results.py
│   ├── run_gpu_pd_topology_full.sh
│   └── check_topology_sanity.sh
├── docs/
│   ├── old_README.md
│   └── images/
└── results/                   # 已生成或历史实验输出
```

## 4. 环境准备

这个仓库没有 `requirements.txt` 或 `pyproject.toml`。核心 MVP 代码只依赖 Python 标准库；测试需要 `pytest`；历史仿真和 GPU 脚本会额外需要 `numpy`、`pandas`、`matplotlib`、`torch` 等。

建议先用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install pytest
```

如果你只看主线 MVP，安装 `pytest` 就够了。

如果要运行 `scripts/simulate_pd_concurrent.py` 或 `scripts/analyze_pd_results.py`：

```bash
python -m pip install numpy pandas matplotlib
```

如果要跑真实 GPU/NCCL benchmark：

```bash
python -m pip install torch
```

真实 GPU 脚本还依赖可用的 CUDA/NCCL 环境，并通常需要通过 `torchrun` 启动多进程。

## 5. 快速跑通主线 MVP

从项目根目录运行：

```bash
python examples/run_two_stage_demo.py
```

这个 demo 会构造 6 个请求，覆盖轻载、长 prompt、decode 队列压力等情况。每个请求会经历：

1. 构造 `Request` 和 `SystemState`。
2. 调 `router.select_route(..., stage="prefill")` 选择 prefill route。
3. 根据 prompt 长度构造 `KVMetadata`。
4. 调 `router.select_route(..., stage="decode", kv_meta=...)` 选择 decode route。
5. 用 `combine_stage_routes()` 把两个阶段 route 合成最终 handoff route。
6. 用 `TransferPlanner` 生成 `TransferPlan`。
7. 用 `TransferEngine` 返回模拟 transfer 结果。
8. 用 `SLOCostModel` 打印 TTFT、E2E、SLO violation 等指标。

你应该会看到类似字段：

```text
request_id  prompt  output prefill route    decode route    prefill ms  transfer ms   decode ms    TTFT ms     E2E ms  SLO violation
req-001        512      64 balanced         balanced              ...
req-003       8192      96 strong_prefill   balanced              ...
req-006       6144     128 balanced         aligned_lane          ...
```

关注两个点：

- 长 prompt 倾向选择 `strong_prefill`，因为 prefill TP 更强。
- decode 队列压力高时倾向选择 `aligned_lane`，因为 decode DP lane 更多，能缓解队列集中。

## 6. 运行测试

```bash
python -m pytest tests
```

测试覆盖的行为：

- 拓扑字段校验和 world size 计算。
- 两阶段 route 合成。
- cost model 的单调性，比如更长 prompt 应该有更高 prefill cost。
- router 在轻载、长 prompt、decode 压力下的选择。
- decode 阶段必须提供 `KVMetadata`。
- transfer plan 的 rank 映射和 KV bytes 计算。

如果你改了调度策略，至少要跑完整测试。

## 7. 核心概念

### 7.1 Request

位置：`pd_disaggregation/core/request.py`

`Request` 表示一个 serving 请求：

```python
Request(
    request_id="req-001",
    prompt_len=512,
    output_len=64,
    arrival_time_ms=0.0,
    slo_ttft_ms=180.0,
    slo_e2e_ms=850.0,
)
```

字段含义：

- `request_id`：请求 ID。
- `prompt_len`：输入 prompt token 数。
- `output_len`：预期输出 token 数。
- `arrival_time_ms`：请求到达时间。
- `slo_ttft_ms`：time to first token 的 SLO。
- `slo_e2e_ms`：端到端延迟 SLO。

### 7.2 KVMetadata

位置：`pd_disaggregation/core/request.py`

`KVMetadata` 描述 KV cache 的形状，用于估算 handoff bytes：

```python
KVMetadata(
    num_layers=32,
    num_heads=32,
    head_dim=128,
    seq_len=request.prompt_len,
    dtype_bytes=2,
    tp_size=prefill_route.prefill_tp,
)
```

如果没有显式传入 `estimated_kv_bytes`，会自动计算：

```text
2 * num_layers * num_heads * head_dim * seq_len * dtype_bytes
```

这里的 `2` 代表 K 和 V 两份 tensor。

### 7.3 SystemState

位置：`pd_disaggregation/core/system_state.py`

`SystemState` 表示路由时调度器能观察到的系统压力：

```python
SystemState(
    prefill_queue_len=3,
    decode_queue_len=12,
    active_prefill_workers=2,
    active_decode_workers=2,
    gpu_count=8,
    current_time_ms=10.0,
)
```

字段含义：

- `prefill_queue_len`：prefill 队列中等待的请求数。
- `decode_queue_len`：decode 队列中等待的请求数。
- `active_prefill_workers`：可用 prefill worker 数。
- `active_decode_workers`：可用 decode worker 数。
- `gpu_count`：当前可用 GPU/rank 数。
- `current_time_ms`：当前模拟时间。

Router 不直接读真实 GPU 状态，而是读这个抽象状态。以后接入真实 serving 系统时，scheduler 需要把自身状态转换成 `SystemState`。

## 8. 默认拓扑候选

位置：`pd_disaggregation/core/topology.py`

默认候选由 `default_topologies()` 返回：

| route | prefill | decode | 主要意图 |
| --- | --- | --- | --- |
| `balanced` | TP2 x DP2 | TP2 x DP2 | 轻载默认选择 |
| `strong_prefill` | TP4 x DP1 | TP2 x DP2 | 长 prompt 或 TTFT 风险高时加强 prefill |
| `cross_tp` | TP2 x DP2 | TP4 x DP1 | decode 单请求计算更强，但 decode lane 少 |
| `aligned_lane` | TP2 x DP2 | TP1 x DP4 | decode 并行 lane 多，缓解队列压力 |

`TopologyConfig` 的关键字段：

- `prefill_tp` / `prefill_dp`：prefill 池的 tensor parallel 和 data parallel。
- `decode_tp` / `decode_dp`：decode 池的 tensor parallel 和 data parallel。
- `communication_factor`：相对通信成本因子。
- `queue_parallelism_factor`：队列并行能力因子。

`required_gpu_count` 的计算方式：

```text
prefill_world_size + decode_world_size
= prefill_tp * prefill_dp + decode_tp * decode_dp
```

默认候选都需要 8 个 rank，因为 prefill 池和 decode 池同时存在。

## 9. 两阶段路由流程

主逻辑在 `pd_disaggregation/core/router.py`。

### 9.1 第一阶段：prefill route

调用：

```python
prefill_route = router.select_route(
    request,
    system_state,
    stage="prefill",
)
```

决策时机：请求进入 prefill 前。

目标：决定 KV cache 由什么 prefill 拓扑生产。

重要规则：

- 如果系统轻载、prompt 不长、默认 `balanced` 能满足 TTFT，则直接选 `balanced`。
- 否则比较不同 prefill layout 的估算代价。
- 多个 route 如果拥有相同 prefill layout，只保留一个作为 prefill 候选。

最后这一点很重要：`cross_tp` 和 `aligned_lane` 的 prefill layout 都是 TP2 x DP2，它们真正的差异在 decode 阶段。因此 prefill 阶段不会把 decode-only 的差异误认为 prefill 优化。

prefill 阶段 score 大致由两部分组成：

```text
prefill_queue_delay + prefill_time + TTFT violation penalty
```

当 prompt 很长时，`strong_prefill` 的 TP4 会降低 prefill compute，所以经常胜出。

### 9.2 第二阶段：decode route

调用：

```python
decode_route = router.select_route(
    request,
    system_state,
    stage="decode",
    kv_meta=kv_meta,
)
```

决策时机：prefill 完成后，KV metadata 已知。

目标：决定 KV cache handoff 到哪个 decode 拓扑，并选择 decode 执行布局。

decode 阶段必须提供 `KVMetadata`，因为 transfer cost 依赖 KV bytes。

轻载时仍优先 `balanced`。非轻载时会比较完整 decode 代价：

```text
transfer_time + decode_queue_delay + decode_time_per_token * output_len + SLO penalty
```

当 decode queue 很长时，`aligned_lane` 往往更优，因为它是 TP1 x DP4，虽然单 token 计算不一定最快，但 lane 更多，排队时间可能显著下降。

### 9.3 合成最终 handoff route

调用：

```python
resolved_route = combine_stage_routes(prefill_route, decode_route)
```

这个函数会生成一个新的 `TopologyConfig`：

- route name 形如 `prefill_route->decode_route`。
- prefill TP/DP 来自第一阶段。
- decode TP/DP 来自第二阶段。
- `communication_factor` 取两者较大值。
- `queue_parallelism_factor` 采用 decode route 的值。

也就是说，两阶段选择不会强行要求 prefill 和 decode route 是同一个名字。

## 10. 成本模型怎么理解

位置：`pd_disaggregation/core/cost_model.py`

`SLOCostModel` 是可解释的启发式模型，不是黑盒优化器。它会估算：

- `prefill_time_ms`
- `decode_time_per_token_ms`
- `transfer_time_ms`
- `prefill_queue_delay_ms`
- `decode_queue_delay_ms`
- `total_ttft_ms`
- `total_decode_latency_ms`
- `total_e2e_latency_ms`
- `slo_violation`

### 10.1 Prefill 估算

prefill compute 随 `prompt_len` 增长，并受 TP 加速：

```text
2.5 + 22.0 * (prompt_len / 1024) / tp_efficiency(prefill_tp)
```

TP 不是线性加速，效率函数是：

```text
1.0 + 0.78 * (tp_size - 1)
```

TP 越大通信成本也越高：

```text
0.75 * (prefill_tp - 1) * communication_factor
```

队列压力会额外放大 prefill time。

### 10.2 Decode 估算

decode 每 token 时间同样受 decode TP 加速，也有 TP 通信成本。

decode queue delay 使用一个 MVP 假设：

```text
队列中每个请求平均还有 48 个 token 要 decode
```

这个假设写在代码注释里，是为了在没有真实 scheduler 状态的 standalone 阶段仍能模拟 decode 队列压力。未来接入真实系统时，应该用实际剩余 token、KV block、batch 状态替换它。

### 10.3 Transfer 估算

transfer latency 来自 `TransferProfiler`：

```text
startup_latency_ms + estimated_kv_bytes / effective_bandwidth * communication_factor
```

默认 profile 在 `profiler.py`：

| route | bandwidth_gbps | startup_latency_ms |
| --- | ---: | ---: |
| `balanced` | 112.0 | 0.08 |
| `strong_prefill` | 104.0 | 0.10 |
| `cross_tp` | 96.0 | 0.13 |
| `aligned_lane` | 118.0 | 0.07 |

也可以从 CSV 加载：

```python
from pd_disaggregation.core import TransferProfiler

profiler = TransferProfiler.from_csv("path/to/profile.csv")
```

CSV 格式：

```csv
route_name,bandwidth_gbps,startup_latency_ms
balanced,112.0,0.08
aligned_lane,118.0,0.07
```

## 11. TransferPlan 和模拟执行

### 11.1 rank 映射

位置：`pd_disaggregation/core/handles.py`

`RouteHandleBundle.from_topologies()` 会为每个 topology 构造连续 rank 映射：

```text
source_ranks = [0, ..., prefill_world_size - 1]
target_ranks = [prefill_world_size, ..., prefill_world_size + decode_world_size - 1]
```

这只是 standalone MVP 的简单抽象，方便后续替换成真实 process group、NCCL communicator 或 KV transfer handle。

### 11.2 生成 transfer plan

位置：`pd_disaggregation/core/transfer_plan.py`

调用：

```python
plan = planner.build_transfer_plan(
    request=request,
    route=resolved_route,
    kv_meta=kv_meta,
)
```

`TransferPlan` 包含：

- `request_id`
- `route_name`
- `estimated_transfer_bytes`
- `estimated_transfer_time_ms`
- `source_ranks`
- `target_ranks`

### 11.3 模拟执行

位置：`pd_disaggregation/core/transfer_engine.py`

当前 `TransferEngine.run(plan)` 不做真实传输，只返回预测结果：

```python
result = TransferEngine().run(plan)
```

返回的 `TransferResult.simulated` 默认为 `True`。

未来接入真实系统时，这一层应该替换成真实 KV movement 后端；上层 router、planner 和测试可以尽量保持稳定。

## 12. 从代码角度看完整调用链

最小调用链如下：

```python
from pd_disaggregation.core import (
    DynamicPDRouter,
    KVMetadata,
    Request,
    SystemState,
    TransferEngine,
    TransferPlanner,
    combine_stage_routes,
)

request = Request("req", 4096, 128, 0.0, 300.0, 1800.0)
state = SystemState(2, 12, 2, 2, 8, 0.0)

router = DynamicPDRouter()
planner = TransferPlanner(router.cost_model.profiler)
engine = TransferEngine()

prefill_route = router.select_route(request, state, stage="prefill")

kv_meta = KVMetadata(
    num_layers=32,
    num_heads=32,
    head_dim=128,
    seq_len=request.prompt_len,
    dtype_bytes=2,
    tp_size=prefill_route.prefill_tp,
)

decode_route = router.select_route(
    request,
    state,
    stage="decode",
    kv_meta=kv_meta,
)

resolved_route = combine_stage_routes(prefill_route, decode_route)
plan = planner.build_transfer_plan(request, resolved_route, kv_meta)
result = engine.run(plan)
estimate = router.cost_model.estimate(request, state, resolved_route, kv_meta)
```

如果你要 debug 某个请求为什么选了某条 route，最直接的方法是对每个候选调用：

```python
for route in router.topologies:
    estimate = router.cost_model.estimate(request, state, route, kv_meta)
    print(route.name, estimate)
```

## 13. 如何修改调度策略

常见改动点：

### 13.1 加一个新拓扑

修改 `pd_disaggregation/core/topology.py` 的 `default_topologies()`：

```python
TopologyConfig(
    name="my_route",
    prefill_tp=...,
    prefill_dp=...,
    decode_tp=...,
    decode_dp=...,
    communication_factor=...,
    queue_parallelism_factor=...,
)
```

注意：

- TP/DP 必须为正数。
- `required_gpu_count` 不能超过 `SystemState.gpu_count`，否则 router 会过滤掉。
- 如果新 route 需要单独 transfer profile，也要更新 `TransferProfiler.default()` 或加载 CSV。

### 13.2 改 prefill 决策

修改 `DynamicPDRouter.select_prefill_route()`。

重点关注：

- 什么情况下直接返回 baseline。
- prefill candidates 是否需要按 layout 去重。
- score 是否更强调 TTFT、prefill queue、长 prompt 或吞吐。

改完建议补测试：

- 轻载短请求仍选默认 route。
- 长 prompt 选择更强 prefill。
- decode-only route 不应在 prefill 阶段被误选。

### 13.3 改 decode 决策

修改 `DynamicPDRouter.select_decode_route()`。

重点关注：

- decode queue 对 route 的影响。
- transfer cost 是否应该更重。
- TTFT 和 E2E penalty 的权重。
- 是否要考虑输出长度、remaining tokens、batching 或 request priority。

改完建议补测试：

- decode queue 高时选择更多 decode DP lane。
- KV 很大时 transfer profile 能影响选择。
- SLO 收紧时选择有变化。

### 13.4 改成本模型

修改 `SLOCostModel`。

推荐做法：

- 保持每个估算项可以单独解释。
- 不要把所有逻辑塞进一个总分公式。
- 如果引入新的系统观测值，先把字段加到 `SystemState` 或新的 metadata 类型中。
- 为单调行为补测试，例如 prompt 更长、queue 更长、GPU 更少时估算应如何变化。

## 14. 历史 GPU / NCCL 脚本

`scripts/` 里的脚本主要来自前一阶段实验，重点是比较 post-prefill decode-side transfer/routing：

### 14.1 `scripts/pd_topology.py`

定义 GPU benchmark 用的拓扑 preset：

| preset | prefill | cross decode | grouped decode | total ranks |
| --- | --- | --- | --- | ---: |
| `p2t2_x1t4_g2t2` | DP2 TP2 | DP1 TP4 | DP2 TP2 | 8 |
| `p4t1_x1t4_g4t1` | DP4 TP1 | DP1 TP4 | DP4 TP1 | 8 |

可以打印支持的 topology：

```bash
python scripts/bench_hetero_kv_transfer.py --print-topologies
```

### 14.2 `scripts/bench_hetero_kv_transfer.py`

用途：真实或接近真实地测 KV transfer handle。

支持的 handle：

- `cross_tp`
- `aligned_lane`
- `aligned_lane_grouped`
- `dynamic`

它会构建 prefill/decode rank、transfer edge、message、process group，并测量 transfer latency、effective bandwidth 等。

典型运行需要 `torchrun`，例如 8 个进程：

```bash
torchrun --nproc_per_node=8 scripts/bench_hetero_kv_transfer.py \
  --backend nccl \
  --mode async \
  --topology p2t2_x1t4_g2t2 \
  --handle cross_tp \
  --use-process-groups \
  --seq-len 4096 \
  --chunk-tokens 4096 \
  --num-layers 32
```

如果没有 GPU/NCCL 环境，不建议从这里开始。

### 14.3 `scripts/simulate_pd_concurrent.py`

用途：用测得的 transfer profile 模拟并发请求流，比较：

- 固定 `cross_tp`
- 固定 `aligned_lane_grouped`
- 动态策略 `dynamic`

示例：

```bash
python scripts/simulate_pd_concurrent.py \
  --summary results/transfer_profile_summary.csv \
  --num-requests 200 \
  --arrival-pattern burst \
  --arrival-rate-rps 35 \
  --output-dir results/simulation/grouped_lane
```

输出包括：

- `generated_workload.csv`
- `sim_request_trace.csv`
- `sim_summary.csv`
- `figures/*.png`
- `manifest.json`

### 14.4 `scripts/analyze_pd_results.py`

用途：分析并发 benchmark 结果，生成图表。

默认读取：

```text
results/concurrent/nccl/
```

示例：

```bash
python scripts/analyze_pd_results.py \
  --root . \
  --output-dir results/report_assets \
  --show-values
```

这个脚本更偏历史报告生成；如果你现在只理解两阶段 MVP，可以先跳过。

## 15. 旧实验和当前 MVP 的关系

旧实验主要回答：

> 在 prefill 已经固定后，decode-side route 选 `cross_tp` 还是 `aligned_lane_grouped`，会如何影响 handoff gap、decode queue wait、E2E latency 和 makespan？

当前 MVP 往前推进了一步：

> 不只在 prefill 完成后选 decode route，而是在 prefill 开始前也做一次路由决策，让长 prompt 或 TTFT 风险高的请求能选择更合适的 KV 生产拓扑。

因此：

- `scripts/` 是历史实验和 profiling 基础。
- `pd_disaggregation/core/` 是当前更干净、更可测试的两阶段调度抽象。
- 后续集成真实 serving 系统时，应优先保持 `core/` 的接口清楚，再把 `scripts/` 里的真实 transfer/profile 能力逐步接回来。

## 16. 常见排查

### 16.1 `ModuleNotFoundError: No module named 'pd_disaggregation'`

请从项目根目录运行命令：

```bash
python examples/run_two_stage_demo.py
python -m pytest tests
```

demo 里已经把项目根目录加入 `sys.path`，测试也应从根目录跑。

### 16.2 `No topology fits the available GPU count`

`SystemState.gpu_count` 太小。默认拓扑都需要 8 个 rank：

```text
prefill_world_size + decode_world_size = 4 + 4 = 8
```

如果你设置 `gpu_count < 8`，默认候选会全部被过滤。

### 16.3 decode 阶段报 `kv_meta is required`

decode route 选择必须提供 `KVMetadata`：

```python
router.select_route(request, state, stage="decode", kv_meta=kv_meta)
```

因为 decode 阶段要估算 KV transfer cost。

### 16.4 `results/` 目录被 git 忽略

`.gitignore` 里忽略了 `results/`。本地可以保留实验输出，但默认不会提交。

### 16.5 运行仿真脚本缺少 pandas/numpy/matplotlib

核心 MVP 不需要这些依赖；历史仿真脚本需要：

```bash
python -m pip install numpy pandas matplotlib
```

### 16.6 运行 GPU benchmark 缺少 torch 或 NCCL

`bench_hetero_kv_transfer.py` 需要 PyTorch distributed 环境。没有 GPU 或 NCCL 时，先不要从 GPU benchmark 入手，先跑 MVP demo 和单元测试。

## 17. 重新上手时最应该读的文件

推荐按这个顺序：

1. `examples/run_two_stage_demo.py`
2. `pd_disaggregation/core/request.py`
3. `pd_disaggregation/core/system_state.py`
4. `pd_disaggregation/core/topology.py`
5. `pd_disaggregation/core/cost_model.py`
6. `pd_disaggregation/core/router.py`
7. `pd_disaggregation/core/transfer_plan.py`
8. `tests/test_router.py`
9. `tests/test_cost_model.py`
10. `docs/old_README.md`

如果只看一个文件，先看 `examples/run_two_stage_demo.py`。它把整个主线流程串起来了。

## 18. 当前实现范围和下一步

已经实现：

- 纯 Python 两阶段 routing API。
- 可解释的 SLO cost model。
- 默认拓扑候选。
- KV metadata 到 transfer bytes 的估算。
- profile-driven transfer latency 估算。
- transfer plan 生成。
- simulated transfer engine。
- demo 和单元测试。

尚未实现：

- 真实 `nano-vLLM` scheduler hook。
- 真实 KV cache 分配、移动和释放。
- 真实 NCCL communicator 生命周期管理。
- 基于线上观测数据更新 cost model。
- 基于真实 batch/remaining-token 状态的 decode queue 估算。

建议下一步：

1. 把真实 scheduler 的请求对象映射到 `Request`。
2. 把真实队列、worker、GPU 状态映射到 `SystemState`。
3. 在 prefill admission 前调用 `select_prefill_route()`。
4. 在 prefill 完成、KV metadata 可用后调用 `select_decode_route()`。
5. 用真实 profiling 数据替换默认 `TransferProfiler`。
6. 把 `TransferEngine` 替换为真实 KV handoff 后端。
7. 保留现有单元测试，并为真实集成层增加端到端测试。
