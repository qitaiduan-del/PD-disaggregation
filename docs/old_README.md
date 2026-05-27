# PD Disaggregation Experiments

该目录是独立于上层 `mini_platform` 路由示例的 PD（prefill/decode）分离实验项目，包含异构 KV 传输、并发调度模拟、拓扑验证和 GPU 全量运行脚本。

## 目录结构

```text
pd_disaggregation/
  scripts/
    pd_topology.py                 # 拓扑预设和校验
    bench_hetero_kv_transfer.py    # KV 传输微基准
    bench_pd_concurrent.py         # 并发 PD benchmark
    simulate_pd_concurrent.py      # 基于传输 profile 的模拟
    analyze_pd_results.py          # 汇总和绘图
    check_topology_sanity.sh       # Gloo/NCCL sanity check
    run_gpu_pd_topology_full.sh    # 8 GPU 全量流程
  results/
    transfer_profile_summary.csv   # 已有传输 profile 汇总
    concurrent/                    # 按策略或 NCCL 实验保存的并发结果
    simulation/                    # 模拟结果
    sanity/                        # 拓扑 sanity 结果
```

`results/` 和运行产生的 `logs/` 由仓库根目录的 `.gitignore` 排除，不与源代码混在版本历史中。

## 运行位置

以下命令均从本目录执行：

```bash
cd pd_disaggregation
python -m py_compile scripts/*.py
python scripts/simulate_pd_concurrent.py
python scripts/bench_pd_concurrent.py --print-topologies
bash scripts/check_topology_sanity.sh
```

GPU 全量实验需要 8 张可见 GPU：

```bash
bash scripts/run_gpu_pd_topology_full.sh
```

## 结果命名

现有的临时输出已按含义整理：

```text
results/concurrent/dynamic/
results/concurrent/cross_tp/
results/concurrent/aligned_lane_grouped/
results/simulation/legacy_aligned_lane/
results/simulation/grouped_lane/
results/sanity/gloo/<topology>/
```

其中 `legacy_aligned_lane` 保留早期 aligned-lane 策略结果，`grouped_lane` 对应后续 grouped-lane 对比，不能互相覆盖。
