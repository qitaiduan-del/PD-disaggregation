#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib

mkdir -p results/transfer results/concurrent/nccl results/report_assets/figures logs
mkdir -p "${MPLCONFIGDIR}"

echo "Step 0. Environment check"
nvidia-smi
python - <<'PY'
import torch

print("torch version:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"device {i}: {torch.cuda.get_device_name(i)}")
PY

echo "Step 1. py_compile"
python -m py_compile \
  scripts/pd_topology.py \
  scripts/bench_hetero_kv_transfer.py \
  scripts/bench_pd_concurrent.py \
  scripts/simulate_pd_concurrent.py \
  scripts/analyze_pd_results.py

echo "Step 2. Topology presets"
python scripts/bench_pd_concurrent.py --print-topologies

echo "Step 3. NCCL all_reduce sanity"
cat > /tmp/test_nccl.py <<'PY'
import os

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
x = torch.tensor([float(rank)], device=f"cuda:{local_rank}")
dist.all_reduce(x, op=dist.ReduceOp.SUM)
if rank == 0:
    print(f"world_size: {world_size}")
    print(f"all_reduce result: {float(x.item())}")
dist.destroy_process_group()
PY
torchrun --standalone --nproc_per_node=8 /tmp/test_nccl.py | tee logs/nccl_all_reduce_sanity.log
grep -q "world_size: 8" logs/nccl_all_reduce_sanity.log
grep -q "all_reduce result: 28.0" logs/nccl_all_reduce_sanity.log

topologies=(p2t2_x1t4_g2t2 p4t1_x1t4_g4t1)
handles=(cross_tp aligned_lane_grouped)
policies=(cross_tp aligned_lane_grouped dynamic)

echo "Step 4. Grouped handle sanity"
for topo in "${topologies[@]}"; do
  torchrun --standalone --nproc_per_node=8 \
    scripts/bench_hetero_kv_transfer.py \
    --backend nccl \
    --topology "${topo}" \
    --use-process-groups \
    --handle aligned_lane_grouped \
    --mode async \
    --seq-len 128 \
    --num-layers 2 \
    --num-kv-heads 32 \
    --head-dim 128 \
    --chunk-tokens 128 \
    --warmup 1 \
    --iters 2 \
    --validate \
    --print-plan \
    --output-json "results/transfer/sanity_aligned_lane_grouped_${topo}.json" \
    | tee "logs/sanity_aligned_lane_grouped_${topo}.log"
done

echo "Step 5. Raw transfer calibration"
for topo in "${topologies[@]}"; do
  for handle in "${handles[@]}"; do
    for spec in "4096 4096" "4096 256" "8192 8192" "8192 256"; do
      read -r seq chunk <<< "${spec}"
      torchrun --standalone --nproc_per_node=8 \
        scripts/bench_hetero_kv_transfer.py \
        --backend nccl \
        --topology "${topo}" \
        --use-process-groups \
        --handle "${handle}" \
        --mode async \
        --seq-len "${seq}" \
        --num-layers 32 \
        --num-kv-heads 32 \
        --head-dim 128 \
        --chunk-tokens "${chunk}" \
        --warmup 3 \
        --iters 10 \
        --print-rank-times \
        --output-json "results/transfer/raw_${topo}_${handle}_s${seq}_c${chunk}.json" \
        | tee "logs/raw_${topo}_${handle}_s${seq}_c${chunk}.log"
    done
  done
done

echo "Step 6. Concurrent PD full experiment"
for topo in "${topologies[@]}"; do
  for workload in standard_64 heavy_128; do
    if [[ "${workload}" == "standard_64" ]]; then
      num_requests=64
      burst_size=8
      burst_interval_ms=180
    else
      num_requests=128
      burst_size=16
      burst_interval_ms=120
    fi

    for chunk_case in pack chunk256; do
      chunk_args=()
      if [[ "${chunk_case}" == "chunk256" ]]; then
        chunk_args=(--chunk-mode fixed --chunk-tokens 256)
      fi

      for policy in "${policies[@]}"; do
        out_dir="results/concurrent/nccl/${topo}/${workload}_${chunk_case}/${policy}"
        torchrun --standalone --nproc_per_node=8 \
          scripts/bench_pd_concurrent.py \
          --backend nccl \
          --topology "${topo}" \
          --policy "${policy}" \
          --mode async \
          --num-requests "${num_requests}" \
          --arrival-pattern burst \
          --burst-size "${burst_size}" \
          --burst-interval-ms "${burst_interval_ms}" \
          "${chunk_args[@]}" \
          --output-dir "${out_dir}" \
          | tee "logs/pd_${topo}_${workload}_${chunk_case}_${policy}.log"
      done
    done
  done
done

echo "Step 7. Summary table"
python - <<'PY'
import json
from pathlib import Path

import pandas as pd

rows = []
for summary_path in sorted(Path("results/concurrent/nccl").glob("*/*/*/summary_*.json")):
    policy_dir = summary_path.parent.name
    case_dir = summary_path.parent.parent.name
    topo_dir = summary_path.parent.parent.parent.name
    if case_dir.endswith("_chunk256"):
        workload = case_dir[:-len("_chunk256")]
        chunk_case = "chunk256"
    elif case_dir.endswith("_pack"):
        workload = case_dir[:-len("_pack")]
        chunk_case = "pack"
    else:
        workload = case_dir
        chunk_case = "unknown"

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    topology = payload.get("topology", {})
    args = payload.get("args", {})
    rows.append({
        "topology": topology.get("topology", args.get("topology_name", topo_dir)),
        "workload": workload,
        "chunk_case": chunk_case,
        "policy": summary.get("policy", args.get("policy", policy_dir)),
        "num_requests": summary.get("num_requests", args.get("num_requests")),
        "mean_actual_transfer_ms": summary.get("mean_actual_transfer_ms"),
        "p99_handoff_gap_ms": summary.get("p99_handoff_gap_ms"),
        "p99_decode_queue_wait_ms": summary.get("p99_decode_queue_wait_ms"),
        "p99_e2e_ms": summary.get("p99_e2e_ms"),
        "makespan_ms": summary.get("makespan_ms"),
        "cross_tp_fraction": summary.get("cross_tp_fraction"),
        "grouped_fraction": summary.get("grouped_fraction"),
        "request_level_filtering": payload.get("request_level_filtering", summary.get("request_level_filtering")),
    })

df = pd.DataFrame(rows)
policy_order = {"cross_tp": 0, "aligned_lane_grouped": 1, "dynamic": 2}
if not df.empty:
    df["_policy_order"] = df["policy"].map(policy_order)
    df = df.sort_values(["topology", "workload", "chunk_case", "_policy_order"]).drop(columns=["_policy_order"])
Path("results/report_assets").mkdir(parents=True, exist_ok=True)
df.to_csv("results/report_assets/pd_concurrent_summary.csv", index=False)
print(df.to_string(index=False))
PY

echo "Step 8. Figures and meeting report"
python scripts/analyze_pd_results.py --root . --output-dir results/report_assets --show-values

echo "Step 9. Package results"
tar -czf results/pd_all_results.tar.gz \
  results/transfer \
  results/concurrent/nccl \
  results/report_assets \
  logs \
  scripts/pd_topology.py \
  scripts/bench_hetero_kv_transfer.py \
  scripts/bench_pd_concurrent.py \
  scripts/simulate_pd_concurrent.py \
  scripts/analyze_pd_results.py \
  scripts/run_gpu_pd_topology_full.sh \
  scripts/check_topology_sanity.sh

echo "Done: results/pd_all_results.tar.gz"
