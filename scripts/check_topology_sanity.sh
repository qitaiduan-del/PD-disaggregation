#!/usr/bin/env bash
set -euo pipefail

python -m py_compile \
  scripts/pd_topology.py \
  scripts/bench_hetero_kv_transfer.py \
  scripts/bench_pd_concurrent.py \
  scripts/simulate_pd_concurrent.py \
  scripts/analyze_pd_results.py

run_gloo_sanity() {
  local topo="$1"
  local out_dir="results/sanity/gloo/${topo}"

  torchrun --standalone --nproc_per_node=8 \
    scripts/bench_pd_concurrent.py \
    --topology "${topo}" \
    --backend gloo \
    --policy dynamic \
    --mode async \
    --num-requests 16 \
    --arrival-pattern burst \
    --validate-first \
    --num-layers 1 \
    --num-kv-heads 4 \
    --head-dim 8 \
    --dtype float32 \
    --input-lens 16 \
    --input-probs 1.0 \
    --output-lens 4 \
    --output-probs 1.0 \
    --chunk-mode packed \
    --print-every 8 \
    --output-dir "${out_dir}"
}

run_gloo_sanity p2t2_x1t4_g2t2
run_gloo_sanity p4t1_x1t4_g4t1

if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 8 else 1)"; then
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

  torchrun --standalone --nproc_per_node=8 \
    scripts/bench_pd_concurrent.py \
    --backend nccl \
    --topology p2t2_x1t4_g2t2 \
    --policy cross_tp \
    --mode async \
    --num-requests 8 \
    --arrival-pattern burst \
    --burst-size 8 \
    --burst-interval-ms 180 \
    --validate-first \
    --output-dir results/concurrent/nccl/sanity_cross_async_route_group

  torchrun --standalone --nproc_per_node=8 \
    scripts/bench_pd_concurrent.py \
    --backend nccl \
    --topology p2t2_x1t4_g2t2 \
    --policy dynamic \
    --mode async \
    --num-requests 8 \
    --arrival-pattern burst \
    --burst-size 8 \
    --burst-interval-ms 180 \
    --validate-first \
    --output-dir results/concurrent/nccl/sanity_dynamic_async_route_group
fi
