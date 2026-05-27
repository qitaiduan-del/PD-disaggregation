#!/usr/bin/env python3
"""
Approximate concurrent PD-serving benchmark with real repeated KV-transfer waves.

Request-filtered transfer
-------------------------
Each request triggers only the transfer edges belonging to its own prefill DP
and, for grouped-lane transfer, its own lane.

  cross_tp:
    A request from one P-DP transfers that P-DP's KV shards to the cross-TP
    decode group.

  aligned_lane_grouped:
    A request from P-DP i transfers only through grouped lane i.

Gloo logic test
---------------
torchrun --standalone --nproc_per_node=8 \
  scripts/bench_pd_concurrent.py \
  --backend gloo \
  --policy dynamic \
  --num-requests 16 \
  --arrival-pattern burst \
  --validate-first \
  --output-dir results/concurrent/dynamic

NCCL run
--------
NCCL_DEBUG=WARN \
NCCL_IB_DISABLE=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
TORCH_NCCL_BLOCKING_WAIT=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  scripts/bench_pd_concurrent.py \
  --backend nccl \
  --policy dynamic \
  --num-requests 64 \
  --arrival-pattern burst \
  --profile-summary results/transfer_profile_summary.csv \
  --output-dir results/concurrent/nccl/dynamic_64_pack
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

if "--print-topologies" in sys.argv:
    from pd_topology import format_topology_table

    print(format_topology_table())
    raise SystemExit(0)

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from pd_topology import add_topology_args, apply_topology_preset, topology_summary_dict

# Running this file from scripts/ puts scripts/ on sys.path, so the helper
# module resolves as a direct import.
from bench_hetero_kv_transfer import (  # type: ignore
    TransferHandle,
    allocate_local_buffers,
    barrier,
    build_handle_bundle,
    build_messages,
    dtype_from_name,
    execute_once_handle,
    parse_active_prefill_dps,
    setup_dist,
    validate_recv_buffers,
)


# =============================================================================
# Data models
# =============================================================================


@dataclass(frozen=True)
class Request:
    req_id: int
    arrival_ms: float
    input_len: int
    output_len: int


@dataclass
class RequestResult:
    req_id: int
    arrival_ms: float
    input_len: int
    output_len: int
    prefill_dp: int
    policy: str
    selected_handle: str
    decode_lane: int
    prefill_start_ms: float
    prefill_end_ms: float
    transfer_start_ms: float
    transfer_end_ms: float
    decode_start_ms: float
    first_decode_token_ms: float
    decode_end_ms: float
    prefill_queue_wait_ms: float
    transfer_queue_wait_ms: float
    decode_queue_wait_ms: float
    actual_transfer_ms: float
    decode_ms_per_token: float
    ttft_ms: float
    handoff_gap_ms: float
    e2e_ms: float
    itl_proxy_ms: float
    ttft_slo_violation: bool
    itl_slo_violation: bool


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrent PD benchmark with real repeated KV transfers.")

    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument(
        "--policy",
        choices=["cross_tp", "aligned_lane_grouped", "dynamic"],
        default="dynamic",
        help="Routing policy for requests.",
    )
    parser.add_argument("--output-dir", type=str, default="results/concurrent/run")
    parser.add_argument("--profile-summary", type=str, default=None, help="Optional results/transfer_profile_summary.csv for initial transfer estimates.")

    # World/model layout.
    parser.add_argument("--prefill-dp", type=int, default=2)
    parser.add_argument("--prefill-tp", type=int, default=2)
    parser.add_argument("--decode-dp", type=int, default=1)
    parser.add_argument("--decode-tp", type=int, default=4)
    add_topology_args(parser)
    parser.add_argument("--active-prefill-dps", type=str, default="all")
    parser.add_argument("--requests-per-prefill-dp", type=int, default=1)

    # KV shape.
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--chunk-tokens", type=int, default=4096)
    parser.add_argument(
        "--chunk-mode",
        choices=["packed", "fixed"],
        default="packed",
        help="packed: chunk_tokens=input_len for each request; fixed: use --chunk-tokens.",
    )

    # Workload.
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--arrival-pattern", choices=["burst", "fixed", "poisson"], default="burst")
    parser.add_argument("--arrival-rate-rps", type=float, default=35.0)
    parser.add_argument("--burst-size", type=int, default=8)
    parser.add_argument("--burst-interval-ms", type=float, default=180.0)
    parser.add_argument("--input-lens", type=str, default="1024,2048,4096,8192")
    parser.add_argument("--input-probs", type=str, default="0.35,0.30,0.25,0.10")
    parser.add_argument("--output-lens", type=str, default="64,128,256")
    parser.add_argument("--output-probs", type=str, default="0.50,0.35,0.15")

    # Queue/latency models.
    parser.add_argument("--prefill-base-ms", type=float, default=4.0)
    parser.add_argument("--prefill-ms-per-1k", type=float, default=32.0)
    parser.add_argument("--prefill-jitter-ratio", type=float, default=0.08)
    parser.add_argument("--decode-tp4-ms", type=float, default=7.5)
    parser.add_argument("--decode-tp2-ms", type=float, default=11.0)
    parser.add_argument("--ttft-slo-ms", type=float, default=250.0)
    parser.add_argument("--itl-slo-ms", type=float, default=80.0)
    parser.add_argument("--slo-penalty", type=float, default=5.0)

    # Dynamic policy.
    parser.add_argument("--dynamic-mode", choices=["cost", "threshold"], default="cost")
    parser.add_argument("--dynamic-seq-threshold", type=int, default=4096)
    parser.add_argument(
        "--dynamic-allow-any-aligned-lane",
        action="store_true",
        help=(
            "Experimental only. Not supported in this request-filtered transfer graph, because "
            "P-DP0->D-DP1 and P-DP1->D-DP0 cross-lane grouped edges are not prebuilt."
        ),
    )

    # Benchmark behavior.
    parser.add_argument("--mode", choices=["sync", "async", "threaded-sync"], default="async")
    parser.add_argument("--warmup-per-shape", type=int, default=0)
    parser.add_argument("--validate-first", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--print-every", type=int, default=8)

    args = parser.parse_args()
    apply_topology_preset(args)
    return args


# =============================================================================
# Utility
# =============================================================================


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    total = sum(vals)
    if total <= 0:
        raise ValueError("Probability list must have positive sum.")
    return [v / total for v in vals]


def percentile(values: Iterable[float], q: float) -> float:
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def mean(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def prefill_ms(req: Request, args: argparse.Namespace, rng: random.Random) -> float:
    base = args.prefill_base_ms + args.prefill_ms_per_1k * (req.input_len / 1024.0)
    jitter = 1.0 + rng.uniform(-args.prefill_jitter_ratio, args.prefill_jitter_ratio)
    return max(0.0, base * jitter)


def make_child_args(args: argparse.Namespace, seq_len: int, chunk_tokens: int) -> argparse.Namespace:
    """Build an args object compatible with build_messages()."""
    d = vars(args).copy()
    d["seq_len"] = seq_len
    d["chunk_tokens"] = chunk_tokens
    return argparse.Namespace(**d)


def request_chunk_tokens(req: Request, args: argparse.Namespace) -> int:
    if args.chunk_mode == "packed":
        return req.input_len
    return min(args.chunk_tokens, req.input_len)


def prefill_rank(dp_id: int, tp_id: int, args: argparse.Namespace) -> int:
    if dp_id < 0 or dp_id >= args.prefill_dp:
        raise ValueError(f"Invalid prefill dp_id={dp_id}; valid range is [0, {args.prefill_dp}).")
    if tp_id < 0 or tp_id >= args.prefill_tp:
        raise ValueError(f"Invalid prefill tp_id={tp_id}; valid range is [0, {args.prefill_tp}).")
    return dp_id * args.prefill_tp + tp_id


def decode_rank(dp_id: int, tp_id: int, args: argparse.Namespace) -> int:
    if dp_id < 0 or dp_id >= args.decode_dp:
        raise ValueError(f"Invalid decode dp_id={dp_id}; valid range is [0, {args.decode_dp}).")
    if tp_id < 0 or tp_id >= args.decode_tp:
        raise ValueError(f"Invalid decode tp_id={tp_id}; valid range is [0, {args.decode_tp}).")
    decode_base = args.prefill_dp * args.prefill_tp
    return decode_base + dp_id * args.decode_tp + tp_id


def ranks_for_prefill_dp(request_dp: int, args: argparse.Namespace) -> Tuple[int, ...]:
    return tuple(prefill_rank(request_dp, tp_id, args) for tp_id in range(args.prefill_tp))


def ranks_for_cross_route(request_dp: int, args: argparse.Namespace) -> Tuple[int, ...]:
    if args.decode_dp <= 0 or args.decode_tp <= 0:
        raise ValueError("Invalid cross decode layout.")
    prefill_ranks = list(ranks_for_prefill_dp(request_dp, args))
    decode_ranks = [decode_rank(dp_id, tp_id, args) for dp_id in range(args.decode_dp) for tp_id in range(args.decode_tp)]
    return tuple(prefill_ranks + decode_ranks)


def ranks_for_grouped_lane(lane_id: int, args: argparse.Namespace) -> Tuple[int, ...]:
    if args.grouped_decode_dp != args.prefill_dp or args.grouped_decode_tp != args.prefill_tp:
        raise ValueError(
            "Route-specific grouped lane groups require grouped decode DP/TP to match prefill DP/TP. "
            f"Got grouped_decode_dp={args.grouped_decode_dp}, grouped_decode_tp={args.grouped_decode_tp}, "
            f"prefill_dp={args.prefill_dp}, prefill_tp={args.prefill_tp}."
        )
    if lane_id < 0 or lane_id >= args.prefill_dp:
        raise ValueError(f"Invalid grouped lane_id={lane_id}; valid range is [0, {args.prefill_dp}).")

    decode_base = args.prefill_dp * args.prefill_tp
    prefill_ranks = [prefill_rank(lane_id, tp_id, args) for tp_id in range(args.prefill_tp)]
    decode_ranks = [decode_base + lane_id * args.grouped_decode_tp + tp_id for tp_id in range(args.grouped_decode_tp)]
    return tuple(prefill_ranks + decode_ranks)


# =============================================================================
# Workload and transfer estimates
# =============================================================================


def generate_workload(args: argparse.Namespace) -> List[Request]:
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    input_lens = parse_int_list(args.input_lens)
    input_probs = parse_float_list(args.input_probs)
    output_lens = parse_int_list(args.output_lens)
    output_probs = parse_float_list(args.output_probs)

    reqs: List[Request] = []
    t = 0.0
    for i in range(args.num_requests):
        if args.arrival_pattern == "fixed":
            t = i * (1000.0 / args.arrival_rate_rps)
        elif args.arrival_pattern == "poisson":
            if i == 0:
                t = 0.0
            else:
                t += float(np_rng.exponential(1000.0 / args.arrival_rate_rps))
        elif args.arrival_pattern == "burst":
            burst_id = i // args.burst_size
            offset = i % args.burst_size
            t = burst_id * args.burst_interval_ms + offset * 1.0
        else:
            raise ValueError(args.arrival_pattern)

        reqs.append(
            Request(
                req_id=i,
                arrival_ms=t,
                input_len=int(rng.choices(input_lens, weights=input_probs, k=1)[0]),
                output_len=int(rng.choices(output_lens, weights=output_probs, k=1)[0]),
            )
        )
    return reqs


class TransferEstimator:
    """Transfer-time estimator used by the dynamic policy.

    It loads optional results/transfer_profile_summary.csv profile points, then maintains an EMA
    from measured transfers. Estimates are used only for routing; actual transfer
    time is measured by the real P2P wave.
    """

    def __init__(self, args: argparse.Namespace):
        self.default_gbps = 110.0
        self.alpha = 0.30
        self.ema: Dict[Tuple[str, int], float] = {}
        if args.profile_summary:
            self._load_profile(Path(args.profile_summary))

    def _load_profile(self, path: Path) -> None:
        if not path.exists():
            return
        df = pd.read_csv(path)
        required = {"handle_arg", "mode", "seq_len", "chunk_tokens", "num_layers", "avg_ms"}
        if not required.issubset(df.columns):
            return

        for col in ["seq_len", "chunk_tokens", "num_layers", "avg_ms"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        work = df[
            (df["mode"] == "async")
            & (df["handle_arg"].isin(["cross_tp", "aligned_lane", "aligned_lane_grouped"]))
            & (df["seq_len"] == df["chunk_tokens"])
            & (df["num_layers"] == 32)
            & df["avg_ms"].notna()
        ]

        for _, r in work.iterrows():
            handle = str(r["handle_arg"])
            seq_len = int(r["seq_len"])
            avg_ms = float(r["avg_ms"])
            self.ema[(handle, seq_len)] = avg_ms
            # If no grouped profile exists yet, aligned_lane is the closest fallback.
            if handle == "aligned_lane":
                self.ema[("aligned_lane_grouped", seq_len)] = avg_ms

    def estimate_from_shape(self, handle: str, seq_len: int, args: argparse.Namespace) -> float:
        del handle
        dtype_bytes = torch.tensor([], dtype=dtype_from_name(args.dtype)).element_size()
        total_bytes = args.requests_per_prefill_dp * args.num_layers * 2 * seq_len * args.num_kv_heads * args.head_dim * dtype_bytes
        return total_bytes / (self.default_gbps * 1e9) * 1000.0

    def estimate(self, handle: str, seq_len: int, args: argparse.Namespace) -> float:
        key = (handle, seq_len)
        if key in self.ema:
            return self.ema[key]

        pts = sorted((s, v) for (h, s), v in self.ema.items() if h == handle)
        if len(pts) >= 2:
            xs = np.array([p[0] for p in pts], dtype=np.float64)
            ys = np.array([p[1] for p in pts], dtype=np.float64)
            if seq_len <= xs[0]:
                return float(ys[0] * seq_len / xs[0])
            if seq_len >= xs[-1]:
                slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
                return float(max(0.0, ys[-1] + slope * (seq_len - xs[-1])))
            return float(np.interp(seq_len, xs, ys))

        return self.estimate_from_shape(handle, seq_len, args)

    def update(self, handle: str, seq_len: int, measured_ms: float) -> None:
        key = (handle, seq_len)
        if key not in self.ema:
            self.ema[key] = measured_ms
        else:
            self.ema[key] = (1.0 - self.alpha) * self.ema[key] + self.alpha * measured_ms


# =============================================================================
# Transfer execution
# =============================================================================


class TransferExecutor:
    """Executes request-filtered transfer waves.

    This class is intentionally stricter than the raw transfer benchmark:
    it filters edges/messages so that a request only transfers its own prefill-DP
    KV cache shards. That makes the repeated-transfer benchmark closer to real
    request-level PD handoff.
    """

    def __init__(self, args: argparse.Namespace, rank: int, device: torch.device, base_handles: Dict[str, TransferHandle]):
        self.args = args
        self.rank = rank
        self.device = device
        self.dtype = dtype_from_name(args.dtype)
        self.base_handles = base_handles
        self.route_groups: Dict[Tuple[str, int, int], dist.ProcessGroup] = {}
        self.route_group_ranks: Dict[Tuple[str, int, int], Tuple[int, ...]] = {}
        self.cache: Dict[
            Tuple[str, int, int, int, int],
            Tuple[TransferHandle, Dict[int, torch.Tensor], Dict[int, torch.Tensor]],
        ] = {}
        self.warmed: set[Tuple[str, int, int, int, int]] = set()
        self._build_route_groups()

    def _build_route_groups(self) -> None:
        # All ranks must call new_group in this exact order, including ranks
        # outside a specific route group.
        for request_dp in range(self.args.prefill_dp):
            key = ("cross_tp", request_dp, -1)
            ranks = ranks_for_cross_route(request_dp, self.args)
            self.route_group_ranks[key] = ranks
            self.route_groups[key] = dist.new_group(ranks=list(ranks), backend=self.args.backend)

        for lane_id in range(self.args.prefill_dp):
            key = ("aligned_lane_grouped", lane_id, lane_id)
            ranks = ranks_for_grouped_lane(lane_id, self.args)
            self.route_group_ranks[key] = ranks
            self.route_groups[key] = dist.new_group(ranks=list(ranks), backend=self.args.backend)

        barrier(self.device)

    def _route_key(self, handle_name: str, request_dp: int, lane_id: int) -> Tuple[str, int, int]:
        if handle_name == "cross_tp":
            return ("cross_tp", request_dp, -1)
        if handle_name == "aligned_lane_grouped":
            if lane_id != request_dp:
                raise RuntimeError(
                    "Request-filtered aligned_lane_grouped transfer requires lane_id == request_dp. "
                    f"Got request_dp={request_dp}, lane_id={lane_id}."
                )
            return ("aligned_lane_grouped", request_dp, lane_id)
        raise ValueError(f"Unsupported handle in concurrent benchmark: {handle_name}")

    def _filter_edges_for_request(
        self,
        handle_name: str,
        base: TransferHandle,
        request_dp: int,
        lane_id: int,
    ):
        if handle_name == "cross_tp":
            # One request belongs to one prefill DP, but may fan out to all decode TP ranks.
            return [edge for edge in base.edges if edge.request_dp == request_dp]

        if handle_name == "aligned_lane_grouped":
            # Grouped lane transfer should only use the selected lane. In the current
            # graph, lane_id == request_dp.
            return [edge for edge in base.edges if edge.request_dp == request_dp and edge.lane_id == lane_id]

        raise ValueError(f"Unsupported handle in concurrent benchmark: {handle_name}")

    def get_transfer_objects(
        self,
        handle_name: str,
        seq_len: int,
        chunk_tokens: int,
        request_dp: int,
        lane_id: int,
    ) -> Tuple[TransferHandle, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        key = (handle_name, seq_len, chunk_tokens, request_dp, lane_id)
        if key in self.cache:
            return self.cache[key]

        base = self.base_handles[handle_name]
        route_key = self._route_key(handle_name, request_dp, lane_id)
        if route_key not in self.route_groups:
            raise RuntimeError(f"No route-specific process group for route={route_key}.")

        filtered_edges = self._filter_edges_for_request(handle_name, base, request_dp, lane_id)
        if not filtered_edges:
            raise RuntimeError(
                f"No transfer edges for handle={handle_name}, request_dp={request_dp}, lane_id={lane_id}. "
                "If you enabled --dynamic-allow-any-aligned-lane, disable it unless cross-lane grouped edges are prebuilt."
            )

        msg_args = make_child_args(self.args, seq_len=seq_len, chunk_tokens=chunk_tokens)
        messages = build_messages(msg_args, filtered_edges, self.dtype)

        handle = replace(
            base,
            group_ranks=self.route_group_ranks[route_key],
            edges=filtered_edges,
            messages=messages,
            process_group=self.route_groups[route_key],
            lane_groups={},
            lane_group_ranks={},
        )
        send_bufs, recv_bufs = allocate_local_buffers(
            self.rank,
            messages,
            self.dtype,
            self.device,
            validate=self.args.validate_first,
        )

        self.cache[key] = (handle, send_bufs, recv_bufs)
        return self.cache[key]

    def run_once(
        self,
        handle_name: str,
        seq_len: int,
        chunk_tokens: int,
        request_dp: int,
        lane_id: int,
        do_validate: bool = False,
    ) -> float:
        handle, send_bufs, recv_bufs = self.get_transfer_objects(
            handle_name=handle_name,
            seq_len=seq_len,
            chunk_tokens=chunk_tokens,
            request_dp=request_dp,
            lane_id=lane_id,
        )

        key = (handle_name, seq_len, chunk_tokens, request_dp, lane_id)
        if self.args.warmup_per_shape > 0 and key not in self.warmed:
            for _ in range(self.args.warmup_per_shape):
                execute_once_handle(self.args.mode, self.rank, handle, send_bufs, recv_bufs)
                barrier(self.device)
            self.warmed.add(key)

        barrier(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            execute_once_handle(self.args.mode, self.rank, handle, send_bufs, recv_bufs)
            end.record()
            torch.cuda.synchronize(self.device)
            local_ms = float(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            execute_once_handle(self.args.mode, self.rank, handle, send_bufs, recv_bufs)
            local_ms = (time.perf_counter() - t0) * 1000.0

        t = torch.tensor([local_ms], dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        global_ms = float(t.item())

        if do_validate:
            validate_recv_buffers(self.rank, handle.messages, recv_bufs, self.device)

        barrier(self.device)
        return global_ms


# =============================================================================
# Policy simulation plus real transfer
# =============================================================================


def choose_prefill_dp(prefill_available: Sequence[float]) -> int:
    return int(min(range(len(prefill_available)), key=lambda i: prefill_available[i]))


def evaluate_candidate(
    req: Request,
    prefill_end: float,
    handle: str,
    lane: int,
    estimator: TransferEstimator,
    args: argparse.Namespace,
    cross_transfer_available: float,
    grouped_transfer_available: Sequence[float],
    cross_decode_available: float,
    grouped_decode_available: Sequence[float],
) -> float:
    transfer_est = estimator.estimate(handle, req.input_len, args)
    if handle == "cross_tp":
        t_start = max(prefill_end, cross_transfer_available)
        t_end = t_start + transfer_est
        d_start = max(t_end, cross_decode_available)
        decode_ms = args.decode_tp4_ms
    elif handle == "aligned_lane_grouped":
        t_start = max(prefill_end, grouped_transfer_available[lane])
        t_end = t_start + transfer_est
        d_start = max(t_end, grouped_decode_available[lane])
        decode_ms = args.decode_tp2_ms
    else:
        raise ValueError(handle)

    first_decode = d_start + decode_ms
    decode_end = d_start + req.output_len * decode_ms
    handoff_gap = first_decode - prefill_end
    e2e = decode_end - req.arrival_ms

    penalty = 0.0
    if handoff_gap > args.itl_slo_ms:
        penalty += args.slo_penalty * (handoff_gap - args.itl_slo_ms)
    return e2e + penalty


def summarize(results: List[RequestResult], policy: str) -> Dict[str, object]:
    return {
        "policy": policy,
        "num_requests": len(results),
        "mean_ttft_ms": mean(r.ttft_ms for r in results),
        "p90_ttft_ms": percentile((r.ttft_ms for r in results), 90),
        "p99_ttft_ms": percentile((r.ttft_ms for r in results), 99),
        "mean_handoff_gap_ms": mean(r.handoff_gap_ms for r in results),
        "p90_handoff_gap_ms": percentile((r.handoff_gap_ms for r in results), 90),
        "p99_handoff_gap_ms": percentile((r.handoff_gap_ms for r in results), 99),
        "mean_decode_queue_wait_ms": mean(r.decode_queue_wait_ms for r in results),
        "p90_decode_queue_wait_ms": percentile((r.decode_queue_wait_ms for r in results), 90),
        "p99_decode_queue_wait_ms": percentile((r.decode_queue_wait_ms for r in results), 99),
        "mean_e2e_ms": mean(r.e2e_ms for r in results),
        "p90_e2e_ms": percentile((r.e2e_ms for r in results), 90),
        "p99_e2e_ms": percentile((r.e2e_ms for r in results), 99),
        "mean_actual_transfer_ms": mean(r.actual_transfer_ms for r in results),
        "ttft_slo_violation_rate": mean(float(r.ttft_slo_violation) for r in results),
        "itl_slo_violation_rate": mean(float(r.itl_slo_violation) for r in results),
        "cross_tp_fraction": mean(float(r.selected_handle == "cross_tp") for r in results),
        "grouped_fraction": mean(float(r.selected_handle == "aligned_lane_grouped") for r in results),
        "makespan_ms": max(r.decode_end_ms for r in results) - min(r.arrival_ms for r in results),
        "request_level_filtering": True,
    }


def format_route_group_ranks(route_group_ranks: Dict[Tuple[str, int, int], Tuple[int, ...]]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for (handle_name, request_dp, lane_id), ranks in sorted(route_group_ranks.items()):
        if handle_name == "cross_tp":
            key = f"cross_tp_dp{request_dp}"
        else:
            key = f"aligned_lane_grouped_lane{lane_id}"
        out[key] = list(ranks)
    return out


def run_benchmark(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> Tuple[List[RequestResult], Dict[str, object], Dict[str, List[int]]]:
    if args.dynamic_allow_any_aligned_lane:
        raise RuntimeError(
            "--dynamic-allow-any-aligned-lane is disabled for this request-filtered transfer graph. "
            "The current aligned_lane_grouped handle only has P-DP_i->lane_i edges."
        )

    expected_world_size = args.expected_world_size
    if world_size != expected_world_size:
        raise RuntimeError(f"WORLD_SIZE={world_size}; expected {expected_world_size}.")

    dtype = dtype_from_name(args.dtype)
    active_dps = parse_active_prefill_dps(args.active_prefill_dps, args.prefill_dp)

    # Grouped lane execution uses one process group per lane.
    args.use_process_groups = True
    # build_handle_bundle expects seq_len/chunk_tokens fields even though messages
    # are rebuilt per request.
    if not hasattr(args, "seq_len"):
        args.seq_len = max(parse_int_list(args.input_lens))

    bundle = build_handle_bundle(args, active_dps, dtype, device)
    required = {"cross_tp", "aligned_lane_grouped"}
    missing = required - set(bundle.handles)
    if missing:
        raise RuntimeError(f"bench_hetero_kv_transfer.py does not provide required handles: {missing}")

    executor = TransferExecutor(args, rank, device, bundle.handles)
    route_group_ranks = format_route_group_ranks(executor.route_group_ranks)

    if rank == 0:
        print("=" * 100)
        print("Concurrent PD benchmark with request-filtered real KV-transfer waves")
        print("=" * 100)
        print(f"policy             : {args.policy}")
        print(f"backend/mode       : {args.backend} / {args.mode}")
        print(f"num_requests       : {args.num_requests}")
        print(f"arrival_pattern    : {args.arrival_pattern}")
        print(f"topology           : {json.dumps(topology_summary_dict(args), sort_keys=True)}")
        print(f"input_lens         : {args.input_lens}")
        print(f"output_lens        : {args.output_lens}")
        print(f"chunk_mode         : {args.chunk_mode}, chunk_tokens={args.chunk_tokens}")
        print("request filtering  : enabled")
        print("route async groups : enabled")
        for key, ranks in route_group_ranks.items():
            if key.startswith("cross_tp_dp"):
                print(f"route group cross_tp dp{key.removeprefix('cross_tp_dp')}: {ranks}")
            else:
                print(f"route group grouped lane{key.removeprefix('aligned_lane_grouped_lane')}: {ranks}")
        print(f"handles            : {json.dumps(bundle.summary(), indent=2)}")
        print("=" * 100)

    requests = generate_workload(args)
    estimator = TransferEstimator(args)
    rng = random.Random(args.seed + 123)

    # Validate both handle paths once on the smallest request shape.
    if args.validate_first:
        min_len = min(r.input_len for r in requests)
        chunk = min_len if args.chunk_mode == "packed" else min(args.chunk_tokens, min_len)
        validate_routes: List[Tuple[str, int, int]] = [
            ("cross_tp", 0, 0),
            ("cross_tp", args.prefill_dp - 1, 0),
            ("aligned_lane_grouped", 0, 0),
            ("aligned_lane_grouped", args.prefill_dp - 1, args.prefill_dp - 1),
        ]
        seen_validate_routes: set[Tuple[str, int, int]] = set()
        for handle_name, request_dp, lane_id in validate_routes:
            route = (handle_name, request_dp, lane_id)
            if route in seen_validate_routes:
                continue
            seen_validate_routes.add(route)
            executor.run_once(handle_name, min_len, chunk, request_dp=request_dp, lane_id=lane_id, do_validate=True)
        if rank == 0:
            print(f"validation         : passed for request-filtered routes {sorted(seen_validate_routes)}")

    prefill_available = [0.0 for _ in range(args.prefill_dp)]
    cross_transfer_available = 0.0
    grouped_transfer_available = [0.0 for _ in range(args.prefill_dp)]
    cross_decode_available = 0.0
    grouped_decode_available = [0.0 for _ in range(args.prefill_dp)]

    results: List[RequestResult] = []

    for req in sorted(requests, key=lambda r: (r.arrival_ms, r.req_id)):
        prefill_dp = choose_prefill_dp(prefill_available)
        p_start = max(req.arrival_ms, prefill_available[prefill_dp])
        p_end = p_start + prefill_ms(req, args, rng)
        prefill_available[prefill_dp] = p_end

        if args.policy == "cross_tp":
            selected_handle = "cross_tp"
            lane = 0
        elif args.policy == "aligned_lane_grouped":
            selected_handle = "aligned_lane_grouped"
            lane = prefill_dp
        elif args.policy == "dynamic":
            if args.dynamic_mode == "threshold":
                if req.input_len >= args.dynamic_seq_threshold:
                    selected_handle = "aligned_lane_grouped"
                    lane = prefill_dp
                else:
                    selected_handle = "cross_tp"
                    lane = 0
            else:
                candidates: List[Tuple[float, str, int]] = []
                candidates.append(
                    (
                        evaluate_candidate(
                            req,
                            p_end,
                            "cross_tp",
                            0,
                            estimator,
                            args,
                            cross_transfer_available,
                            grouped_transfer_available,
                            cross_decode_available,
                            grouped_decode_available,
                        ),
                        "cross_tp",
                        0,
                    )
                )
                # For the current request-filtered graph, grouped lane is tied to prefill_dp.
                candidates.append(
                    (
                        evaluate_candidate(
                            req,
                            p_end,
                            "aligned_lane_grouped",
                            prefill_dp,
                            estimator,
                            args,
                            cross_transfer_available,
                            grouped_transfer_available,
                            cross_decode_available,
                            grouped_decode_available,
                        ),
                        "aligned_lane_grouped",
                        prefill_dp,
                    )
                )
                _, selected_handle, lane = min(candidates, key=lambda x: x[0])
        else:
            raise ValueError(args.policy)

        chunk = request_chunk_tokens(req, args)
        actual_transfer_ms = executor.run_once(
            selected_handle,
            req.input_len,
            chunk,
            request_dp=prefill_dp,
            lane_id=lane,
            do_validate=False,
        )
        estimator.update(selected_handle, req.input_len, actual_transfer_ms)

        if selected_handle == "cross_tp":
            t_start = max(p_end, cross_transfer_available)
            t_end = t_start + actual_transfer_ms
            cross_transfer_available = t_end
            d_start = max(t_end, cross_decode_available)
            decode_ms = args.decode_tp4_ms
            d_end = d_start + req.output_len * decode_ms
            cross_decode_available = d_end
        else:
            t_start = max(p_end, grouped_transfer_available[lane])
            t_end = t_start + actual_transfer_ms
            grouped_transfer_available[lane] = t_end
            d_start = max(t_end, grouped_decode_available[lane])
            decode_ms = args.decode_tp2_ms
            d_end = d_start + req.output_len * decode_ms
            grouped_decode_available[lane] = d_end

        first_decode = d_start + decode_ms
        ttft = p_end - req.arrival_ms
        handoff_gap = first_decode - p_end
        itl_proxy = max(handoff_gap, decode_ms)

        result = RequestResult(
            req_id=req.req_id,
            arrival_ms=req.arrival_ms,
            input_len=req.input_len,
            output_len=req.output_len,
            prefill_dp=prefill_dp,
            policy=args.policy,
            selected_handle=selected_handle,
            decode_lane=lane,
            prefill_start_ms=p_start,
            prefill_end_ms=p_end,
            transfer_start_ms=t_start,
            transfer_end_ms=t_end,
            decode_start_ms=d_start,
            first_decode_token_ms=first_decode,
            decode_end_ms=d_end,
            prefill_queue_wait_ms=p_start - req.arrival_ms,
            transfer_queue_wait_ms=t_start - p_end,
            decode_queue_wait_ms=d_start - t_end,
            actual_transfer_ms=actual_transfer_ms,
            decode_ms_per_token=decode_ms,
            ttft_ms=ttft,
            handoff_gap_ms=handoff_gap,
            e2e_ms=d_end - req.arrival_ms,
            itl_proxy_ms=itl_proxy,
            ttft_slo_violation=ttft > args.ttft_slo_ms,
            itl_slo_violation=itl_proxy > args.itl_slo_ms,
        )
        results.append(result)

        if rank == 0 and (req.req_id % args.print_every == 0 or req.req_id == args.num_requests - 1):
            print(
                f"req={req.req_id:04d} input={req.input_len} output={req.output_len} "
                f"pdp={prefill_dp} handle={selected_handle} lane={lane} "
                f"transfer={actual_transfer_ms:.3f}ms decode_q={result.decode_queue_wait_ms:.3f}ms "
                f"handoff={handoff_gap:.3f}ms e2e={result.e2e_ms:.3f}ms"
            )

    summary = summarize(results, args.policy)
    return results, summary, route_group_ranks


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    rank, world_size, device_id, device = setup_dist(args.backend)
    del device_id

    try:
        results, summary, route_group_ranks = run_benchmark(args, rank, world_size, device)
        barrier(device)

        if rank == 0:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            trace_path = out_dir / f"trace_{args.policy}.csv"
            summary_path = out_dir / f"summary_{args.policy}.json"
            pd.DataFrame([asdict(r) for r in results]).to_csv(trace_path, index=False)
            payload = {
                "args": vars(args),
                "topology": topology_summary_dict(args),
                "request_level_filtering": True,
                "route_specific_process_groups": True,
                "route_group_ranks": route_group_ranks,
                "summary": summary,
                "notes": {
                    "request_level_filtering": True,
                    "route_specific_process_groups": True,
                    "meaning": "Each request transfers only the edges belonging to its prefill_dp and selected lane.",
                },
            }
            summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            print("\n=== Concurrent PD benchmark summary ===")
            for k, v in summary.items():
                if isinstance(v, float):
                    print(f"{k:32s}: {v:.4f}")
                else:
                    print(f"{k:32s}: {v}")
            print(f"saved_trace                    : {trace_path}")
            print(f"saved_summary                  : {summary_path}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
