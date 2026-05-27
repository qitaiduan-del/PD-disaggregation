#!/usr/bin/env python3
"""
Heterogeneous KV Cache Transfer Microbenchmark with lane-level grouped handles.

The benchmark builds these transfer handles:

  1. cross_tp
     Prefill uses the selected prefill DP/TP layout.
     Decode uses the selected cross decode DP/TP layout.
     Transfers run through one process group.

  2. aligned_lane
     Decode mirrors the prefill DP/TP layout.
     Transfers run through one process group with aligned DP lanes.

  3. aligned_lane_grouped
     Decode mirrors the prefill DP/TP layout.
     Each DP lane runs through its own process group.

This benchmark measures raw KV transfer behavior. Request routing, SLOs, and
decode queue pressure belong to the concurrent PD benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from pd_topology import add_topology_args, apply_topology_preset, topology_summary_dict


# =============================================================================
# 1. Data structures
# =============================================================================


@dataclass(frozen=True)
class LayoutConfig:
    prefill_dp: int = 2
    prefill_tp: int = 2
    decode_dp: int = 1
    decode_tp: int = 4

    @property
    def prefill_world_size(self) -> int:
        return self.prefill_dp * self.prefill_tp

    @property
    def decode_world_size(self) -> int:
        return self.decode_dp * self.decode_tp

    @property
    def total_world_size(self) -> int:
        return self.prefill_world_size + self.decode_world_size


@dataclass(frozen=True)
class Role:
    rank: int
    side: str  # "prefill" or "decode"
    dp: int
    tp: int

    def label(self) -> str:
        prefix = "P" if self.side == "prefill" else "D"
        return f"{prefix}-DP{self.dp}-TP{self.tp}"


@dataclass(frozen=True)
class TransferEdge:
    edge_id: int
    request_dp: int
    src_rank: int
    dst_rank: int
    prefill_tp: int
    decode_tp: int
    kv_head_start: int
    kv_head_end: int
    lane_id: int = -1

    @property
    def kv_head_count(self) -> int:
        return self.kv_head_end - self.kv_head_start


@dataclass(frozen=True)
class TransferMessage:
    msg_id: int
    edge_id: int
    src_rank: int
    dst_rank: int
    token_start: int
    token_count: int
    numel: int
    request_dp: int = -1
    lane_id: int = -1


@dataclass(frozen=True)
class TransferHandle:
    """A prebuilt transfer handle.

    process_group:
      Used by cross_tp and aligned_lane: one communicator for the whole handle.

    lane_groups / lane_group_ranks:
      Used by aligned_lane_grouped: one communicator per decode lane.
    """

    name: str
    layout: LayoutConfig
    group_ranks: Tuple[int, ...]
    edges: List[TransferEdge]
    messages: List[TransferMessage]
    process_group: Optional[dist.ProcessGroup] = None
    lane_groups: Dict[int, dist.ProcessGroup] = field(default_factory=dict)
    lane_group_ranks: Dict[int, Tuple[int, ...]] = field(default_factory=dict)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_messages(self) -> int:
        return len(self.messages)

    @property
    def is_grouped(self) -> bool:
        return bool(self.lane_groups)


@dataclass
class HandleBundle:
    handles: Dict[str, TransferHandle]

    def get(self, name: str) -> TransferHandle:
        if name not in self.handles:
            raise KeyError(f"Unknown transfer handle: {name}. Available: {list(self.handles)}")
        return self.handles[name]

    def summary(self) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for name, h in self.handles.items():
            out[name] = {
                "prefill_dp": h.layout.prefill_dp,
                "prefill_tp": h.layout.prefill_tp,
                "decode_dp": h.layout.decode_dp,
                "decode_tp": h.layout.decode_tp,
                "total_world_size": h.layout.total_world_size,
                "num_edges": h.num_edges,
                "num_messages": h.num_messages,
                "grouped": h.is_grouped,
                "lane_group_ranks": {str(k): list(v) for k, v in h.lane_group_ranks.items()},
            }
        return out


@dataclass
class RunMetrics:
    avg_ms: float
    p50_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    effective_gbps: float
    total_logical_bytes: int
    num_edges: int
    num_messages: int
    message_bytes_min: int
    message_bytes_max: int
    message_bytes_avg: float
    all_iter_ms: List[float]


# =============================================================================
# 2. Argument parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark heterogeneous KV-cache transfer with prebuilt transfer handles."
    )

    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    parser.add_argument(
        "--mode",
        type=str,
        default="async",
        choices=["sync", "async", "threaded-sync"],
        help=(
            "sync: deterministic edge/message loop; "
            "async: batch_isend_irecv; "
            "threaded-sync: Python-threaded blocking P2P baseline."
        ),
    )

    parser.add_argument(
        "--handle",
        type=str,
        default="cross_tp",
        choices=["cross_tp", "aligned_lane", "aligned_lane_grouped", "dynamic"],
        help="Prebuilt transfer handle to use. dynamic selects one handle at runtime.",
    )
    parser.add_argument(
        "--dynamic-seq-threshold",
        type=int,
        default=4096,
        help="When --handle dynamic is used, choose --dynamic-aligned-handle if seq_len >= this threshold.",
    )
    parser.add_argument(
        "--dynamic-aligned-handle",
        type=str,
        default="aligned_lane_grouped",
        choices=["aligned_lane", "aligned_lane_grouped"],
        help="The aligned handle selected by --handle dynamic for long sequences.",
    )
    parser.add_argument(
        "--use-process-groups",
        action="store_true",
        help=(
            "Pre-create torch.distributed process groups for transfer handles. "
            "aligned_lane_grouped requires this flag."
        ),
    )

    parser.add_argument("--prefill-dp", type=int, default=2)
    parser.add_argument("--prefill-tp", type=int, default=2)
    parser.add_argument("--decode-dp", type=int, default=1)
    parser.add_argument("--decode-tp", type=int, default=4)
    add_topology_args(parser)
    parser.add_argument(
        "--active-prefill-dps",
        type=str,
        default="all",
        help="Comma list such as '0' or '0,1'. Default 'all' means all prefill DP instances send concurrently.",
    )

    parser.add_argument("--requests-per-prefill-dp", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=1024,
        help=(
            "Number of sequence tokens per message. "
            "Use seq_len for one packed message per edge; use block size such as 16/32/256 for block-like transfer."
        ),
    )

    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--validate", action="store_true", help="Fill and check tensors. Use only for small debug runs.")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--print-rank-times", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)

    args = parser.parse_args()
    apply_topology_preset(args)
    return args


# =============================================================================
# 3. Distributed setup helpers
# =============================================================================


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def setup_dist(backend: str) -> Tuple[int, int, int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Please launch with torchrun so RANK and WORLD_SIZE are set.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA, but torch.cuda.is_available() is False.")
        num_gpus = torch.cuda.device_count()
        if num_gpus <= 0:
            raise RuntimeError("No visible CUDA devices.")
        device_id = local_rank % num_gpus
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
    else:
        device_id = -1
        device = torch.device("cpu")

    dist.init_process_group(backend=backend)
    return rank, world_size, device_id, device


def barrier(device: torch.device) -> None:
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


# =============================================================================
# 4. Layout / handle construction
# =============================================================================


def role_for_rank(rank: int, layout: LayoutConfig) -> Role:
    if rank < layout.prefill_world_size:
        dp = rank // layout.prefill_tp
        tp = rank % layout.prefill_tp
        return Role(rank=rank, side="prefill", dp=dp, tp=tp)

    rel = rank - layout.prefill_world_size
    dp = rel // layout.decode_tp
    tp = rel % layout.decode_tp
    return Role(rank=rank, side="decode", dp=dp, tp=tp)


def parse_active_prefill_dps(text: str, prefill_dp: int) -> List[int]:
    if text == "all":
        return list(range(prefill_dp))
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    for dp in values:
        if dp < 0 or dp >= prefill_dp:
            raise ValueError(f"Invalid active prefill DP id {dp}; valid range is [0, {prefill_dp}).")
    return values


def build_cross_tp_edges(layout: LayoutConfig, active_prefill_dps: Sequence[int], num_kv_heads: int) -> List[TransferEdge]:
    if layout.decode_dp != 1:
        raise NotImplementedError("cross_tp requires Decode DP=1, e.g. DP2TP2 -> DP1TP4.")
    if layout.decode_tp % layout.prefill_tp != 0:
        raise ValueError("cross_tp requires decode_tp to be divisible by prefill_tp.")
    if num_kv_heads % layout.decode_tp != 0:
        raise ValueError("num_kv_heads must be divisible by decode_tp.")

    heads_per_decode_tp = num_kv_heads // layout.decode_tp
    decode_per_prefill_tp = layout.decode_tp // layout.prefill_tp

    edges: List[TransferEdge] = []
    edge_id = 0
    for dp in active_prefill_dps:
        for p_tp in range(layout.prefill_tp):
            src_rank = dp * layout.prefill_tp + p_tp
            d_tp_begin = p_tp * decode_per_prefill_tp
            d_tp_end = (p_tp + 1) * decode_per_prefill_tp
            for d_tp in range(d_tp_begin, d_tp_end):
                dst_rank = layout.prefill_world_size + d_tp
                h0 = d_tp * heads_per_decode_tp
                h1 = (d_tp + 1) * heads_per_decode_tp
                edges.append(
                    TransferEdge(
                        edge_id=edge_id,
                        request_dp=dp,
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                        prefill_tp=p_tp,
                        decode_tp=d_tp,
                        kv_head_start=h0,
                        kv_head_end=h1,
                        lane_id=-1,
                    )
                )
                edge_id += 1
    return edges


def build_aligned_lane_edges(layout: LayoutConfig, active_prefill_dps: Sequence[int], num_kv_heads: int) -> List[TransferEdge]:
    if layout.prefill_dp != layout.decode_dp:
        raise ValueError("aligned_lane requires prefill_dp == decode_dp.")
    if layout.prefill_tp != layout.decode_tp:
        raise ValueError("aligned_lane requires prefill_tp == decode_tp.")
    if num_kv_heads % layout.decode_tp != 0:
        raise ValueError("num_kv_heads must be divisible by decode_tp.")

    heads_per_tp = num_kv_heads // layout.decode_tp

    edges: List[TransferEdge] = []
    edge_id = 0
    for dp in active_prefill_dps:
        for tp in range(layout.prefill_tp):
            src_rank = dp * layout.prefill_tp + tp
            dst_rank = layout.prefill_world_size + dp * layout.decode_tp + tp
            h0 = tp * heads_per_tp
            h1 = (tp + 1) * heads_per_tp
            edges.append(
                TransferEdge(
                    edge_id=edge_id,
                    request_dp=dp,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    prefill_tp=tp,
                    decode_tp=tp,
                    kv_head_start=h0,
                    kv_head_end=h1,
                    lane_id=dp,
                )
            )
            edge_id += 1
    return edges


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def build_messages(args: argparse.Namespace, edges: Sequence[TransferEdge], dtype: torch.dtype) -> List[TransferMessage]:
    _ = torch.tensor([], dtype=dtype).element_size()

    if args.chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive.")

    messages: List[TransferMessage] = []
    msg_id = 0
    for edge in edges:
        for chunk_idx in range(ceil_div(args.seq_len, args.chunk_tokens)):
            token_start = chunk_idx * args.chunk_tokens
            token_count = min(args.chunk_tokens, args.seq_len - token_start)
            numel = (
                args.requests_per_prefill_dp
                * args.num_layers
                * 2
                * token_count
                * edge.kv_head_count
                * args.head_dim
            )
            messages.append(
                TransferMessage(
                    msg_id=msg_id,
                    edge_id=edge.edge_id,
                    src_rank=edge.src_rank,
                    dst_rank=edge.dst_rank,
                    token_start=token_start,
                    token_count=token_count,
                    numel=numel,
                    request_dp=edge.request_dp,
                    lane_id=edge.lane_id,
                )
            )
            msg_id += 1
    return messages


def build_aligned_lane_group_ranks(layout: LayoutConfig) -> Dict[int, Tuple[int, ...]]:
    """Return lane_id -> ranks for aligned DP2TP2 -> DP2TP2 layout."""
    if layout.prefill_dp != layout.decode_dp or layout.prefill_tp != layout.decode_tp:
        raise ValueError("Lane grouping requires aligned prefill/decode DP and TP.")

    groups: Dict[int, Tuple[int, ...]] = {}
    for dp in range(layout.prefill_dp):
        prefill_ranks = [dp * layout.prefill_tp + tp for tp in range(layout.prefill_tp)]
        decode_ranks = [layout.prefill_world_size + dp * layout.decode_tp + tp for tp in range(layout.decode_tp)]
        groups[dp] = tuple(prefill_ranks + decode_ranks)
    return groups


def build_handle_bundle(args: argparse.Namespace, active_dps: Sequence[int], dtype: torch.dtype, device: torch.device) -> HandleBundle:
    """Prebuild all candidate transfer handles in a globally consistent order."""
    del device  # kept for future stream/device-specific communicator extensions.
    handles: Dict[str, TransferHandle] = {}

    def maybe_new_group(name: str, ranks: Tuple[int, ...]) -> Optional[dist.ProcessGroup]:
        # Important: all ranks must call dist.new_group in exactly the same order.
        del name
        if not args.use_process_groups:
            return None
        return dist.new_group(ranks=list(ranks), backend=args.backend)

    # 1. cross_tp: one 8-rank process group.
    cross_layout = LayoutConfig(
        prefill_dp=args.prefill_dp,
        prefill_tp=args.prefill_tp,
        decode_dp=args.decode_dp,
        decode_tp=args.decode_tp,
    )
    cross_edges = build_cross_tp_edges(cross_layout, active_dps, args.num_kv_heads)
    cross_messages = build_messages(args, cross_edges, dtype)
    cross_ranks = tuple(range(cross_layout.total_world_size))
    cross_pg = maybe_new_group("cross_tp", cross_ranks)
    handles["cross_tp"] = TransferHandle(
        name="cross_tp",
        layout=cross_layout,
        group_ranks=cross_ranks,
        edges=cross_edges,
        messages=cross_messages,
        process_group=cross_pg,
    )

    # 2. aligned_lane: one 8-rank process group, but aligned edges.
    aligned_layout = LayoutConfig(
        prefill_dp=args.prefill_dp,
        prefill_tp=args.prefill_tp,
        decode_dp=args.prefill_dp,
        decode_tp=args.prefill_tp,
    )
    aligned_edges = build_aligned_lane_edges(aligned_layout, active_dps, args.num_kv_heads)
    aligned_messages = build_messages(args, aligned_edges, dtype)
    aligned_ranks = tuple(range(aligned_layout.total_world_size))
    aligned_pg = maybe_new_group("aligned_lane", aligned_ranks)
    handles["aligned_lane"] = TransferHandle(
        name="aligned_lane",
        layout=aligned_layout,
        group_ranks=aligned_ranks,
        edges=aligned_edges,
        messages=aligned_messages,
        process_group=aligned_pg,
    )

    # 3. aligned_lane_grouped: same edges, but one process group per lane.
    # All ranks create lane groups in deterministic lane_id order.
    lane_group_ranks = build_aligned_lane_group_ranks(aligned_layout)
    lane_groups: Dict[int, dist.ProcessGroup] = {}
    if args.use_process_groups:
        for lane_id in sorted(lane_group_ranks):
            lane_groups[lane_id] = dist.new_group(ranks=list(lane_group_ranks[lane_id]), backend=args.backend)

    handles["aligned_lane_grouped"] = TransferHandle(
        name="aligned_lane_grouped",
        layout=aligned_layout,
        group_ranks=aligned_ranks,
        edges=aligned_edges,
        messages=aligned_messages,
        process_group=None,
        lane_groups=lane_groups,
        lane_group_ranks=lane_group_ranks,
    )

    return HandleBundle(handles=handles)


def select_handle_name(args: argparse.Namespace, bundle: HandleBundle, active_dps: Sequence[int]) -> str:
    if args.handle != "dynamic":
        return args.handle

    # Threshold routing uses sequence length only; queue-aware routing lives in
    # the concurrent PD benchmark where scheduler state is available.
    if args.seq_len >= args.dynamic_seq_threshold:
        return args.dynamic_aligned_handle

    if len(active_dps) == 1:
        return "cross_tp"

    return "cross_tp"


# =============================================================================
# 5. Buffer allocation and communication kernels
# =============================================================================


def allocate_local_buffers(
    rank: int,
    messages: Sequence[TransferMessage],
    dtype: torch.dtype,
    device: torch.device,
    validate: bool,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    send_bufs: Dict[int, torch.Tensor] = {}
    recv_bufs: Dict[int, torch.Tensor] = {}

    for msg in messages:
        if rank == msg.src_rank:
            buf = torch.empty(msg.numel, dtype=dtype, device=device)
            if validate:
                buf.fill_(float(msg.src_rank + 1))
            send_bufs[msg.msg_id] = buf
        elif rank == msg.dst_rank:
            buf = torch.empty(msg.numel, dtype=dtype, device=device)
            if validate:
                buf.zero_()
            recv_bufs[msg.msg_id] = buf

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return send_bufs, recv_bufs


def run_sync(
    rank: int,
    messages: Sequence[TransferMessage],
    send_bufs: Dict[int, torch.Tensor],
    recv_bufs: Dict[int, torch.Tensor],
    process_group: Optional[dist.ProcessGroup] = None,
) -> None:
    for msg in messages:
        if rank == msg.src_rank:
            dist.send(send_bufs[msg.msg_id], dst=msg.dst_rank, group=process_group)
        elif rank == msg.dst_rank:
            dist.recv(recv_bufs[msg.msg_id], src=msg.src_rank, group=process_group)


def run_async_batch(
    rank: int,
    messages: Sequence[TransferMessage],
    send_bufs: Dict[int, torch.Tensor],
    recv_bufs: Dict[int, torch.Tensor],
    process_group: Optional[dist.ProcessGroup] = None,
) -> None:
    ops: List[dist.P2POp] = []
    for msg in messages:
        if rank == msg.src_rank:
            ops.append(dist.P2POp(dist.isend, send_bufs[msg.msg_id], msg.dst_rank, group=process_group))
        elif rank == msg.dst_rank:
            ops.append(dist.P2POp(dist.irecv, recv_bufs[msg.msg_id], msg.src_rank, group=process_group))

    if not ops:
        return

    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()


def run_threaded_sync(
    rank: int,
    messages: Sequence[TransferMessage],
    send_bufs: Dict[int, torch.Tensor],
    recv_bufs: Dict[int, torch.Tensor],
    process_group: Optional[dist.ProcessGroup] = None,
) -> None:
    errors: List[BaseException] = []
    lock = threading.Lock()

    def worker(fn, *fn_args, **fn_kwargs) -> None:
        try:
            fn(*fn_args, **fn_kwargs)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads: List[threading.Thread] = []
    for msg in messages:
        if rank == msg.src_rank:
            t = threading.Thread(
                target=worker,
                args=(dist.send, send_bufs[msg.msg_id]),
                kwargs={"dst": msg.dst_rank, "group": process_group},
            )
            threads.append(t)
        elif rank == msg.dst_rank:
            t = threading.Thread(
                target=worker,
                args=(dist.recv, recv_bufs[msg.msg_id]),
                kwargs={"src": msg.src_rank, "group": process_group},
            )
            threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise RuntimeError(f"Threaded P2P encountered {len(errors)} error(s). First: {errors[0]}")


def execute_once(
    mode: str,
    rank: int,
    messages: Sequence[TransferMessage],
    send_bufs: Dict[int, torch.Tensor],
    recv_bufs: Dict[int, torch.Tensor],
    process_group: Optional[dist.ProcessGroup] = None,
) -> None:
    if mode == "sync":
        run_sync(rank, messages, send_bufs, recv_bufs, process_group)
    elif mode == "async":
        run_async_batch(rank, messages, send_bufs, recv_bufs, process_group)
    elif mode == "threaded-sync":
        run_threaded_sync(rank, messages, send_bufs, recv_bufs, process_group)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def execute_once_handle(
    mode: str,
    rank: int,
    handle: TransferHandle,
    send_bufs: Dict[int, torch.Tensor],
    recv_bufs: Dict[int, torch.Tensor],
) -> None:
    """Execute a handle.

    Non-grouped handles use one process group.
    Grouped handles route each lane's messages through its lane process group.
    """
    if not handle.is_grouped:
        execute_once(mode, rank, handle.messages, send_bufs, recv_bufs, handle.process_group)
        return

    if not handle.lane_groups:
        raise RuntimeError("Grouped handle requires --use-process-groups so lane_groups are created.")

    for lane_id in sorted(handle.lane_group_ranks):
        lane_ranks = handle.lane_group_ranks[lane_id]
        if rank not in lane_ranks:
            continue
        lane_messages = [m for m in handle.messages if m.lane_id == lane_id]
        execute_once(mode, rank, lane_messages, send_bufs, recv_bufs, handle.lane_groups[lane_id])


def validate_recv_buffers(
    rank: int,
    messages: Sequence[TransferMessage],
    recv_bufs: Dict[int, torch.Tensor],
    device: torch.device,
) -> None:
    local_ok = torch.tensor([1], dtype=torch.int32, device=device)

    for msg in messages:
        if rank != msg.dst_rank:
            continue
        buf = recv_bufs[msg.msg_id]
        expected = float(msg.src_rank + 1)
        if not torch.all(buf == expected):
            local_ok.fill_(0)
            break

    dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
    if int(local_ok.item()) != 1:
        raise RuntimeError("Validation failed: at least one received buffer has unexpected content.")


# =============================================================================
# 6. Benchmark driver
# =============================================================================


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[idx]


def benchmark(args: argparse.Namespace, rank: int, world_size: int, device_id: int, device: torch.device) -> Optional[RunMetrics]:
    del device_id
    dtype = dtype_from_name(args.dtype)
    active_dps = parse_active_prefill_dps(args.active_prefill_dps, args.prefill_dp)

    if args.handle == "aligned_lane_grouped" and not args.use_process_groups:
        raise RuntimeError("aligned_lane_grouped requires --use-process-groups.")

    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"WORLD_SIZE={world_size}, but topology '{args.topology_name}' requires "
            f"{args.expected_world_size} ranks. Use torchrun --nproc_per_node={args.expected_world_size}."
        )

    bundle = build_handle_bundle(args, active_dps, dtype, device)
    selected_name = select_handle_name(args, bundle, active_dps)
    handle = bundle.get(selected_name)

    layout = handle.layout
    edges = handle.edges
    messages = handle.messages

    if world_size != layout.total_world_size:
        raise RuntimeError(
            f"WORLD_SIZE={world_size}, but selected handle '{selected_name}' requires "
            f"{layout.total_world_size} ranks. Use torchrun --nproc_per_node={layout.total_world_size}."
        )

    send_bufs, recv_bufs = allocate_local_buffers(rank, messages, dtype, device, args.validate)

    elem_size = torch.tensor([], dtype=dtype).element_size()
    message_bytes = [msg.numel * elem_size for msg in messages]
    total_logical_bytes = sum(message_bytes)

    role = role_for_rank(rank, layout)

    if rank == 0:
        num_visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print("=" * 100)
        print("Heterogeneous KV Transfer Benchmark with Lane-Level Grouped Handles")
        print("=" * 100)
        print(f"selected handle    : {selected_name}")
        if args.handle == "dynamic":
            print(
                f"dynamic threshold  : seq_len >= {args.dynamic_seq_threshold} -> {args.dynamic_aligned_handle}"
            )
        print(f"topology           : {json.dumps(topology_summary_dict(args), sort_keys=True)}")
        print(f"available handles  : {json.dumps(bundle.summary(), indent=2)}")
        print(
            f"layout             : prefill DP={layout.prefill_dp}, TP={layout.prefill_tp}; "
            f"decode DP={layout.decode_dp}, TP={layout.decode_tp}"
        )
        print(f"active prefill DPs : {active_dps}")
        print(f"world_size         : {world_size}")
        print(f"visible CUDA GPUs  : {num_visible_gpus}")
        if args.backend == "nccl" and num_visible_gpus < world_size:
            print("warning            : ranks are oversubscribed onto fewer GPUs; use only for logic/debug, not performance.")
        print(f"backend/mode       : {args.backend} / {args.mode}")
        print(f"use process group  : {args.use_process_groups}")
        print(f"grouped handle     : {handle.is_grouped}")
        if handle.is_grouped:
            print(f"lane group ranks   : {handle.lane_group_ranks}")
        print(
            f"KV shape           : layers={args.num_layers}, seq_len={args.seq_len}, "
            f"kv_heads={args.num_kv_heads}, head_dim={args.head_dim}, dtype={args.dtype}"
        )
        print(f"chunk_tokens       : {args.chunk_tokens}")
        print(f"requests per P-DP  : {args.requests_per_prefill_dp}")
        print(f"num_edges          : {len(edges)}")
        print(f"num_messages       : {len(messages)}")
        print(f"logical bytes      : {total_logical_bytes / (1024 ** 3):.3f} GiB")
        print(
            f"message bytes      : min={min(message_bytes) / (1024 ** 2):.3f} MiB, "
            f"max={max(message_bytes) / (1024 ** 2):.3f} MiB, "
            f"avg={statistics.mean(message_bytes) / (1024 ** 2):.3f} MiB"
        )
        print("=" * 100)

        if args.print_plan:
            roles = [role_for_rank(r, layout) for r in range(world_size)]
            print("Ranks:")
            for r in roles:
                print(f"  rank {r.rank}: {r.label()}")
            print("Edges:")
            for e in edges:
                src = role_for_rank(e.src_rank, layout).label()
                dst = role_for_rank(e.dst_rank, layout).label()
                print(
                    f"  edge {e.edge_id}: {src} -> {dst}, "
                    f"request_dp={e.request_dp}, lane={e.lane_id}, kv_heads=[{e.kv_head_start},{e.kv_head_end})"
                )
            print("=" * 100)

    barrier(device)

    for _ in range(args.warmup):
        execute_once_handle(args.mode, rank, handle, send_bufs, recv_bufs)
        barrier(device)

    if args.validate:
        execute_once_handle(args.mode, rank, handle, send_bufs, recv_bufs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        validate_recv_buffers(rank, messages, recv_bufs, device)
        barrier(device)
        if rank == 0:
            print("validation         : passed")

    iter_ms: List[float] = []
    rank_local_ms: List[List[float]] = []

    for _ in range(args.iters):
        barrier(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            execute_once_handle(args.mode, rank, handle, send_bufs, recv_bufs)
            end.record()
            torch.cuda.synchronize(device)
            local_ms = float(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            execute_once_handle(args.mode, rank, handle, send_bufs, recv_bufs)
            local_ms = (time.perf_counter() - t0) * 1000.0

        t = torch.tensor([local_ms], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        global_ms = float(t.item())
        iter_ms.append(global_ms)

        if args.print_rank_times:
            gathered = [torch.zeros_like(t) for _ in range(world_size)]
            dist.all_gather(gathered, torch.tensor([local_ms], dtype=torch.float64, device=device))
            rank_local_ms.append([float(x.item()) for x in gathered])

    avg_ms = statistics.mean(iter_ms)
    metrics = RunMetrics(
        avg_ms=avg_ms,
        p50_ms=statistics.median(iter_ms),
        p90_ms=percentile(iter_ms, 0.90),
        min_ms=min(iter_ms),
        max_ms=max(iter_ms),
        effective_gbps=total_logical_bytes / (avg_ms / 1000.0) / 1e9,
        total_logical_bytes=total_logical_bytes,
        num_edges=len(edges),
        num_messages=len(messages),
        message_bytes_min=min(message_bytes),
        message_bytes_max=max(message_bytes),
        message_bytes_avg=statistics.mean(message_bytes),
        all_iter_ms=iter_ms,
    )

    if rank == 0:
        print("Results")
        print("-" * 100)
        print(f"selected_handle    : {selected_name}")
        print(f"avg_ms             : {metrics.avg_ms:.3f}")
        print(f"p50_ms             : {metrics.p50_ms:.3f}")
        print(f"p90_ms             : {metrics.p90_ms:.3f}")
        print(f"min_ms             : {metrics.min_ms:.3f}")
        print(f"max_ms             : {metrics.max_ms:.3f}")
        print(f"effective_GBps     : {metrics.effective_gbps:.3f}")
        print(f"all_iter_ms        : {[round(x, 3) for x in metrics.all_iter_ms]}")
        if args.print_rank_times:
            print("rank_local_ms      :")
            for i, row in enumerate(rank_local_ms):
                print(f"  iter {i}: {[round(x, 3) for x in row]}")
        print("-" * 100)

        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "selected_handle": selected_name,
                "handle_bundle_summary": bundle.summary(),
                "use_process_groups": args.use_process_groups,
                "layout": asdict(layout),
                "role_rank0": role.label(),
                "active_prefill_dps": list(active_dps),
                "args": vars(args),
                "metrics": asdict(metrics),
                "edges": [asdict(e) for e in edges],
                "messages_preview": [asdict(m) for m in messages[: min(16, len(messages))]],
                "rank_roles": [asdict(role_for_rank(r, layout)) for r in range(world_size)],
            }
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"saved_json         : {output_path}")

        return metrics

    return None


# =============================================================================
# 7. Entry point
# =============================================================================


def main() -> None:
    args = parse_args()
    rank, world_size, device_id, device = setup_dist(args.backend)

    try:
        benchmark(args, rank, world_size, device_id, device)
        barrier(device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
