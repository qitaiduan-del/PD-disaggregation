#!/usr/bin/env python3
"""
Topology presets and validation for PD transfer/routing benchmarks.

Design
------
Preset topologies keep one invariant:

  prefill_world_size == cross_decode_world_size == grouped_decode_world_size

This keeps one torchrun WORLD_SIZE compatible with both decode layouts:

  total_world_size = prefill_world_size + decode_world_size

For example, with 8 ranks:

  p2t2_x1t4_g2t2:
    Prefill        DP=2, TP=2  ->  4 ranks
    Cross decode   DP=1, TP=4  ->  4 ranks
    Grouped decode DP=2, TP=2  ->  4 ranks
    Total                       ->  8 ranks

  p4t1_x1t4_g4t1:
    Prefill        DP=4, TP=1  ->  4 ranks
    Cross decode   DP=1, TP=4  ->  4 ranks
    Grouped decode DP=4, TP=1  ->  4 ranks
    Total                       ->  8 ranks

The grouped decode layout is implicit: DP=prefill_dp, TP=prefill_tp.
The cross decode layout is controlled by decode_dp/decode_tp.

Unsupported or inconsistent topologies fail early with a clear error message.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TopologyPreset:
    name: str
    prefill_dp: int
    prefill_tp: int
    cross_decode_dp: int
    cross_decode_tp: int
    description: str

    @property
    def prefill_world_size(self) -> int:
        return self.prefill_dp * self.prefill_tp

    @property
    def cross_decode_world_size(self) -> int:
        return self.cross_decode_dp * self.cross_decode_tp

    @property
    def grouped_decode_dp(self) -> int:
        return self.prefill_dp

    @property
    def grouped_decode_tp(self) -> int:
        return self.prefill_tp

    @property
    def grouped_decode_world_size(self) -> int:
        return self.grouped_decode_dp * self.grouped_decode_tp

    @property
    def total_world_size(self) -> int:
        return self.prefill_world_size + self.cross_decode_world_size


TOPOLOGY_PRESETS: Dict[str, TopologyPreset] = {
    "p2t2_x1t4_g2t2": TopologyPreset(
        name="p2t2_x1t4_g2t2",
        prefill_dp=2,
        prefill_tp=2,
        cross_decode_dp=1,
        cross_decode_tp=4,
        description="8 ranks: Prefill DP2TP2; Cross decode DP1TP4; Grouped decode DP2TP2.",
    ),
    "p4t1_x1t4_g4t1": TopologyPreset(
        name="p4t1_x1t4_g4t1",
        prefill_dp=4,
        prefill_tp=1,
        cross_decode_dp=1,
        cross_decode_tp=4,
        description="8 ranks: Prefill DP4TP1; Cross decode DP1TP4; Grouped decode DP4TP1.",
    ),
}


def add_topology_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--topology",
        type=str,
        default="p2t2_x1t4_g2t2",
        choices=list(TOPOLOGY_PRESETS.keys()) + ["custom"],
        help=(
            "Topology preset. Use custom to keep explicit --prefill-dp/--prefill-tp/"
            "--decode-dp/--decode-tp values. In presets, grouped decode is implicit: "
            "DP=prefill_dp, TP=prefill_tp."
        ),
    )
    parser.add_argument(
        "--print-topologies",
        action="store_true",
        help="Print supported topology presets and exit before running benchmark.",
    )


def format_topology_table() -> str:
    lines = []
    lines.append("Supported topology presets:")
    for name, preset in TOPOLOGY_PRESETS.items():
        lines.append(f"  {name}:")
        lines.append(f"    prefill        : DP={preset.prefill_dp}, TP={preset.prefill_tp}, ranks={preset.prefill_world_size}")
        lines.append(f"    cross decode   : DP={preset.cross_decode_dp}, TP={preset.cross_decode_tp}, ranks={preset.cross_decode_world_size}")
        lines.append(f"    grouped decode : DP={preset.grouped_decode_dp}, TP={preset.grouped_decode_tp}, ranks={preset.grouped_decode_world_size}")
        lines.append(f"    total ranks    : {preset.total_world_size}")
        lines.append(f"    description    : {preset.description}")
    return "\n".join(lines)


def apply_topology_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply preset values to argparse namespace in-place, then validate.

    Benchmark scripts consume args.prefill_dp / args.prefill_tp / args.decode_dp / args.decode_tp.
    Presets write those fields before validation.
    """
    if getattr(args, "print_topologies", False):
        print(format_topology_table())
        raise SystemExit(0)

    topology = getattr(args, "topology", "custom")
    if topology != "custom":
        preset = TOPOLOGY_PRESETS[topology]
        args.prefill_dp = preset.prefill_dp
        args.prefill_tp = preset.prefill_tp
        args.decode_dp = preset.cross_decode_dp
        args.decode_tp = preset.cross_decode_tp
        args.topology_name = preset.name
        args.grouped_decode_dp = preset.grouped_decode_dp
        args.grouped_decode_tp = preset.grouped_decode_tp
        args.expected_world_size = preset.total_world_size
    else:
        args.topology_name = "custom"
        args.grouped_decode_dp = args.prefill_dp
        args.grouped_decode_tp = args.prefill_tp
        args.expected_world_size = args.prefill_dp * args.prefill_tp + args.decode_dp * args.decode_tp

    validate_topology_args(args)
    return args


def validate_topology_args(args: argparse.Namespace) -> None:
    prefill_world = args.prefill_dp * args.prefill_tp
    cross_decode_world = args.decode_dp * args.decode_tp
    grouped_decode_world = args.grouped_decode_dp * args.grouped_decode_tp

    if prefill_world <= 0 or cross_decode_world <= 0 or grouped_decode_world <= 0:
        raise ValueError(
            f"Invalid topology: all world sizes must be positive, got "
            f"prefill={prefill_world}, cross_decode={cross_decode_world}, grouped_decode={grouped_decode_world}."
        )

    # The current handle bundle assumes both candidates share the same decode rank pool size.
    if cross_decode_world != grouped_decode_world:
        raise ValueError(
            "Unsupported topology: cross decode world size must equal grouped decode world size. "
            f"Got cross_decode_world={cross_decode_world}, grouped_decode_world={grouped_decode_world}. "
            "Use a preset or choose custom values satisfying decode_dp*decode_tp == prefill_dp*prefill_tp."
        )

    # The cross_tp edge builder maps prefill shards into one decode DP instance.
    if args.decode_dp != 1:
        raise ValueError(
            "Unsupported topology: current cross_tp builder requires cross decode DP=1. "
            f"Got decode_dp={args.decode_dp}."
        )

    if args.decode_tp % args.prefill_tp != 0:
        raise ValueError(
            "Unsupported topology: cross decode TP must be divisible by prefill TP so each prefill TP shard "
            f"can fan out evenly. Got decode_tp={args.decode_tp}, prefill_tp={args.prefill_tp}."
        )

    if hasattr(args, "num_kv_heads") and args.num_kv_heads % args.decode_tp != 0:
        raise ValueError(
            "Unsupported KV shape: num_kv_heads must be divisible by cross decode TP. "
            f"Got num_kv_heads={args.num_kv_heads}, decode_tp={args.decode_tp}."
        )

    if hasattr(args, "num_kv_heads") and args.num_kv_heads % args.prefill_tp != 0:
        raise ValueError(
            "Unsupported KV shape: num_kv_heads must be divisible by grouped decode TP / prefill TP. "
            f"Got num_kv_heads={args.num_kv_heads}, prefill_tp={args.prefill_tp}."
        )


def topology_summary_dict(args: argparse.Namespace) -> dict:
    return {
        "topology": getattr(args, "topology_name", getattr(args, "topology", "unknown")),
        "prefill_dp": args.prefill_dp,
        "prefill_tp": args.prefill_tp,
        "prefill_world_size": args.prefill_dp * args.prefill_tp,
        "cross_decode_dp": args.decode_dp,
        "cross_decode_tp": args.decode_tp,
        "cross_decode_world_size": args.decode_dp * args.decode_tp,
        "grouped_decode_dp": args.grouped_decode_dp,
        "grouped_decode_tp": args.grouped_decode_tp,
        "grouped_decode_world_size": args.grouped_decode_dp * args.grouped_decode_tp,
        "expected_world_size": args.expected_world_size,
    }
