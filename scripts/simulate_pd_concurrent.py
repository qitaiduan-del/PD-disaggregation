#!/usr/bin/env python3
"""
Concurrent PD scheduling simulator.

The simulator uses measured KV-transfer profiles to compare cross-TP,
aligned-lane-grouped, and dynamic routing policies under the same request
arrival stream. Matplotlib uses the Agg backend so figures can be generated on
headless GPU servers.

Run from project root:

python scripts/simulate_pd_concurrent.py \
  --summary results/transfer_profile_summary.csv \
  --num-requests 200 \
  --arrival-pattern burst \
  --arrival-rate-rps 35 \
  --output-dir results/simulation/grouped_lane

Outputs:
  results/simulation/grouped_lane/generated_workload.csv
  results/simulation/grouped_lane/sim_request_trace.csv
  results/simulation/grouped_lane/sim_summary.csv
  results/simulation/grouped_lane/figures/*.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

# Critical for remote/headless machines.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
class SimulatedRequest:
    policy: str
    req_id: int
    arrival_ms: float
    input_len: int
    output_len: int
    prefill_dp: int
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
    transfer_ms: float
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
    parser = argparse.ArgumentParser(description="Concurrent PD scheduling simulator using measured KV transfer profiles.")

    parser.add_argument("--summary", type=str, default="results/transfer_profile_summary.csv")
    parser.add_argument("--results-dir", type=str, default="results/transfer")
    parser.add_argument("--output-dir", type=str, default="results/simulation/grouped_lane")

    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--arrival-pattern", choices=["poisson", "burst", "fixed"], default="burst")
    parser.add_argument("--arrival-rate-rps", type=float, default=35.0)
    parser.add_argument("--burst-size", type=int, default=8)
    parser.add_argument("--burst-interval-ms", type=float, default=180.0)

    parser.add_argument("--input-lens", type=str, default="1024,2048,4096,8192")
    parser.add_argument("--input-probs", type=str, default="0.35,0.30,0.25,0.10")
    parser.add_argument("--output-lens", type=str, default="64,128,256")
    parser.add_argument("--output-probs", type=str, default="0.50,0.35,0.15")

    parser.add_argument("--prefill-dps", type=int, default=2)
    parser.add_argument("--prefill-base-ms", type=float, default=4.0)
    parser.add_argument("--prefill-ms-per-1k", type=float, default=32.0)
    parser.add_argument("--prefill-jitter-ratio", type=float, default=0.08)

    # cross_tp is modeled as one shared TP4 decode queue: faster per token, but centralized.
    parser.add_argument("--decode-tp4-ms", type=float, default=7.5)
    # aligned_lane_grouped is modeled as two TP2 decode lanes: slower per token, but isolated.
    parser.add_argument("--decode-tp2-ms", type=float, default=11.0)

    parser.add_argument("--ttft-slo-ms", type=float, default=250.0)
    parser.add_argument("--itl-slo-ms", type=float, default=80.0)
    parser.add_argument("--slo-penalty", type=float, default=5.0)

    parser.add_argument("--dynamic-mode", choices=["cost", "threshold"], default="cost")
    parser.add_argument("--dynamic-seq-threshold", type=int, default=4096)
    parser.add_argument(
        "--dynamic-allow-any-aligned-lane",
        action="store_true",
        help="Allow dynamic policy to choose the less-loaded aligned lane instead of tying lane to prefill_dp.",
    )

    parser.add_argument("--timeline-max-requests", type=int, default=80)
    return parser.parse_args()


# =============================================================================
# General helpers
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


def safe_mean(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


# =============================================================================
# Load transfer profiles
# =============================================================================


def load_summary(summary_path: Path, results_dir: Path) -> pd.DataFrame:
    if summary_path.exists():
        return pd.read_csv(summary_path)

    rows: List[Dict[str, object]] = []
    for p in sorted(results_dir.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        args = data.get("args", {})
        metrics = data.get("metrics", {})
        layout = data.get("layout", {})
        rows.append(
            {
                "file": p.name,
                "selected_handle": data.get("selected_handle"),
                "handle_arg": args.get("handle"),
                "mode": args.get("mode"),
                "seq_len": args.get("seq_len"),
                "chunk_tokens": args.get("chunk_tokens"),
                "num_layers": args.get("num_layers"),
                "prefill_dp": layout.get("prefill_dp"),
                "prefill_tp": layout.get("prefill_tp"),
                "decode_dp": layout.get("decode_dp"),
                "decode_tp": layout.get("decode_tp"),
                "num_edges": metrics.get("num_edges"),
                "num_messages": metrics.get("num_messages"),
                "avg_ms": metrics.get("avg_ms"),
                "p50_ms": metrics.get("p50_ms"),
                "p90_ms": metrics.get("p90_ms"),
                "effective_gbps": metrics.get("effective_gbps"),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No summary CSV at {summary_path} and no JSON files in {results_dir}")
    return pd.DataFrame(rows)


class TransferProfile:
    def __init__(self, df: pd.DataFrame):
        work = df.copy()
        for col in ["seq_len", "chunk_tokens", "num_layers", "avg_ms"]:
            work[col] = pd.to_numeric(work.get(col), errors="coerce")

        packed = work[
            (work["mode"] == "async")
            & (work["handle_arg"].isin(["cross_tp", "aligned_lane", "aligned_lane_grouped"]))
            & (work["seq_len"] == work["chunk_tokens"])
            & (work["num_layers"] == 32)
            & work["avg_ms"].notna()
        ].copy()

        if packed.empty:
            raise ValueError("No packed async transfer profile found. Check results/transfer_profile_summary.csv.")

        self.points: Dict[str, List[Tuple[float, float]]] = {}
        for handle, g in packed.groupby("handle_arg"):
            agg = g.groupby("seq_len", as_index=False)["avg_ms"].mean().sort_values("seq_len")
            self.points[str(handle)] = [(float(row.seq_len), float(row.avg_ms)) for row in agg.itertuples()]

        if "cross_tp" not in self.points:
            raise ValueError("Missing cross_tp transfer profile.")

        # If a real aligned_lane_grouped profile exists, use it. Otherwise use aligned_lane as fallback.
        if "aligned_lane_grouped" not in self.points:
            if "aligned_lane" not in self.points:
                raise ValueError("Missing aligned_lane/aligned_lane_grouped transfer profile.")
            self.points["aligned_lane_grouped"] = list(self.points["aligned_lane"])

    def transfer_ms(self, handle: str, seq_len: int) -> float:
        if handle == "aligned_lane":
            handle = "aligned_lane_grouped"
        pts = self.points[handle]
        xs = np.array([p[0] for p in pts], dtype=np.float64)
        ys = np.array([p[1] for p in pts], dtype=np.float64)

        if seq_len <= xs[0]:
            return float(ys[0] * seq_len / xs[0])
        if seq_len >= xs[-1]:
            if len(xs) >= 2:
                slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
                return float(max(0.0, ys[-1] + slope * (seq_len - xs[-1])))
            return float(ys[-1] * seq_len / xs[-1])
        return float(np.interp(seq_len, xs, ys))


# =============================================================================
# Workload generation
# =============================================================================


def generate_workload(args: argparse.Namespace) -> List[Request]:
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    input_lens = parse_int_list(args.input_lens)
    input_probs = parse_float_list(args.input_probs)
    output_lens = parse_int_list(args.output_lens)
    output_probs = parse_float_list(args.output_probs)

    requests: List[Request] = []
    for i in range(args.num_requests):
        if args.arrival_pattern == "poisson":
            if i == 0:
                arrival_ms = 0.0
            else:
                arrival_ms = requests[-1].arrival_ms + float(np_rng.exponential(1000.0 / args.arrival_rate_rps))
        elif args.arrival_pattern == "fixed":
            arrival_ms = i * (1000.0 / args.arrival_rate_rps)
        elif args.arrival_pattern == "burst":
            burst_id = i // args.burst_size
            offset = i % args.burst_size
            arrival_ms = burst_id * args.burst_interval_ms + offset * 1.0
        else:
            raise ValueError(args.arrival_pattern)

        requests.append(
            Request(
                req_id=i,
                arrival_ms=arrival_ms,
                input_len=int(rng.choices(input_lens, weights=input_probs, k=1)[0]),
                output_len=int(rng.choices(output_lens, weights=output_probs, k=1)[0]),
            )
        )
    return requests


# =============================================================================
# Simulation
# =============================================================================


def prefill_time_ms(req: Request, args: argparse.Namespace, rng: random.Random) -> float:
    base = args.prefill_base_ms + args.prefill_ms_per_1k * (req.input_len / 1024.0)
    jitter = 1.0 + rng.uniform(-args.prefill_jitter_ratio, args.prefill_jitter_ratio)
    return max(0.0, base * jitter)


def summarize_rows(rows: List[SimulatedRequest], policy: str) -> Dict[str, float | str | int]:
    return {
        "policy": policy,
        "num_requests": len(rows),
        "mean_ttft_ms": safe_mean(r.ttft_ms for r in rows),
        "p50_ttft_ms": percentile((r.ttft_ms for r in rows), 50),
        "p90_ttft_ms": percentile((r.ttft_ms for r in rows), 90),
        "p99_ttft_ms": percentile((r.ttft_ms for r in rows), 99),
        "mean_handoff_gap_ms": safe_mean(r.handoff_gap_ms for r in rows),
        "p50_handoff_gap_ms": percentile((r.handoff_gap_ms for r in rows), 50),
        "p90_handoff_gap_ms": percentile((r.handoff_gap_ms for r in rows), 90),
        "p99_handoff_gap_ms": percentile((r.handoff_gap_ms for r in rows), 99),
        "mean_decode_queue_wait_ms": safe_mean(r.decode_queue_wait_ms for r in rows),
        "p90_decode_queue_wait_ms": percentile((r.decode_queue_wait_ms for r in rows), 90),
        "p99_decode_queue_wait_ms": percentile((r.decode_queue_wait_ms for r in rows), 99),
        "mean_e2e_ms": safe_mean(r.e2e_ms for r in rows),
        "p90_e2e_ms": percentile((r.e2e_ms for r in rows), 90),
        "p99_e2e_ms": percentile((r.e2e_ms for r in rows), 99),
        "ttft_slo_violation_rate": safe_mean(float(r.ttft_slo_violation) for r in rows),
        "itl_slo_violation_rate": safe_mean(float(r.itl_slo_violation) for r in rows),
        "cross_tp_fraction": safe_mean(float(r.selected_handle == "cross_tp") for r in rows),
        "grouped_fraction": safe_mean(float(r.selected_handle == "aligned_lane_grouped") for r in rows),
        "makespan_ms": max(r.decode_end_ms for r in rows) - min(r.arrival_ms for r in rows),
    }


def simulate_policy(
    requests: List[Request],
    policy: str,
    profile: TransferProfile,
    args: argparse.Namespace,
) -> Tuple[List[SimulatedRequest], Dict[str, float | str | int]]:
    rng = random.Random(args.seed + abs(hash(policy)) % 10_000)

    prefill_available = [0.0 for _ in range(args.prefill_dps)]
    cross_transfer_available = 0.0
    grouped_transfer_available = [0.0 for _ in range(args.prefill_dps)]
    cross_decode_available = 0.0
    grouped_decode_available = [0.0 for _ in range(args.prefill_dps)]

    simulated: List[SimulatedRequest] = []

    def predict_option(req: Request, prefill_end: float, handle: str, lane: int) -> Tuple[float, float, float, float, float, float]:
        if handle == "cross_tp":
            transfer_ms = profile.transfer_ms("cross_tp", req.input_len)
            transfer_start = max(prefill_end, cross_transfer_available)
            transfer_end = transfer_start + transfer_ms
            decode_ms = args.decode_tp4_ms
            decode_start = max(transfer_end, cross_decode_available)
        elif handle == "aligned_lane_grouped":
            transfer_ms = profile.transfer_ms("aligned_lane_grouped", req.input_len)
            transfer_start = max(prefill_end, grouped_transfer_available[lane])
            transfer_end = transfer_start + transfer_ms
            decode_ms = args.decode_tp2_ms
            decode_start = max(transfer_end, grouped_decode_available[lane])
        else:
            raise ValueError(handle)

        first_decode_token = decode_start + decode_ms
        decode_end = decode_start + req.output_len * decode_ms
        handoff_gap = first_decode_token - prefill_end
        e2e = decode_end - req.arrival_ms

        penalty = 0.0
        if handoff_gap > args.itl_slo_ms:
            penalty += args.slo_penalty * (handoff_gap - args.itl_slo_ms)
        if decode_ms > args.itl_slo_ms:
            penalty += args.slo_penalty * (decode_ms - args.itl_slo_ms)
        cost = e2e + penalty
        return cost, transfer_start, transfer_end, decode_start, first_decode_token, decode_end

    for req in sorted(requests, key=lambda r: (r.arrival_ms, r.req_id)):
        prefill_dp = min(range(args.prefill_dps), key=lambda d: prefill_available[d])
        p_start = max(req.arrival_ms, prefill_available[prefill_dp])
        p_end = p_start + prefill_time_ms(req, args, rng)
        prefill_available[prefill_dp] = p_end

        if policy == "cross_tp":
            selected_handle = "cross_tp"
            lane = 0
        elif policy == "aligned_lane_grouped":
            selected_handle = "aligned_lane_grouped"
            lane = prefill_dp
        elif policy == "dynamic":
            if args.dynamic_mode == "threshold":
                selected_handle = "aligned_lane_grouped" if req.input_len >= args.dynamic_seq_threshold else "cross_tp"
                lane = prefill_dp if selected_handle == "aligned_lane_grouped" else 0
            else:
                candidates: List[Tuple[float, str, int]] = []
                candidates.append((predict_option(req, p_end, "cross_tp", 0)[0], "cross_tp", 0))
                lanes = list(range(args.prefill_dps)) if args.dynamic_allow_any_aligned_lane else [prefill_dp]
                for ln in lanes:
                    candidates.append((predict_option(req, p_end, "aligned_lane_grouped", ln)[0], "aligned_lane_grouped", ln))
                _, selected_handle, lane = min(candidates, key=lambda x: x[0])
        else:
            raise ValueError(policy)

        _, t_start, t_end, d_start, first_decode_token, d_end = predict_option(req, p_end, selected_handle, lane)

        if selected_handle == "cross_tp":
            cross_transfer_available = t_end
            cross_decode_available = d_end
            decode_ms = args.decode_tp4_ms
            transfer_ms = profile.transfer_ms("cross_tp", req.input_len)
        else:
            grouped_transfer_available[lane] = t_end
            grouped_decode_available[lane] = d_end
            decode_ms = args.decode_tp2_ms
            transfer_ms = profile.transfer_ms("aligned_lane_grouped", req.input_len)

        ttft = p_end - req.arrival_ms
        handoff_gap = first_decode_token - p_end
        itl_proxy = max(handoff_gap, decode_ms)

        simulated.append(
            SimulatedRequest(
                policy=policy,
                req_id=req.req_id,
                arrival_ms=req.arrival_ms,
                input_len=req.input_len,
                output_len=req.output_len,
                prefill_dp=prefill_dp,
                selected_handle=selected_handle,
                decode_lane=lane,
                prefill_start_ms=p_start,
                prefill_end_ms=p_end,
                transfer_start_ms=t_start,
                transfer_end_ms=t_end,
                decode_start_ms=d_start,
                first_decode_token_ms=first_decode_token,
                decode_end_ms=d_end,
                prefill_queue_wait_ms=p_start - req.arrival_ms,
                transfer_queue_wait_ms=t_start - p_end,
                decode_queue_wait_ms=d_start - t_end,
                transfer_ms=transfer_ms,
                decode_ms_per_token=decode_ms,
                ttft_ms=ttft,
                handoff_gap_ms=handoff_gap,
                e2e_ms=d_end - req.arrival_ms,
                itl_proxy_ms=itl_proxy,
                ttft_slo_violation=ttft > args.ttft_slo_ms,
                itl_slo_violation=itl_proxy > args.itl_slo_ms,
            )
        )

    return simulated, summarize_rows(simulated, policy)


# =============================================================================
# Plotting
# =============================================================================


def save_bar(summary_df: pd.DataFrame, col: str, ylabel: str, title: str, out_path: Path, annotate: bool = True) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(summary_df["policy"], summary_df[col])
    ax.set_xlabel("Policy")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    if annotate:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_fraction_bar(summary_df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(summary_df))
    width = 0.35
    ax.bar(x - width / 2, summary_df["cross_tp_fraction"], width, label="cross_tp")
    ax.bar(x + width / 2, summary_df["grouped_fraction"], width, label="aligned_lane_grouped")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["policy"])
    ax.set_ylabel("Fraction")
    ax.set_title("Dynamic routing composition")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_timeline(trace_df: pd.DataFrame, out_path: Path, max_reqs: int = 80) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    shown = trace_df.sort_values(["policy", "req_id"]).groupby("policy").head(max_reqs)

    # Put each policy in a separated vertical band.
    y_offset = 0
    yticks: List[int] = []
    ylabels: List[str] = []
    for policy, g in shown.groupby("policy", sort=False):
        for _, r in g.iterrows():
            y = y_offset + int(r["req_id"])
            ax.plot([r["prefill_start_ms"], r["prefill_end_ms"]], [y, y], linewidth=2, label="prefill" if y_offset == 0 and int(r["req_id"]) == 0 else None)
            ax.plot([r["transfer_start_ms"], r["transfer_end_ms"]], [y, y], linewidth=2, label="transfer" if y_offset == 0 and int(r["req_id"]) == 0 else None)
            ax.plot([r["decode_start_ms"], r["decode_end_ms"]], [y, y], linewidth=2, label="decode" if y_offset == 0 and int(r["req_id"]) == 0 else None)
        yticks.append(y_offset)
        ylabels.append(policy)
        y_offset += max_reqs + 10

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Policy bands / request ids")
    ax.set_title(f"Request timeline, first {max_reqs} requests per policy")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all(summary_df: pd.DataFrame, trace_df: pd.DataFrame, out_dir: Path, timeline_max_requests: int) -> List[Path]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    generated = [
        save_bar(summary_df, "p99_handoff_gap_ms", "P99 handoff gap (ms)", "P99 handoff gap", fig_dir / "p99_handoff_gap.png"),
        save_bar(summary_df, "mean_decode_queue_wait_ms", "Mean decode queue wait (ms)", "Mean decode queue wait", fig_dir / "mean_decode_queue_wait.png"),
        save_bar(summary_df, "p99_decode_queue_wait_ms", "P99 decode queue wait (ms)", "P99 decode queue wait", fig_dir / "p99_decode_queue_wait.png"),
        save_bar(summary_df, "p99_e2e_ms", "P99 end-to-end latency (ms)", "P99 end-to-end latency", fig_dir / "p99_e2e.png"),
        save_bar(summary_df, "makespan_ms", "Makespan (ms)", "Makespan", fig_dir / "makespan.png"),
        save_bar(summary_df, "itl_slo_violation_rate", "ITL SLO violation rate", "ITL SLO violation rate", fig_dir / "itl_slo_violation_rate.png", annotate=False),
        save_fraction_bar(summary_df, fig_dir / "routing_fraction.png"),
        save_timeline(trace_df, fig_dir / "timeline_all_policies.png", max_reqs=timeline_max_requests),
    ]

    return generated


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(Path(args.summary), Path(args.results_dir))
    profile = TransferProfile(df)

    print("Loaded transfer profile:")
    for handle, pts in profile.points.items():
        print(f"  {handle}: {pts}")

    requests = generate_workload(args)
    req_df = pd.DataFrame([asdict(r) for r in requests])
    req_df.to_csv(out_dir / "generated_workload.csv", index=False)

    all_rows: List[SimulatedRequest] = []
    summaries: List[Dict[str, float | str | int]] = []
    for policy in ["cross_tp", "aligned_lane_grouped", "dynamic"]:
        rows, summary = simulate_policy(requests, policy, profile, args)
        all_rows.extend(rows)
        summaries.append(summary)

    trace_df = pd.DataFrame([asdict(r) for r in all_rows])
    summary_df = pd.DataFrame(summaries)

    trace_path = out_dir / "sim_request_trace.csv"
    summary_path = out_dir / "sim_summary.csv"
    trace_df.to_csv(trace_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    generated_figs = plot_all(summary_df, trace_df, out_dir, args.timeline_max_requests)

    print("\n=== Simulation summary ===")
    cols = [
        "policy",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_handoff_gap_ms",
        "p99_handoff_gap_ms",
        "mean_decode_queue_wait_ms",
        "p99_decode_queue_wait_ms",
        "p99_e2e_ms",
        "itl_slo_violation_rate",
        "cross_tp_fraction",
        "grouped_fraction",
        "makespan_ms",
    ]
    print(summary_df[cols].to_string(index=False))

    print("\nSaved outputs:")
    print(f"  {out_dir / 'generated_workload.csv'}")
    print(f"  {trace_path}")
    print(f"  {summary_path}")
    print(f"  {out_dir / 'figures'}")

    print("\nGenerated figures:")
    for p in generated_figs:
        status = "OK" if p.exists() and p.stat().st_size > 0 else "MISSING"
        print(f"  [{status}] {p}")

    # Write a small manifest for easy inspection.
    manifest = {
        "output_dir": str(out_dir),
        "summary_csv": str(summary_path),
        "trace_csv": str(trace_path),
        "figures": [str(p) for p in generated_figs],
        "args": vars(args),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
