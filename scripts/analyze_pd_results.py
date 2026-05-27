#!/usr/bin/env python3
"""Analyze concurrent PD dynamic-routing results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICIES = ["cross_tp", "aligned_lane_grouped", "dynamic"]
POLICY_LABELS = {
    "cross_tp": "Cross TP",
    "aligned_lane_grouped": "Grouped Lane",
    "dynamic": "Dynamic",
}
POLICY_COLORS = {
    "cross_tp": "#4C78A8",
    "aligned_lane_grouped": "#F58518",
    "dynamic": "#54A24B",
}
METRICS = [
    ("p99_handoff_gap_ms", "P99 handoff gap (ms)"),
    ("p99_decode_queue_wait_ms", "P99 decode queue wait (ms)"),
    ("p99_e2e_ms", "P99 E2E latency (ms)"),
    ("makespan_ms", "Makespan (ms)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PD concurrent benchmark results.")
    parser.add_argument("--root", type=str, default=".", help="Project root containing results/concurrent/nccl/.")
    parser.add_argument("--output-dir", type=str, default="results/report_assets")
    parser.add_argument("--show-values", action="store_true", help="Annotate bars with numeric values.")
    return parser.parse_args()


def split_case(case_name: str) -> Tuple[str, str]:
    for suffix in ("_chunk256", "_pack"):
        if case_name.endswith(suffix):
            return case_name[: -len(suffix)], suffix[1:]
    return case_name, "unknown"


def load_rows(root: Path) -> pd.DataFrame:
    results_root = root / "results" / "concurrent" / "nccl"
    rows: List[Dict[str, object]] = []
    for summary_path in sorted(results_root.glob("*/*/*/summary_*.json")):
        policy_dir = summary_path.parent.name
        case_dir = summary_path.parent.parent.name
        topology_dir = summary_path.parent.parent.parent.name
        workload, chunk_case = split_case(case_dir)

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        topology = payload.get("topology", {})
        args = payload.get("args", {})

        policy = str(summary.get("policy", args.get("policy", policy_dir)))
        row: Dict[str, object] = {
            "topology": str(topology.get("topology", args.get("topology_name", topology_dir))),
            "workload": workload,
            "chunk_case": chunk_case,
            "policy": policy,
            "num_requests": summary.get("num_requests", args.get("num_requests")),
            "request_level_filtering": bool(
                payload.get("request_level_filtering", summary.get("request_level_filtering", False))
            ),
            "summary_path": str(summary_path),
        }
        for key in [
            "mean_actual_transfer_ms",
            "p99_handoff_gap_ms",
            "p99_decode_queue_wait_ms",
            "p99_e2e_ms",
            "makespan_ms",
            "cross_tp_fraction",
            "grouped_fraction",
            "mean_decode_queue_wait_ms",
        ]:
            row[key] = summary.get(key, np.nan)
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No summary_*.json files found under {results_root}")

    df = pd.DataFrame(rows)
    df["policy"] = pd.Categorical(df["policy"], categories=POLICIES, ordered=True)
    df = df.sort_values(["topology", "workload", "chunk_case", "policy"]).reset_index(drop=True)
    return df


def case_label(row: pd.Series) -> str:
    return f"{row['topology']}\n{row['workload']}\n{row['chunk_case']}"


def annotate_bars(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out_base: Path, show_values: bool) -> None:
    cases = df[["topology", "workload", "chunk_case"]].drop_duplicates().reset_index(drop=True)
    x = np.arange(len(cases))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(12, len(cases) * 1.5), 5.5), dpi=180)
    for i, policy in enumerate(POLICIES):
        values = []
        for _, case in cases.iterrows():
            mask = (
                (df["topology"] == case["topology"])
                & (df["workload"] == case["workload"])
                & (df["chunk_case"] == case["chunk_case"])
                & (df["policy"] == policy)
            )
            values.append(float(df.loc[mask, metric].iloc[0]) if mask.any() else np.nan)
        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )
        if show_values:
            annotate_bars(ax, bars)

    labels = [f"{r.topology}\n{r.workload}\n{r.chunk_case}" for r in cases.itertuples()]
    ax.set_title(ylabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_latency_panel(df: pd.DataFrame, out_base: Path, show_values: bool) -> None:
    cases = df[["topology", "workload", "chunk_case"]].drop_duplicates().reset_index(drop=True)
    x = np.arange(len(cases))
    width = 0.24
    labels = [f"{r.topology}\n{r.workload}\n{r.chunk_case}" for r in cases.itertuples()]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=180)
    for ax, (metric, ylabel) in zip(axes.flat, METRICS):
        for i, policy in enumerate(POLICIES):
            values = []
            for _, case in cases.iterrows():
                mask = (
                    (df["topology"] == case["topology"])
                    & (df["workload"] == case["workload"])
                    & (df["chunk_case"] == case["chunk_case"])
                    & (df["policy"] == policy)
                )
                values.append(float(df.loc[mask, metric].iloc[0]) if mask.any() else np.nan)
            bars = ax.bar(x + (i - 1) * width, values, width, color=POLICY_COLORS[policy], label=POLICY_LABELS[policy])
            if show_values:
                annotate_bars(ax, bars)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.grid(axis="y", alpha=0.25)

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("PD Concurrent Routing Latency Panel", y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_dynamic_fraction(df: pd.DataFrame, out_base: Path, show_values: bool) -> None:
    work = df[df["policy"] == "dynamic"].copy()
    labels = [case_label(r) for _, r in work.iterrows()]
    x = np.arange(len(work))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(12, len(work) * 1.5), 5.5), dpi=180)
    bars1 = ax.bar(x - width / 2, work["cross_tp_fraction"], width, label="Cross TP fraction", color=POLICY_COLORS["cross_tp"])
    bars2 = ax.bar(
        x + width / 2,
        work["grouped_fraction"],
        width,
        label="Grouped lane fraction",
        color=POLICY_COLORS["aligned_lane_grouped"],
    )
    if show_values:
        annotate_bars(ax, bars1)
        annotate_bars(ax, bars2)
    ax.set_title("Dynamic Routing Fraction")
    ax.set_ylabel("Fraction of requests")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_transfer_vs_queue(df: pd.DataFrame, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    for policy in POLICIES:
        work = df[df["policy"] == policy]
        ax.scatter(
            work["mean_actual_transfer_ms"],
            work["p99_decode_queue_wait_ms"],
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
            s=70,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
        )
    ax.set_title("Transfer Time vs Decode Queue Tail")
    ax.set_xlabel("Mean actual transfer (ms)")
    ax.set_ylabel("P99 decode queue wait (ms)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "topology",
        "workload",
        "chunk_case",
        "policy",
        "mean_actual_transfer_ms",
        "p99_handoff_gap_ms",
        "p99_decode_queue_wait_ms",
        "p99_e2e_ms",
        "makespan_ms",
        "cross_tp_fraction",
        "grouped_fraction",
    ]
    table = df[cols].copy()
    for col in table.select_dtypes(include=[np.number]).columns:
        table[col] = table[col].map(lambda x: "" if not np.isfinite(x) else f"{x:.3f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in cols) + " |" for _, row in table.iterrows()]

    lines = [
        "# PD Concurrent Routing Summary",
        "",
        f"Rows analyzed: {len(df)}",
        "",
        header,
        sep,
        *rows,
        "",
        "Figures are saved under `results/report_assets/figures/` in PNG and PDF formats.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(root)
    csv_path = out_dir / "pd_concurrent_summary.csv"
    df.to_csv(csv_path, index=False)

    plot_latency_panel(df, fig_dir / "latency_panel", args.show_values)
    for metric, ylabel in METRICS:
        plot_metric(df, metric, ylabel, fig_dir / metric, args.show_values)
    plot_dynamic_fraction(df, fig_dir / "dynamic_routing_fraction", args.show_values)
    plot_transfer_vs_queue(df, fig_dir / "transfer_vs_queue")

    md_path = out_dir / "group_meeting_summary.md"
    write_markdown(df, md_path)

    print(f"saved_summary_csv : {csv_path}")
    print(f"saved_markdown    : {md_path}")
    print(f"saved_figures     : {fig_dir}")


if __name__ == "__main__":
    main()
