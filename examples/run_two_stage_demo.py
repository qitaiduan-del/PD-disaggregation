#!/usr/bin/env python3
"""Run a small standalone two-stage PD scheduling workload."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pd_disaggregation.core import (  # noqa: E402
    DynamicPDRouter,
    KVMetadata,
    Request,
    SLOCostModel,
    SystemState,
    TransferEngine,
    TransferPlanner,
    combine_stage_routes,
)


def demo_workload() -> list[tuple[Request, SystemState]]:
    """Return requests spanning idle, long-prompt, and burst-pressure cases."""

    definitions = (
        ("req-001", 512, 64, 0.0, 180.0, 850.0, 0, 0),
        ("req-002", 1024, 96, 5.0, 220.0, 1100.0, 1, 1),
        ("req-003", 8192, 96, 8.0, 270.0, 1500.0, 3, 2),
        ("req-004", 4096, 128, 10.0, 320.0, 1800.0, 4, 12),
        ("req-005", 2048, 192, 11.0, 360.0, 2300.0, 7, 18),
        ("req-006", 6144, 128, 13.0, 340.0, 1900.0, 6, 20),
    )
    workload = []
    for request_id, prompt, output, arrival, ttft, e2e, pq, dq in definitions:
        request = Request(request_id, prompt, output, arrival, ttft, e2e)
        state = SystemState(
            prefill_queue_len=pq,
            decode_queue_len=dq,
            active_prefill_workers=2,
            active_decode_workers=2,
            gpu_count=8,
            current_time_ms=arrival,
        )
        workload.append((request, state))
    return workload


def main() -> None:
    """Route requests, build simulated transfers, and print aggregate metrics."""

    router = DynamicPDRouter()
    planner = TransferPlanner(router.cost_model.profiler)
    engine = TransferEngine()
    cost_model: SLOCostModel = router.cost_model
    records: list[tuple[float, float, bool, str, str]] = []

    print("Two-stage PD disaggregation scheduling demo")
    print("=" * 154)
    print(
        f"{'request_id':<10} {'prompt':>7} {'output':>7} {'prefill route':<16} "
        f"{'decode route':<14} {'prefill ms':>11} {'transfer ms':>12} "
        f"{'decode ms':>11} {'TTFT ms':>10} {'E2E ms':>10} {'SLO violation':>14}"
    )

    for request, state in demo_workload():
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
            request, state, stage="decode", kv_meta=kv_meta
        )
        resolved_route = combine_stage_routes(prefill_route, decode_route)
        plan = planner.build_transfer_plan(request, resolved_route, kv_meta)
        engine.run(plan)
        estimate = cost_model.estimate(request, state, resolved_route, kv_meta)
        decode_ms = estimate.decode_time_per_token_ms * request.output_len
        records.append(
            (
                estimate.total_ttft_ms,
                estimate.total_e2e_latency_ms,
                estimate.slo_violation,
                prefill_route.name,
                decode_route.name,
            )
        )
        print(
            f"{request.request_id:<10} {request.prompt_len:>7} {request.output_len:>7} "
            f"{prefill_route.name:<16} {decode_route.name:<14} "
            f"{estimate.prefill_time_ms:>11.2f} {plan.estimated_transfer_time_ms:>12.2f} "
            f"{decode_ms:>11.2f} {estimate.total_ttft_ms:>10.2f} "
            f"{estimate.total_e2e_latency_ms:>10.2f} {str(estimate.slo_violation):>14}"
        )

    prefill_distribution = Counter(record[3] for record in records)
    decode_distribution = Counter(record[4] for record in records)
    print("\nSummary")
    print("-" * 48)
    print(f"total requests       : {len(records)}")
    print(f"SLO violation count  : {sum(record[2] for record in records)}")
    print(f"average TTFT ms      : {mean(record[0] for record in records):.2f}")
    print(f"average E2E latency ms: {mean(record[1] for record in records):.2f}")
    print(f"prefill routes       : {dict(prefill_distribution)}")
    print(f"decode routes        : {dict(decode_distribution)}")


if __name__ == "__main__":
    main()
