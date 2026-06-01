"""Replaceable execution facade for transfer plans."""

from __future__ import annotations

from dataclasses import dataclass

from .transfer_plan import TransferPlan


@dataclass(frozen=True)
class TransferResult:
    """Result reported by the simulated transfer backend."""

    request_id: str
    route_name: str
    transferred_bytes: int
    elapsed_time_ms: float
    simulated: bool = True


class TransferEngine:
    """Run a transfer plan without real KV movement in the standalone MVP."""

    def run(self, plan: TransferPlan) -> TransferResult:
        """Return the predicted execution result for a validated plan."""

        return TransferResult(
            request_id=plan.request_id,
            route_name=plan.route_name,
            transferred_bytes=plan.estimated_transfer_bytes,
            elapsed_time_ms=plan.estimated_transfer_time_ms,
        )
