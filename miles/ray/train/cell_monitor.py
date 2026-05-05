import time
from typing import TYPE_CHECKING

from miles.ray.train.cell_state import (
    CellState,
    StateAllocatedAlive,
    StateAllocatedErrored,
    StateAllocatedUninitialized,
    StatePending,
    StateStopped,
)
from miles.utils.control_server.models import CellCondition, CellStatus, TriState
from miles.utils.health_checker import SimpleHealthChecker, SimpleHealthCheckerConfig

if TYPE_CHECKING:
    from miles.ray.train.cell import RayTrainCell


def create_trainer_cell_health_checker(
    *,
    cell: "RayTrainCell",
    config: SimpleHealthCheckerConfig,
    max_heartbeat_age: float,
) -> SimpleHealthChecker:
    async def _check() -> None:
        if not cell.is_alive:
            return

        now = time.time()
        results = await cell.execute("get_heartbeat_status", mark_errored_on_failure=False)

        for status in results:
            delta = now - status.last_active_timestamp
            if delta > max_heartbeat_age:
                raise RuntimeError(
                    f"Heartbeat stale: last_active={status.last_active_timestamp:.1f}, "
                    f"now={now:.1f}, delta={delta:.1f}s, bump_count={status.bump_count}"
                )

    return SimpleHealthChecker(
        name=f"trainer-cell-{cell.cell_index}",
        check_fn=_check,
        config=config,
    )


def compute_cell_status(state: CellState, health_checker_status: TriState) -> CellStatus:
    match state:
        case StateAllocatedAlive():
            match health_checker_status:
                case TriState.FALSE:
                    healthy = CellCondition.healthy(TriState.FALSE, reason="HealthCheckFailed")
                case TriState.UNKNOWN:
                    healthy = CellCondition.healthy(TriState.UNKNOWN, reason="HealthCheckUnknown")
                case TriState.TRUE:
                    healthy = CellCondition.healthy(TriState.TRUE)
            return CellStatus(phase="Running", conditions=[CellCondition.allocated(TriState.TRUE), healthy])

        case StateAllocatedUninitialized():
            return CellStatus(
                phase="Running",
                conditions=[
                    CellCondition.allocated(TriState.TRUE),
                    CellCondition.healthy(TriState.TRUE),
                ],
            )

        case StateAllocatedErrored():
            return CellStatus(
                phase="Running",
                conditions=[
                    CellCondition.allocated(TriState.TRUE),
                    CellCondition.healthy(TriState.FALSE, reason="ExecutionErrored"),
                ],
            )

        case StatePending():
            return CellStatus(phase="Pending", conditions=[CellCondition.allocated(TriState.FALSE)])

        case StateStopped():
            return CellStatus(phase="Suspended", conditions=[CellCondition.allocated(TriState.FALSE)])

        case _:
            raise NotImplementedError(f"Unknown state: {state}")
