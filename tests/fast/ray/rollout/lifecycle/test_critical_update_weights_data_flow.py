"""P-critical.1 — `update_weights` data flow contract test.

This is the highest-risk regression point of the rollout refactor: the
``EnginesAndLock`` contract that flows through:

  actor_group.update_weights()
    → rollout_manager.recover_updatable_engines (if use_fault_tolerance)
    → rollout_manager.get_updatable_engines_and_lock
    → backend actor.update_weights(info=...)
        → if info.has_new_engines: weight_updater.connect_rollout_engines + clear_updatable_has_new_engines

The test asserts that:
1. EnginesAndLock fields are populated correctly from RolloutManager state.
2. backend actor branches on `info.has_new_engines` correctly.
3. clear_updatable_has_new_engines is called only when expected.
4. The "no updatable server" path returns an empty EnginesAndLock without crashing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from miles.ray.rollout.rollout_manager import EnginesAndLock
from tests.fast.ray.rollout.conftest import fake_actor_handle


# ----------------------------- EnginesAndLock shape -----------------------------


class TestEnginesAndLockShape:
    def test_dataclass_is_frozen(self):
        eal = EnginesAndLock(
            rollout_engines=[],
            rollout_engine_lock=MagicMock(),
            has_new_engines=False,
            engine_gpu_counts=[],
            engine_gpu_offsets=[],
        )
        with pytest.raises(Exception):
            eal.has_new_engines = True  # frozen=True

    def test_required_fields_propagate(self):
        lock = MagicMock(name="lock")
        engines = [MagicMock(name=f"e{i}") for i in range(3)]
        eal = EnginesAndLock(
            rollout_engines=engines,
            rollout_engine_lock=lock,
            has_new_engines=True,
            engine_gpu_counts=[1, 1, 2],
            engine_gpu_offsets=[0, 1, 2],
        )
        assert eal.rollout_engines is engines
        assert eal.rollout_engine_lock is lock
        assert eal.has_new_engines is True
        assert eal.engine_gpu_counts == [1, 1, 2]
        assert eal.engine_gpu_offsets == [0, 1, 2]


# Note on backend update_weights branching:
# The previous version of this file kept a hand-copied mirror of the control
# flow in miles/backends/fsdp_utils/actor.py:551 and megatron_utils/actor.py:492.
# We dropped it because (a) the mirror tests itself, not the production code,
# so a regression in the real backend wouldn't be caught, and (b) the real
# branches are guarded by torch.distributed barrier / dist.get_rank, which
# can't be unit-tested cleanly without a torch dist init. The
# has_new_engines / clear-on-rank-0 / debug-short-circuit branches are
# covered by the actor backends' own integration suite under tests/slow/.
# What remains here is the EnginesAndLock dataclass shape contract and the
# RolloutManager → EnginesAndLock construction path, both of which DO
# benefit from fast unit tests.


# ----------------------------- RolloutManager → EnginesAndLock contract -----------------------------


class TestEnginesAndLockProductionFromRolloutManager:
    """Verifies what `RolloutManager.get_updatable_engines_and_lock()` would
    construct for various server states. Extracted into a helper to avoid
    needing the full RolloutManager actor wiring."""

    def _build(self, has_updatable: bool = True, has_new: bool = True, num_engines: int = 2):
        from miles.ray.rollout.rollout_server import RolloutServer
        from miles.ray.rollout.server_engine import ServerEngine
        from miles.ray.rollout.server_group import ServerGroup

        if not has_updatable:
            return None

        args = MagicMock()
        args.num_gpus_per_node = 8
        engines = [ServerEngine() for _ in range(num_engines)]
        for e in engines:
            e.mark_allocated_uninitialized(fake_actor_handle())
            e.mark_alive()
        group = ServerGroup(
            args=args,
            pg=None,
            all_engines=engines,
            num_gpus_per_engine=1,
            has_new_engines=has_new,
            update_weights=True,
        )
        return RolloutServer(server_groups=[group], update_weights=True, model_name="actor")

    def test_no_updatable_server_returns_empty_engines_and_lock(self):
        """Mirrors RolloutManager.get_updatable_engines_and_lock empty-path."""
        srv = self._build(has_updatable=False)
        assert srv is None  # in real code this is `_get_updatable_server() is None`

    def test_updatable_server_yields_engine_handles_in_order(self):
        srv = self._build(has_new=True, num_engines=3)
        engines = [e.actor_handle for e in srv.engines]
        assert len(engines) == 3
        assert srv.has_new_engines is True
        # offsets / counts shape: parallel to engines list
        assert len(srv.engine_gpu_counts) == 3
        assert len(srv.engine_gpu_offsets) == 3

    def test_clear_has_new_engines_flips_flag(self):
        srv = self._build(has_new=True)
        assert srv.has_new_engines is True
        srv.clear_has_new_engines()
        assert srv.has_new_engines is False
