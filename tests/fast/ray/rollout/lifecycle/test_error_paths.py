"""P1.5 (slim) — error paths.

After "engine.init mid-stream raise" was dropped (testing the mock, not the code),
this file keeps:
- shutdown raise: stop_engines must still mark all engines stopped
- release_memory_occupation failure during recover: documented behavior

Plus a known-limitation snapshot test for simulate_crash → is_allocated."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.rollout.server_group import ServerGroup
from tests.fast.ray.rollout.conftest import fake_actor_handle, make_args


def _build_group(num_engines: int = 3) -> ServerGroup:
    args = make_args(num_gpus_per_node=8)
    engines = [ServerEngine() for _ in range(num_engines)]
    for e in engines:
        actor = fake_actor_handle()
        actor.shutdown.remote = MagicMock(return_value="ref")
        e.mark_allocated_uninitialized(actor)
        e.mark_alive()
    return ServerGroup(
        args=args, pg=None, all_engines=engines,
        num_gpus_per_engine=1, has_new_engines=False, update_weights=True,
    )


class TestShutdownExceptionDoesNotBlockOtherEngines:
    def test_shutdown_raises_but_all_engines_stopped(self, monkeypatch):
        """If engine.shutdown raises, the loop must still mark all engines stopped."""
        group = _build_group(num_engines=3)

        # Make engine[1].shutdown raise via ray.get
        import miles.ray.rollout.server_group as mod
        call_log = []

        def fake_get(_x):
            call_log.append("get")
            if len(call_log) == 2:  # 2nd call (engine[1])
                raise RuntimeError("boom")
            return None

        monkeypatch.setattr(mod.ray, "get", fake_get)
        monkeypatch.setattr(mod.ray, "kill", lambda actor: None)

        group.stop_engines(engine_indices=[0, 1, 2])

        # All 3 should be marked stopped despite engine[1] failing
        for e in group.all_engines:
            assert not e.is_allocated, "even after exception, engine must be marked stopped"


class TestReleaseMemoryFailureDuringRecover:
    """Known: ``ServerGroup.recover`` calls release_memory_occupation, then resume.
    If release raises, asyncio.gather propagates and resume is skipped.

    Behavior to document (not fix in this PR): the engine state is left in a
    half-recovered state. The test asserts current behavior so a future change
    is intentional."""

    @pytest.mark.asyncio
    async def test_release_raise_propagates(self, monkeypatch):
        group = _build_group(num_engines=2)
        for e in group.all_engines:
            e.mark_stopped()

        # Stub start_engines so recover() reaches the release/resume phase
        def fake_start(port_cursors, start_indices=None):
            idxs = start_indices or list(range(len(group.all_engines)))
            for i in idxs:
                actor = fake_actor_handle()
                actor.release_memory_occupation.remote = MagicMock(side_effect=RuntimeError("rel-boom"))
                actor.resume_memory_occupation.remote = MagicMock(return_value="r")
                group.all_engines[i].mark_allocated_uninitialized(actor)
            return [], idxs

        monkeypatch.setattr(group, "start_engines", fake_start)
        group.needs_offload = True

        from miles.ray.rollout.addr_allocator import PortCursors
        with pytest.raises(RuntimeError, match="rel-boom"):
            await group.recover(port_cursors=PortCursors.empty())


class TestServerEngineExplicitStateContract:
    """Pin the by-design contract: ``ServerEngine`` state transitions are
    explicit. The state machine does not (and should not) infer "actor died"
    from out-of-band ray actor termination — callers (``stop_engines`` /
    ``recover``) are responsible for invoking ``mark_stopped`` themselves.

    This is documented intent on ``ServerEngine`` itself; the test pins it
    so that adding implicit liveness probing later requires deleting this
    test deliberately rather than silently changing behavior."""

    def test_state_does_not_auto_change_when_actor_handle_is_discarded(self):
        e = ServerEngine()
        actor = fake_actor_handle()
        e.mark_allocated_uninitialized(actor)
        e.mark_alive()
        # Discarding the mock (or having a real actor die) does NOT update state.
        assert e.is_allocated is True
        assert e.is_alive is True
        # Only an explicit transition flips the bits.
        e.mark_stopped()
        assert not e.is_allocated
