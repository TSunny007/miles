"""P1.1 (slim) — ServerGroup lifecycle, focused on what is NOT covered by
P-critical.3 (rank↔ports) or P-critical.4 (onload-from-disk).

Remaining concerns:
- start_engines short-circuit branches (debug_train_only / placeholder)
- start_engines `start_indices` subset
- start_engines second call skips already-allocated slots
- stop_engines + recover state transitions"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from miles.ray.rollout.addr_allocator import PortCursors
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.rollout.server_group import ServerGroup
from tests.fast.ray.rollout.conftest import fake_actor_handle, make_args


def _build_group(*, num_engines: int = 4, debug_train_only: bool = False,
                 worker_type: str = "regular") -> ServerGroup:
    args = make_args(num_gpus_per_node=8, debug_train_only=debug_train_only)
    engines = [ServerEngine() for _ in range(num_engines)] if worker_type != "placeholder" else []
    return ServerGroup(
        args=args,
        pg=None,
        all_engines=engines,
        num_gpus_per_engine=1,
        has_new_engines=False,
        worker_type=worker_type,
        needs_offload=False,
        update_weights=True,
    )


class TestStartEnginesShortCircuits:
    def test_debug_train_only_returns_immediately(self):
        group = _build_group(debug_train_only=True)
        handles, indices = group.start_engines(PortCursors.empty())
        assert handles == [] and indices == []
        assert group.has_new_engines is False
        for e in group.all_engines:
            assert not e.is_allocated

    def test_placeholder_worker_short_circuits(self):
        group = _build_group(worker_type="placeholder", num_engines=0)
        handles, indices = group.start_engines(PortCursors.empty())
        assert handles == [] and indices == []
        assert group.has_new_engines is False


class TestStartEnginesSubsetAndIdempotence:
    def test_start_indices_filters_to_subset(self, monkeypatch):
        """start_indices=[1,3] only allocates those slots; 0/2 stay Stopped."""
        group = _build_group(num_engines=4)

        # Stub out the real ray actor allocation
        actors = [fake_actor_handle() for _ in range(4)]
        for i, a in enumerate(actors):
            a.name = f"actor{i}"
        _stub_start_engines_internals(monkeypatch, group, actors)

        handles, indices = group.start_engines(PortCursors.empty(), start_indices=[1, 3])
        assert sorted(indices) == [1, 3]
        assert not group.all_engines[0].is_allocated
        assert group.all_engines[1].is_allocated
        assert not group.all_engines[2].is_allocated
        assert group.all_engines[3].is_allocated

    def test_already_allocated_slot_is_skipped(self, monkeypatch):
        """Engines that are already allocated must NOT be re-started."""
        group = _build_group(num_engines=2)

        # Pre-allocate slot 0
        existing_actor = fake_actor_handle()
        existing_actor.name = "existing"
        group.all_engines[0].mark_allocated_uninitialized(existing_actor)

        actors = [fake_actor_handle() for _ in range(2)]
        for i, a in enumerate(actors):
            a.name = f"new{i}"
        _stub_start_engines_internals(monkeypatch, group, actors)

        _handles, indices = group.start_engines(PortCursors.empty())
        assert 0 not in indices, "already-allocated slot must be skipped"
        # slot 0 still holds the existing actor (not replaced)
        assert group.all_engines[0].actor_handle is existing_actor


class TestStopEngines:
    def test_stop_marks_engines_stopped(self):
        group = _build_group(num_engines=3)
        for e in group.all_engines:
            actor = fake_actor_handle()
            actor.shutdown.remote = MagicMock(return_value=None)
            e.mark_allocated_uninitialized(actor)
            e.mark_alive()

        # Patch ray.get/ray.kill to no-op
        import miles.ray.rollout.server_group as mod
        with _patch_ray_calls(mod):
            group.stop_engines(engine_indices=[0, 1, 2])

        for e in group.all_engines:
            assert not e.is_allocated, "engine should be stopped"

    def test_stop_handles_shutdown_exception_gracefully(self):
        """If shutdown.remote raises, the engine is still marked stopped."""
        group = _build_group(num_engines=2)
        for e in group.all_engines:
            actor = fake_actor_handle()
            actor.shutdown.remote = MagicMock(side_effect=RuntimeError("boom"))
            e.mark_allocated_uninitialized(actor)
            e.mark_alive()

        import miles.ray.rollout.server_group as mod
        with _patch_ray_calls(mod, raise_on_get=True):
            # should not propagate; logger.warning is the only side effect
            group.stop_engines(engine_indices=[0, 1])

        for e in group.all_engines:
            assert not e.is_allocated


# ----------------------------- helpers -----------------------------


def _stub_start_engines_internals(monkeypatch, group: ServerGroup, fake_actors: list):
    """Replace ray.remote(SGLangEngine).options(...).remote(...) and ray.get
    with mocks so start_engines doesn't actually launch sglang."""
    import miles.ray.rollout.server_group as mod

    counter = {"i": 0}

    class _OptionsBuilder:
        def remote(self, *args, **kwargs):
            i = counter["i"]
            counter["i"] += 1
            return fake_actors[i % len(fake_actors)]

    class _ActorClass:
        def options(self, **kwargs):
            return _OptionsBuilder()

    monkeypatch.setattr(mod, "SGLangEngine", _ActorClass)
    monkeypatch.setattr(mod.ray, "remote", lambda *a, **k: _ActorClass())
    monkeypatch.setattr(mod.ray, "get", lambda x: ("10.0.0.1", 30000))

    # placeholder pg used by start_engines
    group.pg = (MagicMock(), [0, 1, 2, 3], [0, 1, 2, 3])

    # also replace allocate_rollout_engine_addr_and_ports_normal to return
    # trivial addr_and_ports so init.remote receives kwargs
    def fake_allocate(*args, **kwargs):
        engines = kwargs["rollout_engines"]
        rank_offset = kwargs.get("rank_offset", 0)
        addr_and_ports = {}
        for rank, _ in engines:
            addr_and_ports[rank] = dict(host="10.0.0.1", port=30000 + rank,
                                        nccl_port=31000 + rank,
                                        engine_info_bootstrap_port=32000 + rank,
                                        dist_init_addr=f"10.0.0.1:{33000 + rank}")
        return addr_and_ports, PortCursors(_values={0: 34000})

    monkeypatch.setattr(mod, "allocate_rollout_engine_addr_and_ports_normal", fake_allocate)


def _patch_ray_calls(mod, raise_on_get: bool = False):
    import contextlib

    @contextlib.contextmanager
    def cm():
        original_get = mod.ray.get
        original_kill = mod.ray.kill

        def _get(_x):
            if raise_on_get:
                raise RuntimeError("forced")
            return None

        mod.ray.get = _get
        mod.ray.kill = lambda actor: None
        try:
            yield
        finally:
            mod.ray.get = original_get
            mod.ray.kill = original_kill

    return cm()
