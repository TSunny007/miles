"""``RolloutServer`` aggregation methods driven through real Ray actors.

Each engine is a real ``MockSGLangEngine`` actor. ``check_weights`` /
``offload`` / ``onload`` flow through real ``asyncio.gather`` over real Ray
ObjectRefs — the kind of bug that mock-only tests can't catch.

Cross-group property tests (``engines`` / ``engine_gpu_offsets`` / etc.) and
the heterogeneous ``nodes_per_engine`` ValueError don't need real actors
(they're pure dataclass logic) but they share the same ``ServerGroup`` shape,
so we keep them here to avoid a separate file."""

from __future__ import annotations

import asyncio

import pytest
import ray

from miles.ray.rollout.addr_allocator import PortCursors
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.rollout.server_group import ServerGroup
from tests.fast.ray.rollout.conftest import make_args


def _build_group(*, pg_tuple: tuple, num_engines: int = 2,
                 num_gpus_per_engine: int = 1, gpu_offset: int = 0,
                 needs_offload: bool = False) -> ServerGroup:
    args = make_args(num_gpus_per_node=8)
    engines = [ServerEngine() for _ in range(num_engines)]
    return ServerGroup(
        args=args, pg=pg_tuple, all_engines=engines,
        num_gpus_per_engine=num_gpus_per_engine, has_new_engines=False,
        gpu_offset=gpu_offset, update_weights=True, needs_offload=needs_offload,
    )


def _start_group(group: ServerGroup) -> None:
    handles, _ = group.start_engines(PortCursors.empty())
    ray.get(handles)


def _kill_group(group: ServerGroup) -> None:
    for e in group.all_engines:
        if e.is_allocated:
            ray.kill(e.actor_handle)


# ----------------------------- wait_all_engines_alive -----------------------------


@pytest.mark.asyncio
class TestWaitAllEnginesAlive:
    async def test_returns_immediately_when_all_alive(
        self, patched_sglang_engine, placement_group_factory,
    ):
        pg = placement_group_factory(2)
        group = _build_group(pg_tuple=pg, num_engines=2)
        _start_group(group)
        group.mark_alive([0, 1])
        srv = RolloutServer(server_groups=[group])
        try:
            await asyncio.wait_for(srv.wait_all_engines_alive(timeout=10), timeout=2.0)
        finally:
            _kill_group(group)

    async def test_raises_timeout_when_engine_never_becomes_alive(
        self, patched_sglang_engine, placement_group_factory, monkeypatch,
    ):
        pg = placement_group_factory(2)
        group = _build_group(pg_tuple=pg, num_engines=2)
        _start_group(group)
        # Don't mark_alive — the wait loop must time out.
        srv = RolloutServer(server_groups=[group])

        # Speed up the loop's sleep so the test finishes fast.
        async def _fast_sleep(_):
            return None
        monkeypatch.setattr("asyncio.sleep", _fast_sleep)

        try:
            with pytest.raises(TimeoutError):
                await srv.wait_all_engines_alive(timeout=2)
        finally:
            _kill_group(group)


# ----------------------------- check_weights -----------------------------


@pytest.mark.asyncio
class TestCheckWeightsAggregation:
    async def test_aggregates_across_groups_via_real_asyncio_gather(
        self, patched_sglang_engine, placement_group_factory,
    ):
        """Drives RolloutServer.check_weights through real ``asyncio.gather``
        over real Ray ObjectRefs. Verifies every engine in every group was
        actually invoked (read from each actor's call log)."""
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(3)
        a = _build_group(pg_tuple=pg_a, num_engines=2)
        b = _build_group(pg_tuple=pg_b, num_engines=3)
        _start_group(a)
        _start_group(b)
        a.mark_alive([0, 1])
        b.mark_alive([0, 1, 2])

        srv = RolloutServer(server_groups=[a, b])
        try:
            results = await srv.check_weights(action="report")

            # Outer gather: 2 groups → 2 inner lists; inner: 1 entry per engine
            assert len(results) == 2
            assert len(results[0]) == 2 and len(results[1]) == 3

            # Read each actor's call log and confirm check_weights ran with action="report"
            for g in (a, b):
                for e in g.engines:
                    calls = ray.get(e.actor_handle.get_calls.remote())
                    cw_calls = [c for c in calls if c[0] == "check_weights"]
                    assert len(cw_calls) == 1
                    _, args, _ = cw_calls[0]
                    assert args == ("report",)
        finally:
            _kill_group(a)
            _kill_group(b)


# ----------------------------- cross-group flatten + heterogeneity (no real actors) -----------------------------


class TestCrossGroupShapeProperties:
    """These properties are pure dataclass logic — no need for real actors,
    but the test scaffolding stays consistent with the rest of the file."""

    def test_engines_collects_node0_engines_from_each_group(self, placement_group_factory):
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(2)
        a = _build_group(pg_tuple=pg_a, num_engines=2, gpu_offset=0)
        b = _build_group(pg_tuple=pg_b, num_engines=2, gpu_offset=2)
        srv = RolloutServer(server_groups=[a, b])
        assert len(srv.engines) == 4

    def test_engine_gpu_counts_parallel_to_engines(self, placement_group_factory):
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(2)
        a = _build_group(pg_tuple=pg_a, num_engines=2, num_gpus_per_engine=1)
        b = _build_group(pg_tuple=pg_b, num_engines=2, num_gpus_per_engine=2)
        srv = RolloutServer(server_groups=[a, b])
        assert srv.engine_gpu_counts == [1, 1, 2, 2]

    def test_engine_gpu_offsets_consistent_across_groups(self, placement_group_factory):
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(2)
        a = _build_group(pg_tuple=pg_a, num_engines=2, num_gpus_per_engine=1, gpu_offset=0)
        b = _build_group(pg_tuple=pg_b, num_engines=2, num_gpus_per_engine=2, gpu_offset=4)
        srv = RolloutServer(server_groups=[a, b])
        assert srv.engine_gpu_offsets == [0, 1, 4, 6]


class TestNodesPerEngineHeterogeneity:
    def test_homogeneous_groups_return_single_value(self, placement_group_factory):
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(2)
        a = _build_group(pg_tuple=pg_a, num_gpus_per_engine=1)
        b = _build_group(pg_tuple=pg_b, num_gpus_per_engine=1)
        srv = RolloutServer(server_groups=[a, b])
        assert srv.nodes_per_engine == 1

    def test_heterogeneous_groups_raise_value_error(self, placement_group_factory):
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(2)
        a = _build_group(pg_tuple=pg_a, num_gpus_per_engine=1)   # 1 gpu/engine, 8 gpu/node → 1 node/engine
        b = _build_group(pg_tuple=pg_b, num_gpus_per_engine=16)  # 16 gpu/engine → 2 nodes/engine
        srv = RolloutServer(server_groups=[a, b])
        with pytest.raises(ValueError, match="Heterogeneous nodes_per_engine"):
            _ = srv.nodes_per_engine
