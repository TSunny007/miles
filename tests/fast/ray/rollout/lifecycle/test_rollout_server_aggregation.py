"""P1.2 (slim) — RolloutServer aggregation.

After P-critical.2 trim, this file covers:
- wait_all_engines_alive timeout path
- check_weights aggregation across multi-group
- engines / engine_gpu_offsets / engine_gpu_counts: cross-group flatten
- nodes_per_engine heterogeneous-group ValueError"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.rollout.server_group import ServerGroup
from tests.fast.ray.rollout.conftest import fake_actor_handle, make_args


def _build_group(*, num_engines: int = 2, num_gpus_per_engine: int = 1, gpu_offset: int = 0,
                 alive: bool = True) -> ServerGroup:
    args = make_args(num_gpus_per_node=8)
    engines = [ServerEngine() for _ in range(num_engines)]
    for e in engines:
        actor = fake_actor_handle()
        actor.check_weights.remote = MagicMock(return_value="cw_ref")
        e.mark_allocated_uninitialized(actor)
        if alive:
            e.mark_alive()
    return ServerGroup(
        args=args, pg=None, all_engines=engines,
        num_gpus_per_engine=num_gpus_per_engine, has_new_engines=False,
        gpu_offset=gpu_offset, update_weights=True,
    )


@pytest.mark.asyncio
class TestWaitAllEnginesAlive:
    async def test_returns_immediately_when_all_alive(self):
        group = _build_group(num_engines=3, alive=True)
        srv = RolloutServer(server_groups=[group])
        # asyncio.sleep is the slow path; we just want it to return without timeout
        await asyncio.wait_for(srv.wait_all_engines_alive(timeout=10), timeout=2.0)

    async def test_raises_timeout_when_engine_never_becomes_alive(self, monkeypatch):
        group = _build_group(num_engines=2, alive=False)  # engines never alive
        srv = RolloutServer(server_groups=[group])

        # speed up the loop's sleep to make the test fast
        async def _fast_sleep(_):
            return None
        monkeypatch.setattr("asyncio.sleep", _fast_sleep)

        with pytest.raises(TimeoutError):
            await srv.wait_all_engines_alive(timeout=2)


@pytest.mark.asyncio
class TestCheckWeightsAggregation:
    async def test_check_weights_aggregates_across_groups(self):
        """Drives RolloutServer.check_weights through real asyncio.gather.
        Each ``actor_handle.check_weights.remote(...)`` returns an awaitable
        Future so the gather chain resolves; we then verify *every* allocated
        engine across both groups was invoked with the right action kwarg."""
        group_a = _build_group(num_engines=2)
        group_b = _build_group(num_engines=3)
        srv = RolloutServer(server_groups=[group_a, group_b])

        # Replace the canned "cw_ref" string return from _build_group with a
        # fresh-future side_effect so asyncio.gather can await it.
        def _make_fresh(value):
            def _fresh(*a, **kw):
                f = asyncio.Future()
                f.set_result(value)
                return f
            return _fresh
        for g in (group_a, group_b):
            for e in g.engines:
                e.actor_handle.check_weights.remote = MagicMock(side_effect=_make_fresh(f"cw:{id(e)}"))

        results = await srv.check_weights(action="report")

        # 2 groups → outer gather returns 2 inner lists.
        assert len(results) == 2
        # Each inner list has one entry per allocated engine.
        assert len(results[0]) == 2 and len(results[1]) == 3
        # Every engine got called with the right kwarg.
        for g in (group_a, group_b):
            for e in g.engines:
                e.actor_handle.check_weights.remote.assert_called_with(action="report")


class TestCrossGroupFlatten:
    def test_engines_collects_node0_engines_from_each_group(self):
        a = _build_group(num_engines=2, num_gpus_per_engine=1, gpu_offset=0)
        b = _build_group(num_engines=2, num_gpus_per_engine=1, gpu_offset=2)
        srv = RolloutServer(server_groups=[a, b])
        assert len(srv.engines) == 4

    def test_engine_gpu_counts_parallel_to_engines(self):
        a = _build_group(num_engines=2, num_gpus_per_engine=1)
        b = _build_group(num_engines=2, num_gpus_per_engine=2)
        srv = RolloutServer(server_groups=[a, b])
        assert srv.engine_gpu_counts == [1, 1, 2, 2]

    def test_engine_gpu_offsets_consistent_across_groups(self):
        a = _build_group(num_engines=2, num_gpus_per_engine=1, gpu_offset=0)
        b = _build_group(num_engines=2, num_gpus_per_engine=2, gpu_offset=4)
        srv = RolloutServer(server_groups=[a, b])
        # group A: 0, 1; group B: 4, 6
        assert srv.engine_gpu_offsets == [0, 1, 4, 6]


class TestNodesPerEngineHeterogeneity:
    def test_homogeneous_groups_return_single_value(self):
        a = _build_group(num_gpus_per_engine=1)
        b = _build_group(num_gpus_per_engine=1)
        srv = RolloutServer(server_groups=[a, b])
        assert srv.nodes_per_engine == 1

    def test_heterogeneous_groups_raise_value_error(self):
        # group A: 1 gpu/engine on 8-gpu node → nodes_per_engine = 1
        # group B: 16 gpus/engine on 8-gpu node → nodes_per_engine = 2
        a = _build_group(num_gpus_per_engine=1)
        b = _build_group(num_gpus_per_engine=16)
        srv = RolloutServer(server_groups=[a, b])
        with pytest.raises(ValueError, match="Heterogeneous nodes_per_engine"):
            _ = srv.nodes_per_engine
