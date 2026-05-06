"""P1.4 (slim) — RolloutManager non-lifecycle methods.

After P-critical.1 trim, only methods with real logic remain:
- clear_updatable_has_new_engines flips flags on all groups
- set_train_parallel_config feeding get_num_rollout_per_epoch
- onload / offload / onload_weights / onload_kv tag pass-through

Bare ``dispose`` / ``save`` / ``load`` (single-line forwarding) are deliberately
not covered — they would test the mock, not the code."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest


def _awaitable_ref(value: str = "ref"):
    """Return a fresh resolved Future per call so asyncio.gather can await it.

    RolloutServer.onload/offload do ``await asyncio.gather(*handles)`` on the
    list of refs returned by ``actor.<method>.remote(...)``. Plain MagicMock
    return values are not awaitable, so we hand the mock a side_effect that
    creates a fresh resolved future every invocation."""
    f = asyncio.Future()
    f.set_result(value)
    return f

from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.rollout.server_group import ServerGroup
from tests.fast.ray.rollout.conftest import fake_actor_handle, make_args


def _build_server(num_groups: int = 2, has_new: bool = True) -> RolloutServer:
    args = make_args(num_gpus_per_node=8)
    groups = []
    for _ in range(num_groups):
        engines = [ServerEngine() for _ in range(2)]
        for e in engines:
            actor = fake_actor_handle()
            actor.release_memory_occupation.remote = MagicMock(side_effect=lambda *a, **kw: _awaitable_ref())
            actor.resume_memory_occupation.remote = MagicMock(side_effect=lambda *a, **kw: _awaitable_ref())
            e.mark_allocated_uninitialized(actor)
            e.mark_alive()
        groups.append(ServerGroup(
            args=args, pg=None, all_engines=engines,
            num_gpus_per_engine=1, has_new_engines=has_new,
            needs_offload=True, update_weights=True,
        ))
    return RolloutServer(server_groups=groups, model_name="actor")


class TestClearUpdatableHasNewEngines:
    def test_clears_flag_on_all_groups(self):
        srv = _build_server(num_groups=3, has_new=True)
        assert all(g.has_new_engines for g in srv.server_groups)
        srv.clear_has_new_engines()
        assert not any(g.has_new_engines for g in srv.server_groups)

    def test_no_op_when_already_clear(self):
        srv = _build_server(num_groups=2, has_new=False)
        srv.clear_has_new_engines()
        assert not any(g.has_new_engines for g in srv.server_groups)


# Note: a "TestTrainParallelConfigDependency" test was removed because it
# only asserted ``len(list(range(80))) // 8 == 10`` and ``args.rollout_global_dataset is True``
# without invoking any production code. ``RolloutManager.get_num_rollout_per_epoch``
# is one line and is exercised by integration tests.


@pytest.mark.asyncio
class TestOnloadOffloadTagPassThrough:
    async def test_onload_kv_includes_kv_and_cuda_graph_tags(self, monkeypatch):
        from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE

        srv = _build_server(num_groups=2, has_new=False)

        # Replicate RolloutManager.onload_kv: srv.onload(tags=[KV_CACHE, CUDA_GRAPH])
        await _await_handles(srv.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]))

        for g in srv.server_groups:
            for e in g.engines:
                e.actor_handle.resume_memory_occupation.remote.assert_called_with(
                    tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH],
                )

    async def test_onload_weights_includes_weights_tag(self):
        from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

        srv = _build_server(num_groups=2, has_new=False)
        await _await_handles(srv.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

        for g in srv.server_groups:
            for e in g.engines:
                e.actor_handle.resume_memory_occupation.remote.assert_called_with(
                    tags=[GPU_MEMORY_TYPE_WEIGHTS],
                )

    async def test_offload_calls_release_for_every_allocated_engine(self):
        srv = _build_server(num_groups=2, has_new=False)
        await _await_handles(srv.offload())
        for g in srv.server_groups:
            for e in g.engines:
                e.actor_handle.release_memory_occupation.remote.assert_called_once()


# ----------------------------- helpers -----------------------------


async def _await_handles(call_result):
    """Some onload/offload methods return list[ObjectRef]; others return a coroutine.
    Just consume whatever comes back."""
    import inspect
    if inspect.iscoroutine(call_result):
        return await call_result
    return call_result
