"""``RolloutManager`` cell dispatch + ``EnginesAndLock`` flow driven through
a real Ray actor.

We instantiate ``_RolloutManagerForTest`` (a Ray-actor subclass of
``RolloutManager``) which patches the engine-spawning lower level inside
the worker process and then calls ``super().__init__(args, pg)`` — so the
production ``RolloutManager.__init__`` and ``start_rollout_servers`` paths
run end-to-end against ``MockSGLangEngine`` actors.

Pure routing/flag-flip helpers without Ray content live in
``tests/fast/ray/rollout/test_rollout_manager.py``."""

from __future__ import annotations

import textwrap

import pytest
import ray

from miles.ray.rollout.rollout_manager import RolloutManager
from miles.ray.rollout.rollout_server import RolloutServer
from tests.fast.ray.rollout.conftest import make_args


def _patch_low_level_in_current_process() -> None:
    """Run inside the worker before ``RolloutManager.__init__``. Replaces:
    - ``SGLangEngine`` with ``MockSGLangEngine`` so created actors are mocks.
    - The addr allocator with a deterministic stub.
    - The router/session-server/tracking/http-client init bits with no-ops
      (they would otherwise spawn subprocesses or initialise wandb)."""
    import miles.ray.rollout.rollout_manager as rmgr
    import miles.ray.rollout.server_group as sg
    from miles.ray.rollout.addr_allocator import PortCursors
    from miles.utils.test_utils.mock_sglang_engine import MockSGLangEngine

    sg.SGLangEngine = MockSGLangEngine.__ray_actor_class__

    def _fake_alloc(*args, **kwargs):
        engines = kwargs["rollout_engines"]
        return (
            {
                rank: dict(
                    host="127.0.0.1",
                    port=30000 + rank,
                    nccl_port=31000 + rank,
                    engine_info_bootstrap_port=32000 + rank,
                    dist_init_addr=f"127.0.0.1:{33000 + rank}",
                )
                for rank, _ in engines
            },
            PortCursors(_values={0: 34000}),
        )

    sg.allocate_rollout_engine_addr_and_ports_normal = _fake_alloc

    rmgr.init_tracking = lambda *a, **kw: None
    rmgr.init_http_client = lambda args: None
    rmgr.start_session_server = lambda args: None
    # default args.data_source_path points at miles.data.dummy which doesn't
    # exist in this repo; default rollout/eval function paths are similarly
    # not all importable. None of these are exercised by the routing tests,
    # so swap with no-op stubs.
    rmgr.load_function = lambda path: lambda *a, **kw: None
    rmgr.load_rollout_function = lambda input, path: lambda *a, **kw: None


@ray.remote
class _RolloutManagerForTest(RolloutManager.__ray_actor_class__):
    """Real Ray actor running production ``RolloutManager.__init__`` against
    mock engines. Patches happen in this worker process before super().__init__."""

    def __init__(self, args, pg):
        _patch_low_level_in_current_process()
        super().__init__(args, pg)

    def _peek_servers(self) -> dict[str, RolloutServer]:
        """Test-only inspector: Ray copies state at ``__init__``, mutations
        stay in the worker, so tests pull the worker's view back."""
        return self.servers

    def _kill_engine_at_cell(self, cell_id: int) -> None:
        """Test helper: ray.kill the engine for ``cell_id`` and mark its slot
        stopped so a follow-up ``start_cell`` will recover it."""
        from miles.ray.rollout.server_cell import get_cell_indexer_of_id_map
        idx = get_cell_indexer_of_id_map(self.servers)[cell_id]
        group = self.servers[idx.srv_key].server_groups[idx.group_index]
        for ei in idx.engine_indices:
            engine = group.all_engines[ei]
            if engine.is_allocated:
                ray.kill(engine.actor_handle)
                engine.mark_stopped()


def _write_sglang_config(tmp_path, *, models: list[tuple[str, bool]]) -> str:
    """Write a multi-model sglang yaml: each entry is ``(name, update_weights)``.
    Each model gets one regular group with 2 engines × 1 GPU = 2 GPUs. With N
    models, total GPUs = 2N; ``args.rollout_num_gpus`` must match."""
    lines = ["sglang:"]
    for name, update_weights in models:
        lines.extend([
            f"  - name: {name}",
            f"    update_weights: {str(update_weights).lower()}",
            "    server_groups:",
            "      - worker_type: regular",
            "        num_gpus: 2",
            "        num_gpus_per_engine: 1",
        ])
    cfg_path = tmp_path / "sglang.yaml"
    cfg_path.write_text(textwrap.dedent("\n".join(lines)) + "\n")
    return str(cfg_path)


def _make_test_args(tmp_path, *, models: list[tuple[str, bool]]):
    """Build args that drive RolloutManager.__init__ → start_rollout_servers →
    create N model servers each with 1 group of 2 mock engines."""
    cfg = _write_sglang_config(tmp_path, models=models)
    rollout_num_gpus = 2 * len(models)
    return make_args(
        sglang_config=cfg,
        rollout_num_gpus=rollout_num_gpus,
        # short-circuit start_router (returns early when ip+port already set)
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        # disable everything else that would spawn subprocesses or hit network
        use_session_server=False,
        use_fault_tolerance=False,
        use_wandb=False,
        use_tensorboard=False,
        use_mlflow=False,
        use_distributed_post=False,
        sglang_server_concurrency=1,
    )


def _kill_all_engines(servers: dict[str, RolloutServer]) -> None:
    for srv in servers.values():
        for g in srv.server_groups:
            for e in g.all_engines:
                if e.is_allocated:
                    try:
                        ray.kill(e.actor_handle)
                    except Exception:
                        pass


@pytest.mark.asyncio
class TestRolloutManagerInit:
    async def test_init_creates_mock_servers_via_real_start_rollout_servers(
        self, ray_local_mode, placement_group_factory, tmp_path,
    ):
        """End-to-end: production ``__init__`` + ``start_rollout_servers``
        runs against MockSGLangEngine and produces live mock actors."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _RolloutManagerForTest.remote(args, pg)
        try:
            servers = ray.get(manager._peek_servers.remote())
            assert set(servers.keys()) == {"actor"}
            assert len(servers["actor"].server_groups) == 1
            assert len(servers["actor"].server_groups[0].all_engines) == 2
            for e in servers["actor"].server_groups[0].all_engines:
                assert e.is_allocated
                # Each engine actor was actually init()'d via the mock.
                calls = ray.get(e.actor_handle.get_calls.remote())
                assert "init" in [c[0] for c in calls]
        finally:
            _kill_all_engines(ray.get(manager._peek_servers.remote()))
            ray.kill(manager)


@pytest.mark.asyncio
class TestStartStopCell:
    async def test_start_cell_recovers_a_killed_engine(
        self, ray_local_mode, placement_group_factory, tmp_path,
    ):
        """Kill engine 0 + mark stopped, then ``start_cell.remote(0)`` runs
        ``recover()`` which spawns a fresh mock actor and init()'s it."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _RolloutManagerForTest.remote(args, pg)
        try:
            ray.get(manager._kill_engine_at_cell.remote(0))
            ray.get(manager.start_cell.remote(0))

            servers = ray.get(manager._peek_servers.remote())
            engine = servers["actor"].server_groups[0].all_engines[0]
            assert engine.is_allocated
            calls = ray.get(engine.actor_handle.get_calls.remote())
            assert "init" in [c[0] for c in calls]
        finally:
            _kill_all_engines(ray.get(manager._peek_servers.remote()))
            ray.kill(manager)

    async def test_stop_cell_kills_target_engine_only(
        self, ray_local_mode, placement_group_factory, tmp_path,
    ):
        """``stop_cell.remote(0)`` dispatched via Ray RPC kills cell 0's actor
        for real; cell 1's actor stays alive."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _RolloutManagerForTest.remote(args, pg)
        try:
            servers_before = ray.get(manager._peek_servers.remote())
            actor0 = servers_before["actor"].server_groups[0].all_engines[0].actor_handle
            actor1 = servers_before["actor"].server_groups[0].all_engines[1].actor_handle

            ray.get(manager.stop_cell.remote(0))

            with pytest.raises((ray.exceptions.RayActorError, ray.exceptions.RayTaskError)):
                ray.get(actor0.health_generate.remote(timeout=1.0), timeout=10.0)
            assert ray.get(actor1.health_generate.remote(timeout=1.0)) is True
        finally:
            _kill_all_engines(ray.get(manager._peek_servers.remote()))
            ray.kill(manager)


@pytest.mark.asyncio
class TestGetUpdatableEnginesAndLock:
    async def test_returns_populated_for_updatable_server_in_multi_model_setup(
        self, ray_local_mode, placement_group_factory, tmp_path,
    ):
        """With actor (update_weights=True) + ref (update_weights=False), the
        returned EnginesAndLock contains the actor's engines only."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        manager = _RolloutManagerForTest.remote(args, pg)
        try:
            eal = ray.get(manager.get_updatable_engines_and_lock.remote())
            assert len(eal.rollout_engines) == 2  # actor's 2 engines, not ref's
            assert eal.engine_gpu_counts == [1, 1]
            assert all(isinstance(h, ray.actor.ActorHandle) for h in eal.rollout_engines)
            # Live mock actor: a real call round-trips.
            assert ray.get(eal.rollout_engines[0].health_generate.remote(timeout=1.0)) is True
        finally:
            _kill_all_engines(ray.get(manager._peek_servers.remote()))
            ray.kill(manager)
