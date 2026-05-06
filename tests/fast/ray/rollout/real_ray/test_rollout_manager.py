"""``RolloutManager`` cell dispatch + ``EnginesAndLock`` flow driven through
the production ``RolloutManager`` actor (no subclassing).

The Ray worker is configured via ``runtime_env={"worker_process_setup_hook": ...}``
to patch the engine-spawning + subprocess-launching seams *inside the worker*
before any actor method runs. ``RolloutManager.__init__`` and
``start_rollout_servers`` then execute end-to-end against ``MockSGLangEngine``
actors.

Pure routing/flag-flip helpers without Ray content live in
``tests/fast/ray/rollout/test_rollout_manager.py``."""

from __future__ import annotations

import textwrap

import pytest
import ray

from miles.ray.rollout.rollout_manager import RolloutManager
from tests.fast.ray.rollout.conftest import make_args


def _worker_setup_for_rollout_manager_test() -> None:
    """Worker-process startup hook. Replaces in the worker:
    - ``SGLangEngine`` → ``MockSGLangEngine`` so created actors are mocks.
    - addr allocator → deterministic stub.
    - ``init_tracking`` / ``init_http_client`` / ``start_session_server`` /
      ``load_function`` / ``load_rollout_function`` → no-ops (so the
      ``RolloutManager.__init__`` paths that touch wandb / network /
      not-importable default function paths don't blow up)."""
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
    rmgr.load_function = lambda path: lambda *a, **kw: None
    rmgr.load_rollout_function = lambda input, path: lambda *a, **kw: None


_WORKER_HOOK_PATH = (
    "tests.fast.ray.rollout.real_ray.test_rollout_manager."
    "_worker_setup_for_rollout_manager_test"
)


def _make_manager(args, pg):
    """Spawn ``RolloutManager`` (production class — no subclass) on a worker
    that has been patched via the runtime_env setup hook."""
    return RolloutManager.options(
        runtime_env={"worker_process_setup_hook": _WORKER_HOOK_PATH}
    ).remote(args, pg)


def _write_sglang_config(tmp_path, *, models: list[tuple[str, bool]]) -> str:
    """Write a multi-model sglang yaml — each entry ``(name, update_weights)``.
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
    """Build args that drive ``RolloutManager.__init__`` →
    ``start_rollout_servers`` → N model servers each with 1 group of 2 mock
    engines."""
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


@pytest.mark.asyncio
class TestRolloutManagerInit:
    async def test_init_creates_live_mock_engines_via_real_start_rollout_servers(
        self, ray_local_mode, placement_group_factory, tmp_path,
    ):
        """End-to-end smoke: production ``__init__`` + ``start_rollout_servers``
        runs against MockSGLangEngine; resulting engines are reachable as Ray
        actor handles via the public ``get_updatable_engines_and_lock``."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        try:
            eal = ray.get(manager.get_updatable_engines_and_lock.remote())
            assert len(eal.rollout_engines) == 2
            for h in eal.rollout_engines:
                assert isinstance(h, ray.actor.ActorHandle)
                assert ray.get(h.health_generate.remote(timeout=1.0)) is True
        finally:
            ray.kill(manager)


@pytest.mark.asyncio
class TestStartStopCell:
    async def test_stop_cell_kills_target_engine_only(
        self, ray_local_mode, placement_group_factory, tmp_path,
    ):
        """``stop_cell.remote(0)`` dispatches via real Ray RPC and kills cell
        0's actor; cell 1 untouched."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        try:
            eal = ray.get(manager.get_updatable_engines_and_lock.remote())
            actor0, actor1 = eal.rollout_engines

            ray.get(manager.stop_cell.remote(0))

            with pytest.raises((ray.exceptions.RayActorError, ray.exceptions.RayTaskError)):
                ray.get(actor0.health_generate.remote(timeout=1.0), timeout=10.0)
            assert ray.get(actor1.health_generate.remote(timeout=1.0)) is True
        finally:
            ray.kill(manager)

    async def test_start_cell_recovers_after_stop_cell(
        self, ray_local_mode, placement_group_factory, tmp_path,
    ):
        """stop_cell(0) → start_cell(0) drives a real ``recover()`` that spawns
        a fresh mock actor in place of the killed one."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        try:
            eal_before = ray.get(manager.get_updatable_engines_and_lock.remote())
            actor0_before = eal_before.rollout_engines[0]

            ray.get(manager.stop_cell.remote(0))
            ray.get(manager.start_cell.remote(0))

            eal_after = ray.get(manager.get_updatable_engines_and_lock.remote())
            actor0_after = eal_after.rollout_engines[0]

            assert actor0_after is not actor0_before, "start_cell must produce a fresh actor"
            assert ray.get(actor0_after.health_generate.remote(timeout=1.0)) is True
        finally:
            ray.kill(manager)


@pytest.mark.asyncio
class TestGetUpdatableEnginesAndLock:
    async def test_returns_only_updatable_servers_engines_in_multi_model_setup(
        self, ray_local_mode, placement_group_factory, tmp_path,
    ):
        """With actor (update_weights=True) + ref (update_weights=False), the
        returned EnginesAndLock contains the actor's engines only."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        manager = _make_manager(args, pg)
        try:
            eal = ray.get(manager.get_updatable_engines_and_lock.remote())
            assert len(eal.rollout_engines) == 2  # actor's 2, not ref's 2
            assert eal.engine_gpu_counts == [1, 1]
            assert all(isinstance(h, ray.actor.ActorHandle) for h in eal.rollout_engines)
            assert ray.get(eal.rollout_engines[0].health_generate.remote(timeout=1.0)) is True
        finally:
            ray.kill(manager)
