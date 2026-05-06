"""``RolloutManager`` cell dispatch + ``EnginesAndLock`` flow driven through
a real Ray actor.

We instantiate ``_RolloutManagerForTest`` (a Ray-actor subclass of
``RolloutManager``) so that ``start_cell.remote()`` / ``stop_cell.remote()``
/ ``get_updatable_engines_and_lock.remote()`` actually dispatch through Ray
RPC. Production ``RolloutManager.__init__`` calls ``start_rollout_servers``
which is heavy and creates real SGLang engines; the subclass overrides
``__init__`` to take pre-built servers AND patches ``SGLangEngine`` + the
addr allocator inside the worker process so any ``group.recover()`` driven
by inherited methods still spawns mock actors.

Pure routing/flag-flip helpers without Ray content live in
``tests/fast/ray/rollout/test_rollout_manager.py``."""

from __future__ import annotations

import pytest
import ray

from miles.ray.rollout.addr_allocator import PortCursors
from miles.ray.rollout.rollout_manager import RolloutManager
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.rollout.server_group import ServerGroup
from tests.fast.ray.rollout.conftest import make_args


def _patch_engine_and_allocator_in_current_process() -> None:
    """Replace SGLangEngine with the mock + stub the addr allocator.

    Same shape as the ``patched_sglang_engine`` fixture but as a callable
    so it can run inside a Ray worker (where pytest fixtures don't reach)."""
    import miles.ray.rollout.server_group as sg
    from miles.utils.test_utils.mock_sglang_engine import MockSGLangEngine

    sg.SGLangEngine = MockSGLangEngine.__ray_actor_class__

    def _fake_alloc(*args, **kwargs):
        engines = kwargs["rollout_engines"]
        addr_and_ports = {}
        for rank, _ in engines:
            addr_and_ports[rank] = dict(
                host="127.0.0.1",
                port=30000 + rank,
                nccl_port=31000 + rank,
                engine_info_bootstrap_port=32000 + rank,
                dist_init_addr=f"127.0.0.1:{33000 + rank}",
            )
        return addr_and_ports, PortCursors(_values={0: 34000})

    sg.allocate_rollout_engine_addr_and_ports_normal = _fake_alloc


@ray.remote
class _RolloutManagerForTest(RolloutManager.__ray_actor_class__):
    """Real Ray-actor RolloutManager for routing tests. Bypasses the
    production heavy ``__init__`` (start_rollout_servers etc.) and patches
    SGLangEngine + allocator inside the worker process."""

    def __init__(self, servers: dict[str, RolloutServer]):
        _patch_engine_and_allocator_in_current_process()

        from miles.ray.utils import Lock
        self.servers = servers
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self._health_monitors = []
        self.rollout_id = 0

    def _peek_servers(self) -> dict[str, RolloutServer]:
        """Return the worker's view of ``self.servers`` (for test inspection
        — Ray copies state at __init__, mutations stay in the worker)."""
        return self.servers


def _build_server(*, pg_tuple: tuple, model_name: str, update_weights: bool,
                  num_engines: int = 2, num_groups: int = 1) -> RolloutServer:
    args = make_args(num_gpus_per_node=8)
    groups = []
    for _ in range(num_groups):
        engines = [ServerEngine() for _ in range(num_engines)]
        groups.append(ServerGroup(
            args=args, pg=pg_tuple, all_engines=engines,
            num_gpus_per_engine=1, has_new_engines=False,
            update_weights=update_weights,
        ))
    return RolloutServer(server_groups=groups, model_name=model_name,
                         update_weights=update_weights)


def _start_all(servers: dict[str, RolloutServer]) -> None:
    for srv in servers.values():
        for g in srv.server_groups:
            handles, indices = g.start_engines(PortCursors.empty())
            ray.get(handles)
            g.mark_alive(indices)


def _kill_all(servers: dict[str, RolloutServer]) -> None:
    for srv in servers.values():
        for g in srv.server_groups:
            for e in g.all_engines:
                if e.is_allocated:
                    try:
                        ray.kill(e.actor_handle)
                    except Exception:
                        pass


@pytest.mark.asyncio
class TestStartStopCell:
    async def test_start_cell_creates_actors_for_dead_engines(
        self, patched_sglang_engine, placement_group_factory,
    ):
        """Mark all engines stopped, then ``start_cell.remote(0)`` driven via
        real Ray actor dispatch must run a real ``recover()`` in the worker
        and create a mock SGLangEngine actor with ``init`` called on it."""
        pg = placement_group_factory(2)
        srv = _build_server(pg_tuple=pg, model_name="actor", update_weights=True,
                            num_engines=2)
        for e in srv.server_groups[0].all_engines:
            e.mark_stopped()
        servers = {"actor": srv}

        manager = _RolloutManagerForTest.remote(servers)
        try:
            ray.get(manager.start_cell.remote(0))

            # Ray copies state at __init__; the worker's mutations stay there.
            # Pull the worker's servers view to inspect post-recover state.
            worker_servers = ray.get(manager._peek_servers.remote())
            engine = worker_servers["actor"].server_groups[0].all_engines[0]
            assert engine.is_allocated
            calls = ray.get(engine.actor_handle.get_calls.remote())
            assert "init" in [c[0] for c in calls]
        finally:
            ray.get(manager._peek_servers.remote())  # ensure worker drained
            _kill_all(ray.get(manager._peek_servers.remote()))
            ray.kill(manager)

    async def test_stop_cell_kills_target_engine(
        self, patched_sglang_engine, placement_group_factory,
    ):
        """``stop_cell.remote(0)`` dispatches to worker, which calls
        ``stop_engines`` → ``ray.kill`` on the underlying actor handle.
        Actor handles are stable across processes, so the test holds the
        original handles and verifies follow-up ``.remote()`` raises."""
        pg = placement_group_factory(2)
        srv = _build_server(pg_tuple=pg, model_name="actor", update_weights=True,
                            num_engines=2)
        servers = {"actor": srv}
        _start_all(servers)
        original_actors = [e.actor_handle for e in srv.server_groups[0].all_engines]

        manager = _RolloutManagerForTest.remote(servers)
        try:
            ray.get(manager.stop_cell.remote(0))

            # Cell 0 actor must be dead at the Ray level.
            with pytest.raises((ray.exceptions.RayActorError, ray.exceptions.RayTaskError)):
                ray.get(original_actors[0].health_generate.remote(timeout=1.0), timeout=10.0)
            # Cell 1 actor untouched.
            assert ray.get(original_actors[1].health_generate.remote(timeout=1.0)) is True
        finally:
            _kill_all(servers)
            ray.kill(manager)


@pytest.mark.asyncio
class TestGetUpdatableEnginesAndLock:
    async def test_returns_populated_when_updatable_exists(
        self, patched_sglang_engine, placement_group_factory,
    ):
        """The updatable server's engines / counts / offsets / has_new_engines
        flow into EnginesAndLock through real Ray RPC. Engines come back as
        real Ray ActorHandles."""
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(2)
        ref = _build_server(pg_tuple=pg_a, model_name="ref", update_weights=False,
                            num_engines=2)
        actor = _build_server(pg_tuple=pg_b, model_name="actor", update_weights=True,
                              num_engines=2)
        actor.server_groups[0].has_new_engines = True
        servers = {"actor": actor, "ref": ref}
        _start_all(servers)

        manager = _RolloutManagerForTest.remote(servers)
        try:
            eal = ray.get(manager.get_updatable_engines_and_lock.remote())
            assert eal.has_new_engines is True
            assert len(eal.rollout_engines) == 2
            assert eal.engine_gpu_counts == [1, 1]
            # Handles came back through Ray serialization; still valid ActorHandles.
            assert all(isinstance(h, ray.actor.ActorHandle) for h in eal.rollout_engines)
            # And they actually point at live mock actors.
            assert ray.get(eal.rollout_engines[0].health_generate.remote(timeout=1.0)) is True
        finally:
            _kill_all(servers)
            ray.kill(manager)
