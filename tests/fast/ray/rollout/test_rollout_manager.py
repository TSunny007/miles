"""Pure-Python tests for ``miles/ray/rollout/rollout_manager.py``.

Covers routing helpers (get_updatable_engines_and_lock no-actor paths,
clear_updatable_has_new_engines flag flip) without scheduling actors.

Tests that need real Ray actors live in ``real_ray/test_rollout_manager.py``."""

from __future__ import annotations

import asyncio

import pytest

from tests.fast.ray.rollout.conftest import make_dataclass_group


def _build_dataclass_server(*, model_name: str, update_weights: bool,
                            num_engines: int = 2, num_groups: int = 1):
    """Build a RolloutServer + ServerGroup chain with ``pg=None`` for tests
    that exercise rollout_manager routing without scheduling any actors."""
    from miles.ray.rollout.rollout_server import RolloutServer
    groups = [make_dataclass_group(num_engines=num_engines) for _ in range(num_groups)]
    for g in groups:
        g.update_weights = update_weights
    return RolloutServer(server_groups=groups, model_name=model_name,
                         update_weights=update_weights)


def _build_fake_rollout_manager(servers: dict):
    """Construct a RolloutManager bypassing ``__init__`` (which would call
    ``start_rollout_servers``). Sets just the attrs the routing helpers touch."""
    from miles.ray.rollout.rollout_manager import RolloutManager
    cls = RolloutManager.__ray_actor_class__
    fake = cls.__new__(cls)
    fake.servers = servers
    fake.rollout_engine_lock = asyncio.Lock()
    fake._health_monitors = []
    fake.rollout_id = 0
    return fake


class TestRolloutManagerGetUpdatableEmptyPaths:
    """get_updatable_engines_and_lock — no-actor paths (no updatable server /
    multiple updatables)."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_updatable_server(self):
        from miles.ray.rollout.rollout_manager import EnginesAndLock, RolloutManager

        srv = _build_dataclass_server(model_name="ref", update_weights=False, num_engines=2)
        fake = _build_fake_rollout_manager({"ref": srv})

        eal = await RolloutManager.__ray_actor_class__.get_updatable_engines_and_lock(fake)
        assert isinstance(eal, EnginesAndLock)
        assert eal.rollout_engines == []
        assert eal.has_new_engines is False
        assert eal.engine_gpu_counts == []
        assert eal.engine_gpu_offsets == []
        assert eal.rollout_engine_lock is fake.rollout_engine_lock

    @pytest.mark.asyncio
    async def test_assert_multiple_updatable_servers(self):
        from miles.ray.rollout.rollout_manager import RolloutManager
        servers = {
            "actor1": _build_dataclass_server(model_name="actor1", update_weights=True, num_engines=1),
            "actor2": _build_dataclass_server(model_name="actor2", update_weights=True, num_engines=1),
        }
        fake = _build_fake_rollout_manager(servers)

        with pytest.raises(AssertionError, match="Multiple servers"):
            await RolloutManager.__ray_actor_class__.get_updatable_engines_and_lock(fake)


class TestRolloutManagerClearHasNewEnginesFlag:
    """Pure flag flip — no actors, no async."""

    def test_clears_flag_on_updatable_server(self):
        from miles.ray.rollout.rollout_manager import RolloutManager
        srv = _build_dataclass_server(model_name="actor", update_weights=True,
                                       num_engines=2, num_groups=2)
        for g in srv.server_groups:
            g.has_new_engines = True
        fake = _build_fake_rollout_manager({"actor": srv})

        RolloutManager.__ray_actor_class__.clear_updatable_has_new_engines(fake)

        for g in srv.server_groups:
            assert g.has_new_engines is False

    def test_no_op_when_no_updatable_server(self):
        from miles.ray.rollout.rollout_manager import RolloutManager
        srv = _build_dataclass_server(model_name="ref", update_weights=False, num_engines=2)
        srv.server_groups[0].has_new_engines = True
        fake = _build_fake_rollout_manager({"ref": srv})

        RolloutManager.__ray_actor_class__.clear_updatable_has_new_engines(fake)
        # ref's flag must NOT be cleared (it's not updatable)
        assert srv.server_groups[0].has_new_engines is True
