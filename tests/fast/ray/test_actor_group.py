"""``actor_group.update_weights`` ordering under fault tolerance.

Production code (miles/ray/actor_group.py:122-132):

    async def update_weights(self):
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return
        if self.args.use_fault_tolerance:
            await self.rollout_manager.recover_updatable_engines.remote()  # (1)
        info = await self.rollout_manager.get_updatable_engines_and_lock.remote()  # (2)
        await self._broadcast("update_weights", info=info)  # (3)

If (1) raises, (2) and (3) MUST NOT run — otherwise actors would receive a
stale or inconsistent EnginesAndLock for engines that were not yet recovered.

Tests drive the REAL ``RayTrainGroup.update_weights`` against a fake ``self``
so a regression in actor_group.py is caught here, not in a hand-copied mirror.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.ray.actor_group import RayTrainGroup


def _make_remote_call(callable_or_value, *, is_coro=False):
    """Build a ``foo.remote()`` shaped mock.

    Production calls ``self.rollout_manager.recover_updatable_engines.remote()``
    and awaits the result. ``rollout_manager`` is a ray actor handle, so each
    method has a ``.remote`` attribute that returns an awaitable ObjectRef.
    """
    obj = MagicMock()
    if is_coro:
        obj.remote = AsyncMock(side_effect=callable_or_value)
    else:
        obj.remote = AsyncMock(return_value=callable_or_value)
    return obj


def _make_fake_self(*, use_fault_tolerance=True, debug_train_only=False, debug_rollout_only=False):
    """Build a fake ``self`` that supplies the attributes ``RayTrainGroup.update_weights``
    actually touches: ``args``, ``rollout_manager``, and ``_broadcast``."""
    self_ = SimpleNamespace()
    self_.args = SimpleNamespace(
        debug_train_only=debug_train_only,
        debug_rollout_only=debug_rollout_only,
        use_fault_tolerance=use_fault_tolerance,
    )
    self_.rollout_manager = SimpleNamespace(
        recover_updatable_engines=_make_remote_call(None),
        get_updatable_engines_and_lock=_make_remote_call(SimpleNamespace(name="EnginesAndLock")),
    )
    self_._broadcast = AsyncMock(return_value=None)
    return self_


class TestFaultToleranceFailureContainment:
    """Tripwires: if a step in ``update_weights`` raises, every later step
    MUST NOT execute — otherwise actors receive a stale EnginesAndLock from
    a partial pipeline. Catches regressions where a future ``try/except``
    is added to "tolerate" failures and silently leaks downstream."""

    @pytest.mark.asyncio
    async def test_recover_failure_aborts_get_info_and_broadcast(self):
        s = _make_fake_self()

        async def _recover_fail():
            raise RuntimeError("recover failed")

        s.rollout_manager.recover_updatable_engines.remote = AsyncMock(side_effect=_recover_fail)

        with pytest.raises(RuntimeError, match="recover failed"):
            await RayTrainGroup.update_weights(s)

        s.rollout_manager.get_updatable_engines_and_lock.remote.assert_not_awaited()
        s._broadcast.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_info_failure_skips_broadcast(self):
        """Mid-pipeline: recover ran clean, but get_info raised. broadcast
        must not run with stale (or missing) info."""
        s = _make_fake_self()
        executed: list[str] = []

        async def _recover():
            executed.append("recover")

        async def _get_info():
            executed.append("get_info")
            raise RuntimeError("get_info exploded mid-pipeline")

        async def _broadcast(*args, **kwargs):
            executed.append("broadcast")

        s.rollout_manager.recover_updatable_engines.remote = AsyncMock(side_effect=_recover)
        s.rollout_manager.get_updatable_engines_and_lock.remote = AsyncMock(side_effect=_get_info)
        s._broadcast = AsyncMock(side_effect=_broadcast)

        with pytest.raises(RuntimeError, match="get_info exploded"):
            await RayTrainGroup.update_weights(s)

        assert executed == ["recover", "get_info"]
        s._broadcast.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_debug_modes_short_circuit_completely(self):
        """``debug_train_only`` early-returns before any rollout-side call.
        Guards against someone "optimizing away" the early return."""
        s = _make_fake_self(debug_train_only=True)
        await RayTrainGroup.update_weights(s)
        s.rollout_manager.recover_updatable_engines.remote.assert_not_awaited()
        s.rollout_manager.get_updatable_engines_and_lock.remote.assert_not_awaited()
        s._broadcast.assert_not_awaited()
