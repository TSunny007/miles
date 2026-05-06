"""P-critical.4 — multi-model: ``update_weights=False`` + ``model_path`` →
onload-from-disk path.

Risk pointers:
- server_group.py:204-208: ``if self.update_weights or self.model_path``
  controls whether resume_memory_occupation is enqueued during recover().
- server_group.py:242-250: ``onload_weights_from_disk`` only triggers when
  ``model_path`` is set.

This is the **multi-model (reference / reward)** core bug area: a frozen
model with model_path set should load weights from disk, never from
training updates."""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock

import pytest

from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.rollout.server_group import ServerGroup
from tests.fast.ray.rollout.conftest import fake_actor_handle, make_args


def _make_group(*, update_weights: bool, model_path: str | None, num_engines: int = 2,
                needs_offload: bool = True) -> ServerGroup:
    args = make_args(num_gpus_per_node=8)
    engines = [ServerEngine() for _ in range(num_engines)]
    for e in engines:
        actor = fake_actor_handle()
        e.mark_allocated_uninitialized(actor)
        e.mark_alive()
    group = ServerGroup(
        args=args,
        pg=None,
        all_engines=engines,
        num_gpus_per_engine=1,
        has_new_engines=False,
        worker_type="regular",
        needs_offload=needs_offload,
        update_weights=update_weights,
        model_path=model_path,
    )
    return group


class TestOnloadWeightsFromDisk:
    def test_disk_onload_only_when_both_offload_and_model_path(self):
        # Path 1: needs_offload=True + model_path set → calls update_weights_from_disk
        group = _make_group(update_weights=False, model_path="/some/ref/model")
        handles = group.onload_weights_from_disk()
        assert len(handles) == 2
        for engine in group.engines:
            engine.actor_handle.update_weights_from_disk.remote.assert_called_with("/some/ref/model")

    def test_disk_onload_skipped_when_model_path_missing(self):
        group = _make_group(update_weights=False, model_path=None)
        assert group.onload_weights_from_disk() == []

    def test_disk_onload_skipped_when_offload_disabled(self):
        group = _make_group(update_weights=False, model_path="/x", needs_offload=False)
        assert group.onload_weights_from_disk() == []

    def test_actor_only_called_for_allocated_engines(self):
        group = _make_group(update_weights=False, model_path="/x", num_engines=3)
        # Drop one engine to stopped
        group.all_engines[1].mark_stopped()
        handles = group.onload_weights_from_disk()
        assert len(handles) == 2  # only the 2 still allocated


class TestOnloadOffloadFlowForFrozenModel:
    """Verify the offload/onload tag pass-through is independent of update_weights flag."""

    def test_offload_calls_release_for_all_allocated(self):
        group = _make_group(update_weights=False, model_path="/x")
        handles = group.offload()
        assert len(handles) == 2
        for engine in group.engines:
            engine.actor_handle.release_memory_occupation.remote.assert_called_once()

    def test_onload_passes_tags_through(self):
        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
        group = _make_group(update_weights=False, model_path="/x")
        group.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE])
        for engine in group.engines:
            engine.actor_handle.resume_memory_occupation.remote.assert_called_with(tags=[GPU_MEMORY_TYPE_KV_CACHE])

    def test_offload_skipped_when_needs_offload_false(self):
        group = _make_group(update_weights=False, model_path="/x", needs_offload=False)
        assert group.offload() == []
        assert group.onload(tags=["weights"]) == []


@pytest.mark.asyncio
class TestRecoverResumePathChoosesByUpdateWeightsAndModelPath:
    """server_group.py:207 — ``if self.update_weights or self.model_path``
    decides whether resume_memory_occupation is queued after release.

    Both updatable groups (training target) and frozen-with-model_path
    groups (reference) must trigger resume; only frozen-without-model_path
    skips resume."""

    async def test_updatable_group_resumes(self):
        from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

        group = _make_group(update_weights=True, model_path=None)
        for e in group.all_engines:
            e.mark_stopped()

        with _stub_start(group):
            await group.recover(port_cursors=_empty_cursors())

        for e in group.all_engines:
            # Pin the exact tag — server_group.py:215 must request WEIGHTS.
            e.actor_handle.resume_memory_occupation.remote.assert_called_with(
                tags=[GPU_MEMORY_TYPE_WEIGHTS],
            )

    async def test_frozen_with_model_path_still_resumes(self):
        from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

        group = _make_group(update_weights=False, model_path="/ref/model")
        for e in group.all_engines:
            e.mark_stopped()
        with _stub_start(group):
            await group.recover(port_cursors=_empty_cursors())
        for e in group.all_engines:
            e.actor_handle.resume_memory_occupation.remote.assert_called_with(
                tags=[GPU_MEMORY_TYPE_WEIGHTS],
            )

    async def test_frozen_without_model_path_does_not_resume(self):
        group = _make_group(update_weights=False, model_path=None)
        for e in group.all_engines:
            e.mark_stopped()
        with _stub_start(group):
            await group.recover(port_cursors=_empty_cursors())
        # release happens (offload), but resume should NOT
        for e in group.all_engines:
            e.actor_handle.resume_memory_occupation.remote.assert_not_called()


# ----------------------------- helpers -----------------------------


def _empty_cursors():
    from miles.ray.rollout.addr_allocator import PortCursors
    return PortCursors.empty()


def _stub_start(group: ServerGroup):
    """Replace ServerGroup.start_engines with a no-op that marks engines
    allocated, so recover() reaches the offload/onload phase without ray."""
    import contextlib

    orig = group.start_engines

    def fake(port_cursors, start_indices=None):
        idxs = start_indices if start_indices is not None else list(range(len(group.all_engines)))
        for i in idxs:
            actor = fake_actor_handle()
            actor.release_memory_occupation.remote = MagicMock(return_value="release_ref")
            actor.resume_memory_occupation.remote = MagicMock(return_value="resume_ref")
            group.all_engines[i].mark_allocated_uninitialized(actor)
        return [], idxs

    @contextlib.contextmanager
    def cm():
        group.start_engines = fake
        try:
            yield
        finally:
            group.start_engines = orig

    # Also patch asyncio.gather to evaluate the MagicMock-returned object refs as no-ops
    return _patch_gather_with(cm())


def _patch_gather_with(stack_cm):
    """Compose a context manager that also patches asyncio.gather to ignore
    MagicMock object refs from the stub."""
    import asyncio
    import contextlib

    @contextlib.contextmanager
    def outer():
        original = asyncio.gather

        async def fake_gather(*coros_or_refs, **kw):
            # MagicMock returns are not coroutines; just no-op
            results = []
            for x in coros_or_refs:
                if asyncio.iscoroutine(x):
                    results.append(await x)
                else:
                    results.append(None)
            return results

        asyncio.gather = fake_gather
        try:
            with stack_cm:
                yield
        finally:
            asyncio.gather = original

    return outer()
