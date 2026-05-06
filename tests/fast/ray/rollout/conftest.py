"""Shared fixtures for tests/fast/ray/rollout/.

These fixtures back **all** rollout-refactor tests (P-critical / P0 / P1 / P2).
Per the test plan, foundational and must land first."""

from __future__ import annotations

import textwrap
from argparse import Namespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import ray

from miles.utils.types import Sample


def fake_actor_handle() -> MagicMock:
    """Build a MagicMock that passes isinstance(x, ray.actor.ActorHandle).

    The pydantic _StateAllocated* model validates ``actor_handle: ray.actor.ActorHandle``
    via ``is_instance_of``. MagicMock's ``__class__`` is a property that returns
    ``self._spec_class or type(self)``, so we set ``_spec_class`` directly. Using
    ``spec=`` would also achieve isinstance compatibility but locks attribute access
    to the spec class — and ActorHandle's methods are routed via ``__getattr__``,
    so they don't show up as class attributes. Direct ``_spec_class`` assignment
    keeps arbitrary attribute auto-creation (so ``actor.shutdown.remote(...)``
    chains still work)."""
    m = MagicMock()
    m._spec_class = ray.actor.ActorHandle
    return m


def make_args(**overrides: Any) -> Namespace:
    """Construct a minimal but usable args namespace covering all fields touched
    by `miles/ray/rollout/`. Each test overrides only what it cares about.

    The defaults below were derived by grep-ing every `args.<name>` in the new
    modules. Adding a new field here is fine; deleting one likely breaks tests."""
    defaults: dict[str, Any] = dict(
        # rollout core
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=8,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
        n_samples_per_eval_prompt=4,
        rollout_max_response_len=512,
        rollout_temperature=1.0,
        over_sampling_batch_size=None,
        rollout_global_dataset=False,
        num_rollout=1,
        # batch / training
        global_batch_size=8,
        use_dynamic_global_batch_size=False,
        disable_rollout_trim_samples=False,
        balance_data=False,
        # advantage / reward
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        reward_key=None,
        log_reward_category=None,
        log_passrate=False,
        # placement / colocation
        debug_train_only=False,
        debug_rollout_only=False,
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        critic_num_nodes=0,
        critic_num_gpus_per_node=0,
        use_critic=False,
        critic_train_only=False,
        # sglang router
        sglang_router_ip=None,
        sglang_router_port=None,
        sglang_router_request_timeout_secs=600,
        sglang_dp_size=1,
        sglang_speculative_algorithm=None,
        sglang_config=None,
        sglang_model_routers=None,
        prefill_num_servers=None,
        # routers / session server
        use_miles_router=False,
        use_session_server=False,
        session_server_ip=None,
        session_server_port=None,
        # external rollout
        rollout_external=False,
        rollout_external_engine_addrs=None,
        # offload / fault tolerance
        offload_rollout=False,
        use_fault_tolerance=False,
        rollout_health_check_interval=10.0,
        rollout_health_check_timeout=30.0,
        # checkpoint / data source
        hf_checkpoint="/fake/model",
        rollout_function_path="miles.rollout.sglang_rollout.generate_rollout",
        eval_function_path="miles.rollout.sglang_rollout.eval_generate_rollout",
        data_source_path="miles.data.dummy.DummyDataSource",
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        custom_rollout_log_function_path=None,
        custom_eval_rollout_log_function_path=None,
        # debug data
        save_debug_rollout_data=None,
        load_debug_rollout_data=None,
        load_debug_rollout_data_subsample=None,
        # CI
        ci_test=False,
        # dumper (sglang debug dumper integration)
        dumper_enable=False,
        dumper_inference=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def make_sample(
    *,
    group_index: int = 0,
    index: int = 0,
    response_length: int = 4,
    reward: float | dict | None = 1.0,
    status: Sample.Status = Sample.Status.COMPLETED,
    **overrides: Any,
) -> Sample:
    """Build a Sample with sensible defaults. Token list defaults to a length
    matching ``response_length`` so loss_mask/effective_response_length checks pass."""
    s = Sample(
        group_index=group_index,
        index=index,
        prompt="prompt",
        tokens=list(range(response_length)),
        response="response",
        response_length=response_length,
        label="label",
        reward=reward,
        status=status,
    )
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def make_samples_grouped(
    n_groups: int,
    group_size: int,
    *,
    rewards: list[float] | None = None,
    response_length: int = 4,
) -> list[Sample]:
    """Construct ``n_groups * group_size`` samples laid out group-by-group.

    If ``rewards`` is given, must have length n_groups*group_size."""
    total = n_groups * group_size
    if rewards is not None:
        assert len(rewards) == total, f"rewards must have length {total}, got {len(rewards)}"
    samples: list[Sample] = []
    for g in range(n_groups):
        for k in range(group_size):
            i = g * group_size + k
            r = rewards[i] if rewards is not None else float(k)
            samples.append(
                make_sample(
                    group_index=g,
                    index=i,
                    reward=r,
                    response_length=response_length,
                )
            )
    return samples


def make_sglang_config_yaml(
    *,
    name: str = "default",
    server_groups: list[dict] | None = None,
    update_weights: bool | None = None,
    model_path: str | None = None,
) -> str:
    """Render a small SglangConfig YAML for from_yaml() round-trip tests."""
    server_groups = server_groups or [{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
    lines = ["sglang:", f"  - name: {name}"]
    if model_path is not None:
        lines.append(f"    model_path: {model_path}")
    if update_weights is not None:
        lines.append(f"    update_weights: {str(update_weights).lower()}")
    lines.append("    server_groups:")
    for g in server_groups:
        lines.append(f"      - worker_type: {g['worker_type']}")
        lines.append(f"        num_gpus: {g['num_gpus']}")
        if "num_gpus_per_engine" in g:
            lines.append(f"        num_gpus_per_engine: {g['num_gpus_per_engine']}")
    return "\n".join(lines) + "\n"


# --------------------------- ray fixtures ---------------------------


@pytest.fixture(scope="session")
def ray_local_mode():
    """Session-scoped Ray init. Uses local_mode=False (real mini cluster) so
    that ``@ray.remote`` actors actually run as ray actors. Tests that only
    need pure-Python helpers should not depend on this fixture."""
    import ray

    if not ray.is_initialized():
        ray.init(
            num_cpus=4,
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
        )
    yield
    # do not shutdown here; let the process exit handle it (avoids breaking
    # other session-scoped suites that may share the cluster).


@pytest.fixture
def ray_actor_baseline(ray_local_mode):
    """Snapshot live ray actor count before / after a test. Asserts no leak.

    Kept minimal because in our test layout most actors are torn down by the
    test itself; this fixture only catches accidental leaks from helper code."""
    import ray

    def _count():
        try:
            return len([a for a in ray.util.list_named_actors() if a])
        except Exception:
            return 0

    before = _count()
    yield
    after = _count()
    assert after <= before, f"Ray actor leaked: before={before} after={after}"


@pytest.fixture
def no_subprocess_leak():
    """Catch leaked multiprocessing children (router / session_server).

    Per the test plan (P-infra.4): we deliberately do NOT track ray actor
    counts here because we use ``local_mode``-style minicluster where actors
    are process-internal and Python-GC-collected. The real leak risk is
    multiprocessing children (router / session_server / sgl_router subproc)
    which keep file descriptors open and bind ports across tests."""
    import multiprocessing

    before = set(p.pid for p in multiprocessing.active_children())
    yield
    after = set(p.pid for p in multiprocessing.active_children())
    leaked = after - before
    assert not leaked, f"Subprocess leaked: pids={leaked}"


@pytest.fixture(autouse=True)
def _autouse_subprocess_leak_check():
    """Apply ``no_subprocess_leak`` to every test in this dir.

    Implemented inline so we don't have to remember to add the fixture
    parameter on every test that starts a router / session_server."""
    import multiprocessing

    before = {p.pid for p in multiprocessing.active_children()}
    yield
    after = {p.pid for p in multiprocessing.active_children()}
    leaked = after - before
    if leaked:
        # Tear down leaked children to avoid cascading test failures.
        for p in multiprocessing.active_children():
            if p.pid in leaked:
                try:
                    p.terminate()
                    p.join(timeout=2)
                except Exception:
                    pass
        raise AssertionError(f"Subprocess leaked from previous test: pids={leaked}")


# --------------------------- pytest helpers ---------------------------


def dedent(s: str) -> str:
    """Convenience for inline YAML in tests."""
    return textwrap.dedent(s).lstrip("\n")
