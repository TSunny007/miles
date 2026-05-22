# Motivation of this CI is to prove post cleanup is required for ray job.

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci

import miles.utils.external_utils.command_utils as U

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


# A unit-test stand-in for "the host": records every command that
# `execute_train` issues, and additionally simulates the *effect* of the
# ray-related ones on a `ray_processes` set — `ray start --head` flips it
# ON, `ray stop` / `pkill -9 ray` flips it OFF. `has_ray_processes` is
# therefore "would ray still be alive on a real host if we had really
# run these commands?", which is what end-state assertions hinge on.
class _FakeRayHost:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.ray_processes: set[str] = set()

    def exec_command(self, cmd: str, capture_output: bool = False) -> str | None:
        self.commands.append(cmd)
        if "ray stop" in cmd or "pkill -9 ray" in cmd:
            self.ray_processes.clear()
        if "ray start --head" in cmd:
            self.ray_processes.add("local-ray-head")
        if capture_output:
            return "0"
        return None

    @property
    def has_ray_processes(self) -> bool:
        return bool(self.ray_processes)


# Simpler mock used by tests that only need to inspect the *command
# sequence* (not host end-state). `fail_on_submit` injects an exception
# on `ray job submit` so the `try/finally` path can be exercised.
def _patch_exec_command(monkeypatch, *, fail_on_submit: bool = False) -> list[str]:
    commands: list[str] = []

    def fake_exec_command(cmd: str, capture_output: bool = False) -> str | None:
        commands.append(cmd)
        if fail_on_submit and "ray job submit" in cmd:
            raise subprocess.CalledProcessError(1, cmd)
        if capture_output:
            return "0"
        return None

    monkeypatch.setattr(U, "exec_command", fake_exec_command)
    return commands


# `_cleanup_train_processes` builds ONE long shell string starting with
# `pkill -9 sglang` and passes it to `exec_command`. So in the recorded
# command list, each occurrence of that substring marks exactly one
# invocation of the cleanup helper, and its index tells us *when* it ran
# relative to `ray job submit`.
def _cleanup_commands(commands: list[str]) -> list[str]:
    return [cmd for cmd in commands if "pkill -9 sglang" in cmd]


def _cleanup_command_indices(commands: list[str]) -> list[int]:
    return [i for i, cmd in enumerate(commands) if "pkill -9 sglang" in cmd]


def _ray_submit_index(commands: list[str]) -> int:
    return next(i for i, cmd in enumerate(commands) if "ray job submit" in cmd)


def _run_ray_command(args: list[str]) -> None:
    subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)


# Core observation primitive for the real-ray test: "which ray processes
# are alive *right now* and belong to *this* test (not to anyone else on
# the host)?". Isolation trick: when ray is started with `--temp-dir=X`,
# every daemon and worker it spawns inherits `X` somewhere in its argv,
# so a `ps -o args=` substring filter on `X` carves out exactly this
# test's ray and ignores any unrelated ray cluster on the same host.
# Zombies are skipped because `pkill` cannot reap them and they consume
# no resources — counting them would spuriously claim "ray still alive"
# after a clean cleanup.
def _live_ray_processes_for_temp_dir(temp_dir) -> list[str]:
    result = subprocess.run(
        ["ps", "-eww", "-o", "pid=,stat=,args="],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    processes = []
    for line in result.stdout.splitlines():
        columns = line.strip().split(None, 2)
        if len(columns) != 3:
            continue
        _, stat, args = columns
        if stat.startswith("Z"):
            continue
        if str(temp_dir) in args:
            processes.append(line)
    return processes


# `ray start` / `ray job submit` return before all spawned daemons are
# fully up, and `pkill` similarly takes a moment to propagate. Polling
# is the only way to avoid race-flake. Deadlines are chosen long enough
# (20 s up, 30 s down) to absorb slow CI.
def _wait_for_live_ray_processes(temp_dir) -> list[str]:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        processes = _live_ray_processes_for_temp_dir(temp_dir)
        if processes:
            return processes
        time.sleep(0.5)
    return []


def _wait_for_no_live_ray_processes(temp_dir) -> bool:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not _live_ray_processes_for_temp_dir(temp_dir):
            return True
        time.sleep(0.5)
    return False


# `--num-gpus=0` so this test can run on CPU-only hosts. `--port=0` lets
# the kernel pick a free GCS port so parallel test runs don't collide.
# `--temp-dir` is the isolation key relied upon by
# `_live_ray_processes_for_temp_dir`.
def _start_real_ray(temp_dir) -> None:
    _run_ray_command(
        [
            "ray",
            "start",
            "--head",
            "--node-ip-address=127.0.0.1",
            "--num-cpus=1",
            "--num-gpus=0",
            "--object-store-memory=80000000",
            "--port=0",
            "--dashboard-host=127.0.0.1",
            "--dashboard-port=8265",
            f"--temp-dir={temp_dir}",
            "--disable-usage-stats",
            "--log-style=record",
            "--log-color=false",
        ]
    )


# The job body is irrelevant; the point is that *submitting* anything
# causes ray to spawn the extra worker / runtime-env-agent processes
# whose persistence past submit-return is the exact phenomenon the
# commit fixes. Without a submit, only the head daemons would exist.
def _submit_real_noop_ray_job() -> None:
    _run_ray_command(
        [
            "ray",
            "job",
            "submit",
            "--address=http://127.0.0.1:8265",
            "--",
            sys.executable,
            "-c",
            "print('noop ray job finished')",
        ]
    )


# === Motivation: empirically establish, with a real ray cluster, the
# foundational claim behind the commit — "ray daemons survive past
# `ray job submit`'s return, and only an explicit cleanup reaps them."
# Without this evidence, the whole post-kill argument is hand-waving.
#
# Why gated behind an env var: starts a real ray cluster (heavy) and
# the global pkill inside cleanup can collide with other ray clusters
# on a shared host.
#
# Why this proves the claim: between the two `_wait_for_*` snapshots
# below, *nothing happens except the cleanup call*, so any difference
# in the live-process set is attributable to the cleanup alone.
@pytest.mark.skipif(
    os.environ.get("MILES_TEST_REAL_RAY_CLEANUP") != "1" or shutil.which("ray") is None,
    reason="set MILES_TEST_REAL_RAY_CLEANUP=1 to run the real Ray lifecycle test",
)
def test_real_ray_processes_remain_after_job_submit_until_post_cleanup():
    base_temp_dir = Path(tempfile.mkdtemp(prefix="ray-real-", dir="/tmp"))
    ray_temp_dir = base_temp_dir / "ray"

    try:
        # Wipe any residue from previous failed runs so the temp-dir
        # filter below starts from a known-clean baseline.
        U._cleanup_train_processes(external_ray=False, force_ray_stop=True)

        # (1) Bring up ray and (2) confirm it really came up.
        _start_real_ray(ray_temp_dir)
        assert _wait_for_live_ray_processes(ray_temp_dir)

        # (3) Submit a job; submit returns once the job finishes.
        # (4) Despite submit having returned, ray daemons are still
        # alive — this is the "pre-kill alone is insufficient" fact.
        _submit_real_noop_ray_job()
        assert _wait_for_live_ray_processes(ray_temp_dir)

        # (5) Run the cleanup helper. (6) Now the daemons are gone.
        # (4)→(6) with only step (5) in between = cleanup did the work.
        U._cleanup_train_processes(external_ray=False, force_ray_stop=True)
        assert _wait_for_no_live_ray_processes(ray_temp_dir)
    finally:
        # Defensive: if any assert above raised, don't leave a real ray
        # cluster running on the host.
        U._cleanup_train_processes(external_ray=False, force_ray_stop=True)
        shutil.rmtree(base_temp_dir, ignore_errors=True)


# === Motivation: same claim as the real-ray test above, expressed as a
# deterministic unit test so it runs in every CI without the env-var
# gate. The trick is that `_FakeRayHost` is a state machine whose
# `has_ray_processes` mirrors "would ray be alive on a real host?" — so
# end-state assertions on it stand in for `ps`-snapshot assertions.
#
# Proof mechanism: two arms on two fresh fake hosts.
#   Arm A replays the *pre-commit* control flow by hand (pre-cleanup,
#     start, submit — no post-cleanup) and observes the fake host ends
#     dirty (`has_ray_processes` is True).
#   Arm B calls the real `execute_train`, which now wraps submit in
#     try/finally with a post-cleanup, and observes the fake host ends
#     clean.
# Arm A dirty + Arm B clean ⇒ the post-cleanup is what closes the leak.
def test_execute_train_post_cleanup_prevents_local_ray_process_leak(monkeypatch):
    legacy_host = _FakeRayHost()
    monkeypatch.setattr(U, "exec_command", legacy_host.exec_command)

    U._cleanup_train_processes(external_ray=False, force_ray_stop=True)
    U.exec_command(
        "export PYTHONBUFFERED=16 && "
        "ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats"
    )
    U.exec_command('ray job submit --address="http://127.0.0.1:8265" -- python3 /tmp/train.py --train-backend fsdp')

    assert legacy_host.has_ray_processes
    assert any("ray start --head" in cmd for cmd in legacy_host.commands)
    assert any("ray job submit" in cmd for cmd in legacy_host.commands)

    fixed_host = _FakeRayHost()
    monkeypatch.setattr(U, "exec_command", fixed_host.exec_command)
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY", raising=False)
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")

    U.execute_train(
        train_args="--train-backend fsdp",
        num_gpus_per_node=1,
        megatron_model_type=None,
        train_script="/tmp/train.py",
    )

    assert not fixed_host.has_ray_processes
    assert any("ray start --head" in cmd for cmd in fixed_host.commands)
    assert any("ray job submit" in cmd for cmd in fixed_host.commands)


# === Motivation: pin down the *structural* invariant introduced by the
# commit — on a successful run, `execute_train` issues exactly two
# cleanups, one before submit and one after, both targeting the local
# ray cluster (`ray stop --force`).
#
# Proof mechanism: every cleanup invocation leaves a `pkill -9 sglang`
# marker in the recorded command list; counting markers gives the
# invocation count, and their indices vs `ray job submit`'s index give
# the relative ordering.
def test_execute_train_cleans_up_owned_local_ray_after_submit_succeeds(monkeypatch):
    commands = _patch_exec_command(monkeypatch)
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY", raising=False)
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")

    U.execute_train(
        train_args="--train-backend fsdp",
        num_gpus_per_node=1,
        megatron_model_type=None,
        train_script="/tmp/train.py",
    )

    cleanup_indices = _cleanup_command_indices(commands)
    submit_index = _ray_submit_index(commands)
    assert len(cleanup_indices) == 2
    assert cleanup_indices[0] < submit_index < cleanup_indices[1]
    assert all("ray stop --force" in commands[i] for i in cleanup_indices)


# === Motivation: the most important guarantee of the commit — when
# `ray job submit` *raises*, the post-cleanup must still run, otherwise
# the freshly started ray cluster would leak. This pins down that the
# post-cleanup lives in a `finally`, not in a post-call statement.
#
# Proof mechanism: a regular "cleanup after submit" line would be
# *skipped* by the propagating exception, so `cleanup_indices[1]` would
# not exist. Observing `cleanup_indices[1] > submit_index` despite the
# raised exception forces the conclusion that the call sits in a
# `finally` block.
def test_execute_train_cleans_up_owned_local_ray_when_submit_fails(monkeypatch):
    commands = _patch_exec_command(monkeypatch, fail_on_submit=True)
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY", raising=False)
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")

    with pytest.raises(subprocess.CalledProcessError):
        U.execute_train(
            train_args="--train-backend fsdp",
            num_gpus_per_node=1,
            megatron_model_type=None,
            train_script="/tmp/train.py",
        )

    cleanup_indices = _cleanup_command_indices(commands)
    submit_index = _ray_submit_index(commands)
    assert len(cleanup_indices) == 2
    assert cleanup_indices[0] < submit_index < cleanup_indices[1]
    assert all("ray stop --force" in commands[i] for i in cleanup_indices)


# === Motivation: enforce the external-ray contract — when the user owns
# the ray cluster (`MILES_SCRIPT_EXTERNAL_RAY=1`), `execute_train` must
# not touch it. Violating this would silently kill the user's
# long-running cluster.
#
# Proof mechanism: three orthogonal checks on the recorded command list:
#   (1) post-cleanup is gated by `not external_ray`, so only the
#       pre-cleanup should appear → exactly one cleanup marker.
#   (2) `_cleanup_train_processes(external_ray=True, ...)` strips the
#       `ray stop` and `pkill -9 ray` subcommands from the cleanup
#       string → those substrings must be absent.
#   (3) `ray start --head` is gated by `not external_ray` → must not be
#       issued at all.
def test_execute_train_does_not_stop_external_ray(monkeypatch):
    commands = _patch_exec_command(monkeypatch)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "0")

    U.execute_train(
        train_args="--train-backend fsdp",
        num_gpus_per_node=1,
        megatron_model_type=None,
        train_script="/tmp/train.py",
    )

    cleanup_commands = _cleanup_commands(commands)
    assert len(cleanup_commands) == 1
    assert "ray stop" not in cleanup_commands[0]
    assert "pkill -9 ray" not in cleanup_commands[0]
    assert not any("ray start --head" in cmd for cmd in commands)


# === Motivation: document the intentional-leak design — when submit is
# disabled (`MILES_SCRIPT_ENABLE_RAY_SUBMIT=0`), `execute_train` still
# starts ray locally so the caller can drive it manually, and the
# post-cleanup is *intentionally* skipped so the cluster survives the
# call. Without this test someone "fixing" the perceived leak would
# break the workflow.
#
# Proof mechanism: only the pre-cleanup fires (one marker, contains
# `ray stop --force` because ray is locally owned); `ray start --head`
# is issued; `ray job submit` is not; and the fake host's state machine
# ends with `has_ray_processes` True — i.e., the cluster outlives the
# call by design.
def test_execute_train_leaves_local_ray_running_when_submit_is_disabled(monkeypatch):
    host = _FakeRayHost()
    monkeypatch.setattr(U, "exec_command", host.exec_command)
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY", raising=False)
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "0")

    U.execute_train(
        train_args="--train-backend fsdp",
        num_gpus_per_node=1,
        megatron_model_type=None,
        train_script="/tmp/train.py",
    )

    cleanup_commands = _cleanup_commands(host.commands)
    assert len(cleanup_commands) == 1
    assert "ray stop --force" in cleanup_commands[0]
    assert any("ray start --head" in cmd for cmd in host.commands)
    assert not any("ray job submit" in cmd for cmd in host.commands)
    assert host.has_ray_processes
