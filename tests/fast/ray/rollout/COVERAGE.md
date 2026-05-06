# Hotspot branch coverage gate

Per the test plan (P-infra.3), we **do not** apply a uniform line-coverage
threshold to `miles/ray/rollout/`. A single 80% line gate would force
authors to write meaningless tests against dataclass properties, transient
docstring lines, and one-line forwarding methods.

Instead we measure **branch coverage** on five "hotspot" files where the
combinatorial logic actually lives, and gate on the **total floor** rather
than a per-file 90%:

| File | Why | Current branch cov |
|---|---|---|
| `miles/ray/rollout/addr_allocator.py` | Port arithmetic, multi-node, prefill / external paths — historical bug source (e1c9c8486) | 100% |
| `miles/ray/rollout/server_group.py` | start_engines / stop / recover branches with offload / model_path / update_weights toggles | 94% |
| `miles/ray/rollout/train_data_conversion.py` | GRPO / GSPO / PPO / RPP normalization branches; balance_data partition logic | 92% |
| `miles/ray/rollout/rollout_server.py` | start_rollout_servers config resolution, aggregation across groups, wait_all_engines_alive timeouts | **70% (TODO → 90%)** |
| `miles/ray/rollout/rollout_manager.py` | Cell dispatch, fault-tolerance recovery, weight-update orchestration | **27% (TODO → 90%)** |

**Total: 69%.** The gate is `fail_under = 67` in `.coveragerc` — locks the
current floor with a 2-point buffer; any new test that drops total coverage
below 67% fails the job.

The 90% bar still applies as the *target* for each individual file. The
two laggards (`rollout_server`, `rollout_manager`) need follow-up work:
their lifecycle methods (generate, eval, recover, weight-update flows)
are exercised in `lifecycle/` but mostly via `fake_actor_handle()`, which
short-circuits past the actual method bodies.

## Running locally

```bash
cd <repo>
pytest --cov-config=tests/fast/ray/rollout/.coveragerc \
       --cov=miles.ray.rollout.addr_allocator \
       --cov=miles.ray.rollout.server_group \
       --cov=miles.ray.rollout.train_data_conversion \
       --cov=miles.ray.rollout.rollout_manager \
       --cov=miles.ray.rollout.rollout_server \
       --cov-branch --cov-report=term-missing \
       tests/fast/ray/rollout/
```

Requires `pytest-cov` and `coverage` (in `requirements.txt`).

## CI integration

**TODO** — currently the gate runs only on demand. To wire it into CI:

1. Replace the `pytest tests/fast` invocation (in the `unit-test` job
   matrix entry where `test_file == "fast"`) with the cov command in the
   "Running locally" section above.
2. Re-run `python .github/workflows/generate_github_workflows.py` to
   regenerate `pr-test.yml`.

The `fail_under = 67` directive in `.coveragerc` will fail the job if
total branch coverage drops below the current floor.

When raising the bar (after pushing `rollout_manager` / `rollout_server`
up): edit `.coveragerc`'s `fail_under`, never remove a file from the
source list — the gate's value comes from being comprehensive.

## Other files in `miles/ray/rollout/`

No hard coverage gate. They are tested by P0 / P-critical / P1 / P2 test
chapters; the gate is a regression net for the high-density modules only.
