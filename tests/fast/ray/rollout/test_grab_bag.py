"""Grab-bag of small edge-case tests across miles/ray/rollout/.

Covers single-case edges in rollout_data_conversion, metrics, server_cell,
router_manager, and rollout_server pure helpers — each a one-shot that
doesn't warrant its own file. Kept short: single test per concern."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.fast.ray.rollout.conftest import fake_actor_handle, make_args, make_sample, make_samples_grouped


# --------------------------- rollout_data_conversion ---------------------------


class TestRolloutDataConversionEdges:
    def test_dynamic_global_batch_size_below_dp_size_falls_back(self):
        from miles.ray.rollout.rollout_data_conversion import _compute_dynamic_global_batch_size

        args = make_args(global_batch_size=64)
        gbs = _compute_dynamic_global_batch_size(args, train_parallel_config={"dp_size": 4}, num_samples=2)
        # When num_samples < dp_size we fallback to dp_size
        assert gbs == 4

    def test_postprocess_rollout_data_raises_when_trim_is_zero(self):
        from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data

        args = make_args(global_batch_size=64, disable_rollout_trim_samples=False, use_dynamic_global_batch_size=False)
        with pytest.raises(ValueError, match="Not enough samples"):
            postprocess_rollout_data(args, [make_sample()] * 5, train_parallel_config={"dp_size": 1})

    def test_postprocess_rollout_data_flattens_nested_lists(self):
        from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data

        args = make_args(global_batch_size=2, disable_rollout_trim_samples=True, use_dynamic_global_batch_size=False)
        nested = [[make_sample(index=0), make_sample(index=1)], [make_sample(index=2)]]
        data, _meta = postprocess_rollout_data(args, nested, train_parallel_config={"dp_size": 1})
        assert len(data) == 3


# --------------------------- metrics ---------------------------


class TestMetricsEdges:
    def test_compute_zero_std_metrics_short_circuits_for_ppo(self):
        from miles.ray.rollout.metrics import _compute_zero_std_metrics

        args = make_args(advantage_estimator="ppo")
        out = _compute_zero_std_metrics(args, make_samples_grouped(2, 4, rewards=[1.0] * 8))
        assert out == {}

    def test_compute_zero_std_metrics_buckets_grpo_zero_std_groups(self):
        from miles.ray.rollout.metrics import _compute_zero_std_metrics

        args = make_args(advantage_estimator="grpo", reward_key=None)
        # group 0: all rewards 1.0 (zero std) → bucket "1.0"
        # group 1: all rewards 0.0 (zero std) → bucket "0.0"
        # group 2: mixed → not zero std
        samples = make_samples_grouped(3, 4, rewards=[1.0] * 4 + [0.0] * 4 + [0.0, 1.0, 0.0, 1.0])
        out = _compute_zero_std_metrics(args, samples)
        assert out.get("zero_std/count_1.0") == 1
        assert out.get("zero_std/count_0.0") == 1

    def test_tito_strict_mismatch_raises_under_ci_test(self):
        """Under ci_test=True, a non-zero rate on the strict tito mismatch
        types (special_token_count / special_token_type / non_assistant_text)
        must hard-fail — these signal a TITO algorithm bug, not a benign
        tokenization difference."""
        from miles.ray.rollout.metrics import _compute_metrics_from_samples

        args = make_args(advantage_estimator="ppo", ci_test=True, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        # Plant a special_token_count mismatch on one sample → rate > 0.
        samples[0].metadata = {
            "tito_session_mismatch": [{"type": "special_token_count"}],
        }
        for s in samples[1:]:
            s.metadata = {"tito_session_mismatch": []}
        with pytest.raises(AssertionError, match="special_token_count"):
            _compute_metrics_from_samples(args, samples)

    def test_tito_assistant_text_mismatch_does_not_raise_under_ci_test(self):
        """assistant_text mismatch is non-critical (tokens inherited from
        pretokenized prefix) — even under ci_test, it must NOT raise."""
        from miles.ray.rollout.metrics import _compute_metrics_from_samples

        args = make_args(advantage_estimator="ppo", ci_test=True, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        samples[0].metadata = {
            "tito_session_mismatch": [{"type": "assistant_text"}],
        }
        for s in samples[1:]:
            s.metadata = {"tito_session_mismatch": []}
        out = _compute_metrics_from_samples(args, samples)
        assert out["tito_session_mismatch_rate/assistant_text"] > 0


# --------------------------- server_cell ---------------------------


def _build_servers(*, num_servers: int = 1, groups_per_server: int = 1,
                   engines_per_group: int = 2, num_gpus_per_engine: int = 1) -> dict:
    """Build a ``{key: RolloutServer}`` dict with real ServerGroup / ServerEngine
    plus fake_actor_handle, suitable for driving ``get_cell_indexer_of_id_map``."""
    from miles.ray.rollout.rollout_server import RolloutServer
    from miles.ray.rollout.server_engine import ServerEngine
    from miles.ray.rollout.server_group import ServerGroup
    args = make_args(num_gpus_per_node=8)
    servers: dict = {}
    for s_idx in range(num_servers):
        groups = []
        for _g in range(groups_per_server):
            engines = [ServerEngine() for _ in range(engines_per_group)]
            for e in engines:
                e.mark_allocated_uninitialized(fake_actor_handle())
                e.mark_alive()
            groups.append(ServerGroup(
                args=args, pg=None, all_engines=engines,
                num_gpus_per_engine=num_gpus_per_engine, has_new_engines=False,
                update_weights=True,
            ))
        servers[f"model_{s_idx}"] = RolloutServer(
            server_groups=groups, model_name=f"model_{s_idx}", update_weights=True,
        )
    return servers


class TestServerCellAssertEdges:
    def test_get_cell_indexer_handles_placeholder_group(self):
        """``placeholder`` groups have empty all_engines but the assertion
        ``len(all_engines) == len(engines) * nodes_per_engine`` should still
        hold (0 == 0 * N)."""
        from miles.ray.rollout.server_cell import get_cell_indexer_of_id_map

        srv = MagicMock()
        group = MagicMock()
        group.all_engines = []
        group.engines = []
        group.nodes_per_engine = 2
        srv.server_groups = [group]
        out = get_cell_indexer_of_id_map({"only": srv})
        assert out == []

    def test_dispatches_single_server(self):
        from miles.ray.rollout.server_cell import get_cell_indexer_of_id_map
        servers = _build_servers(num_servers=1, groups_per_server=1, engines_per_group=3)
        cells = get_cell_indexer_of_id_map(servers)
        assert len(cells) == 3
        for cell in cells:
            assert cell.srv_key == "model_0"
            assert cell.group_index == 0
            assert len(cell.engine_indices) == 1  # nodes_per_engine=1

    def test_dispatches_multi_group_continuous_ids(self):
        from miles.ray.rollout.server_cell import get_cell_indexer_of_id_map
        servers = _build_servers(num_servers=1, groups_per_server=2, engines_per_group=2)
        cells = get_cell_indexer_of_id_map(servers)
        assert len(cells) == 4
        # cells 0,1 → group 0; cells 2,3 → group 1
        assert cells[0].group_index == 0 and cells[1].group_index == 0
        assert cells[2].group_index == 1 and cells[3].group_index == 1

    def test_orders_servers_by_key(self):
        from miles.ray.rollout.server_cell import get_cell_indexer_of_id_map
        servers = _build_servers(num_servers=2, groups_per_server=1, engines_per_group=1)
        cells = get_cell_indexer_of_id_map(servers)
        srv_keys_in_order = [c.srv_key for c in cells]
        assert srv_keys_in_order == sorted(srv_keys_in_order)

    def test_engine_indices_span_nodes_per_engine(self):
        """When num_gpus_per_engine=16 (>num_gpus_per_node=8), each cell maps to
        2 contiguous engine slots."""
        from miles.ray.rollout.server_cell import get_cell_indexer_of_id_map
        servers = _build_servers(num_servers=1, groups_per_server=1,
                                 engines_per_group=2, num_gpus_per_engine=16)
        cells = get_cell_indexer_of_id_map(servers)
        assert len(cells) == 1  # 2 engine slots → 1 multi-node cell
        assert cells[0].engine_indices == [0, 1]


# --------------------------- router_manager ---------------------------


class TestRouterManagerErrorPaths:
    def test_pd_disagg_with_miles_router_asserts(self):
        from miles.ray.rollout.router_manager import start_router

        args = make_args(use_miles_router=True, sglang_router_ip=None, sglang_router_port=None)
        with patch("miles.ray.rollout.router_manager.get_host_info", return_value=("h", "127.0.0.1")), \
             patch("miles.ray.rollout.router_manager.find_available_port", return_value=20000):
            with pytest.raises(AssertionError, match="miles router does not support PD"):
                start_router(args, has_pd_disaggregation=True, force_new=False)

    def test_port_conflict_raises_runtime_error(self):
        from miles.ray.rollout.router_manager import start_router

        args = make_args(use_miles_router=False, sglang_router_ip=None, sglang_router_port=None)
        with patch("miles.ray.rollout.router_manager.get_host_info", return_value=("h", "127.0.0.1")), \
             patch("miles.ray.rollout.router_manager.find_available_port", return_value=20000), \
             patch("miles.ray.rollout.router_manager.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="already in use"):
                start_router(args)

    def test_session_server_disabled_returns_silently(self):
        from miles.ray.rollout.router_manager import start_session_server

        args = make_args(use_session_server=False)
        # Should not raise even with no other config.
        start_session_server(args)

    def test_session_server_without_hf_checkpoint_raises(self):
        from miles.ray.rollout.router_manager import start_session_server

        args = make_args(use_session_server=True, hf_checkpoint=None)
        with pytest.raises(ValueError, match="hf-checkpoint"):
            start_session_server(args)

    def test_session_server_port_conflict_raises_runtime_error(self):
        """Symmetric to start_router's port-conflict path: when the configured
        session_server port is already bound, fail loud with a RuntimeError
        rather than silently re-using the stale process."""
        from miles.ray.rollout.router_manager import start_session_server

        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            sglang_router_ip="127.0.0.1",
            sglang_router_port=20000,
            session_server_ip="127.0.0.1",
            session_server_port=20001,
        )
        with patch("miles.ray.rollout.router_manager.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="already in use"):
                start_session_server(args)


# --------------------------- rollout_server pure functions ---------------------------


class TestRolloutServerPureFunctions:
    def test_resolve_sglang_config_yaml_gpu_mismatch_asserts(self, tmp_path):
        from miles.ray.rollout.rollout_server import _resolve_sglang_config

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 1\n"
        )
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=8)
        with pytest.raises(AssertionError, match="total GPUs"):
            _resolve_sglang_config(args)

    def test_compute_rollout_offset_colocate_returns_zero(self):
        from miles.ray.rollout.rollout_server import _compute_rollout_offset

        args = make_args(colocate=True, debug_train_only=False, debug_rollout_only=False,
                         actor_num_nodes=1, actor_num_gpus_per_node=8, use_critic=False)
        assert _compute_rollout_offset(args) == 0

    def test_compute_rollout_offset_critic_train_only(self):
        from miles.ray.rollout.rollout_server import _compute_rollout_offset

        args = make_args(colocate=False, debug_train_only=False, debug_rollout_only=False,
                         critic_train_only=True, critic_num_nodes=1, critic_num_gpus_per_node=4)
        assert _compute_rollout_offset(args) == 4

    def test_compute_rollout_offset_actor_plus_critic(self):
        from miles.ray.rollout.rollout_server import _compute_rollout_offset

        args = make_args(colocate=False, debug_train_only=False, debug_rollout_only=False,
                         critic_train_only=False, use_critic=True,
                         actor_num_nodes=1, actor_num_gpus_per_node=8,
                         critic_num_nodes=1, critic_num_gpus_per_node=4)
        assert _compute_rollout_offset(args) == 12

    def test_compute_megatron_num_gpus_for_actor_only(self):
        from miles.ray.rollout.rollout_server import _compute_megatron_num_gpus
        args = make_args(actor_num_nodes=2, actor_num_gpus_per_node=8,
                         use_critic=False, debug_rollout_only=False, critic_train_only=False)
        assert _compute_megatron_num_gpus(args) == 16

    def test_compute_megatron_num_gpus_with_critic(self):
        from miles.ray.rollout.rollout_server import _compute_megatron_num_gpus
        args = make_args(actor_num_nodes=1, actor_num_gpus_per_node=8,
                         use_critic=True, critic_num_nodes=1, critic_num_gpus_per_node=4,
                         debug_rollout_only=False, critic_train_only=False)
        assert _compute_megatron_num_gpus(args) == 12

    def test_compute_megatron_num_gpus_zero_when_debug_rollout_only(self):
        from miles.ray.rollout.rollout_server import _compute_megatron_num_gpus
        args = make_args(debug_rollout_only=True)
        assert _compute_megatron_num_gpus(args) == 0


# --------------------------- rollout_server cross-group properties ---------------------------


def _build_dataclass_group(*, num_engines: int = 2, num_gpus_per_engine: int = 1,
                           gpu_offset: int = 0) -> "ServerGroup":
    """Build a ServerGroup with ``pg=None`` for pure dataclass-property tests
    that never schedule actors. Each engine is unallocated."""
    from miles.ray.rollout.server_engine import ServerEngine
    from miles.ray.rollout.server_group import ServerGroup
    args = make_args(num_gpus_per_node=8)
    engines = [ServerEngine() for _ in range(num_engines)]
    return ServerGroup(
        args=args, pg=None, all_engines=engines,
        num_gpus_per_engine=num_gpus_per_engine, has_new_engines=False,
        gpu_offset=gpu_offset, update_weights=True,
    )


class TestRolloutServerCrossGroupProperties:
    """RolloutServer cross-group flatten properties — pure dataclass logic,
    no Ray actors required."""

    def test_engines_collects_node0_engines_from_each_group(self):
        from miles.ray.rollout.rollout_server import RolloutServer
        a = _build_dataclass_group(num_engines=2, gpu_offset=0)
        b = _build_dataclass_group(num_engines=2, gpu_offset=2)
        srv = RolloutServer(server_groups=[a, b])
        assert len(srv.engines) == 4

    def test_engine_gpu_counts_parallel_to_engines(self):
        from miles.ray.rollout.rollout_server import RolloutServer
        a = _build_dataclass_group(num_engines=2, num_gpus_per_engine=1)
        b = _build_dataclass_group(num_engines=2, num_gpus_per_engine=2)
        srv = RolloutServer(server_groups=[a, b])
        assert srv.engine_gpu_counts == [1, 1, 2, 2]

    def test_engine_gpu_offsets_consistent_across_groups(self):
        from miles.ray.rollout.rollout_server import RolloutServer
        a = _build_dataclass_group(num_engines=2, num_gpus_per_engine=1, gpu_offset=0)
        b = _build_dataclass_group(num_engines=2, num_gpus_per_engine=2, gpu_offset=4)
        srv = RolloutServer(server_groups=[a, b])
        assert srv.engine_gpu_offsets == [0, 1, 4, 6]


class TestRolloutServerNodesPerEngineHeterogeneity:
    def test_homogeneous_groups_return_single_value(self):
        from miles.ray.rollout.rollout_server import RolloutServer
        a = _build_dataclass_group(num_gpus_per_engine=1)
        b = _build_dataclass_group(num_gpus_per_engine=1)
        srv = RolloutServer(server_groups=[a, b])
        assert srv.nodes_per_engine == 1

    def test_heterogeneous_groups_raise_value_error(self):
        from miles.ray.rollout.rollout_server import RolloutServer
        a = _build_dataclass_group(num_gpus_per_engine=1)    # 1 gpu/engine, 8 gpu/node → 1 node/engine
        b = _build_dataclass_group(num_gpus_per_engine=16)   # 16 gpu/engine → 2 nodes/engine
        srv = RolloutServer(server_groups=[a, b])
        with pytest.raises(ValueError, match="Heterogeneous nodes_per_engine"):
            _ = srv.nodes_per_engine


# --------------------------- rollout_manager pure routing helpers ---------------------------


def _build_dataclass_server(*, model_name: str, update_weights: bool,
                            num_engines: int = 2, num_groups: int = 1) -> "RolloutServer":
    """Build a RolloutServer + ServerGroup chain with ``pg=None`` for tests
    that exercise rollout_manager routing without scheduling any actors."""
    from miles.ray.rollout.rollout_server import RolloutServer
    groups = [_build_dataclass_group(num_engines=num_engines) for _ in range(num_groups)]
    for g in groups:
        g.update_weights = update_weights
    return RolloutServer(server_groups=groups, model_name=model_name,
                         update_weights=update_weights)


def _build_fake_rollout_manager(servers: dict) -> "RolloutManager":
    """Construct a RolloutManager bypassing ``__init__`` (which would call
    ``start_rollout_servers``). Sets just the attrs the routing helpers touch."""
    import asyncio
    from miles.ray.rollout.rollout_manager import RolloutManager
    cls = RolloutManager.__ray_actor_class__
    fake = cls.__new__(cls)
    fake.servers = servers
    fake.rollout_engine_lock = asyncio.Lock()
    fake._health_monitors = []
    fake.rollout_id = 0
    return fake


class TestRolloutManagerGetUpdatableEmptyPaths:
    """Pure routing tests for RolloutManager.get_updatable_engines_and_lock —
    cover the two no-actor paths (no updatable server / multiple updatables)."""

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
