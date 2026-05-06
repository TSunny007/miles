"""P0.4 — grab-bag of small but valuable single-case tests.

Each entry here is a one-shot edge case that doesn't warrant its own file.
Per the test plan, these are kept short — single test per concern."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.fast.ray.rollout.conftest import make_args, make_sample, make_samples_grouped


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
