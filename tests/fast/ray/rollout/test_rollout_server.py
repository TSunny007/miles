"""Pure-Python tests for ``miles/ray/rollout/rollout_server.py``.

Covers the resolve / compute helpers and the cross-group dataclass
properties (engines / engine_gpu_counts / engine_gpu_offsets /
nodes_per_engine).

Tests that need real Ray actors live in ``real_ray/test_rollout_server.py``."""

from __future__ import annotations

import pytest

from tests.fast.ray.rollout.conftest import make_args, make_dataclass_group


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


class TestRolloutServerCrossGroupProperties:
    """Cross-group flatten properties — pure dataclass logic, no Ray actors."""

    def test_engines_collects_node0_engines_from_each_group(self):
        from miles.ray.rollout.rollout_server import RolloutServer
        a = make_dataclass_group(num_engines=2, gpu_offset=0)
        b = make_dataclass_group(num_engines=2, gpu_offset=2)
        srv = RolloutServer(server_groups=[a, b])
        assert len(srv.engines) == 4

    def test_engine_gpu_counts_parallel_to_engines(self):
        from miles.ray.rollout.rollout_server import RolloutServer
        a = make_dataclass_group(num_engines=2, num_gpus_per_engine=1)
        b = make_dataclass_group(num_engines=2, num_gpus_per_engine=2)
        srv = RolloutServer(server_groups=[a, b])
        assert srv.engine_gpu_counts == [1, 1, 2, 2]

    def test_engine_gpu_offsets_consistent_across_groups(self):
        from miles.ray.rollout.rollout_server import RolloutServer
        a = make_dataclass_group(num_engines=2, num_gpus_per_engine=1, gpu_offset=0)
        b = make_dataclass_group(num_engines=2, num_gpus_per_engine=2, gpu_offset=4)
        srv = RolloutServer(server_groups=[a, b])
        assert srv.engine_gpu_offsets == [0, 1, 4, 6]


class TestRolloutServerNodesPerEngineHeterogeneity:
    def test_homogeneous_groups_return_single_value(self):
        from miles.ray.rollout.rollout_server import RolloutServer
        a = make_dataclass_group(num_gpus_per_engine=1)
        b = make_dataclass_group(num_gpus_per_engine=1)
        srv = RolloutServer(server_groups=[a, b])
        assert srv.nodes_per_engine == 1

    def test_heterogeneous_groups_raise_value_error(self):
        from miles.ray.rollout.rollout_server import RolloutServer
        a = make_dataclass_group(num_gpus_per_engine=1)    # 1 gpu/engine, 8 gpu/node → 1 node/engine
        b = make_dataclass_group(num_gpus_per_engine=16)   # 16 gpu/engine → 2 nodes/engine
        srv = RolloutServer(server_groups=[a, b])
        with pytest.raises(ValueError, match="Heterogeneous nodes_per_engine"):
            _ = srv.nodes_per_engine
