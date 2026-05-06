"""rank ↔ addr_and_ports index consistency in ``ServerGroup.start_engines``.

server_group.py:153-160 builds init_handles by iterating ``new_engines`` with
``(global_rank, engine)`` pairs; addr_allocator.py:85 uses
``current_rank = rank + i`` as the dict key. When ``rank_offset != 0`` or
``nodes_per_engine > 1``, the two index spaces must agree."""

from __future__ import annotations

from miles.ray.rollout.addr_allocator import allocate_rollout_engine_addr_and_ports_normal
from tests.fast.ray.rollout.conftest import fake_engine, make_args


class TestRankPortConsistency:
    def test_rank_offset_kwargs_keyed_by_global_rank(self, patch_ray_get):
        """When rank_offset=4, addr_and_ports must be keyed by ranks 4..7,
        not 0..3.

        ``num_gpus_per_node=4`` so a 4-engine group exactly fills one node —
        otherwise the allocator pads up to ``num_engines_per_node``."""
        args = make_args(num_gpus_per_node=4, sglang_dp_size=1)
        engines = [(rank, fake_engine(port_seed=0)) for rank in range(4, 8)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=1, rank_offset=4,
        )
        assert set(addr_and_ports.keys()) == {4, 5, 6, 7}

    def test_each_global_rank_has_complete_kwargs(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine(port_seed=0)) for rank in range(4)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=1,
        )
        for rank in range(4):
            kw = addr_and_ports[rank]
            for key in ("host", "port", "nccl_port", "dist_init_addr"):
                assert key in kw, f"rank {rank} missing {key}"

    def test_multinode_engine_shares_dist_init_addr_across_node_ranks(self, patch_ray_get):
        """nodes_per_engine=2 (16 gpus, 8 per node) — both ranks of one
        multi-node engine MUST get the same dist_init_addr."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(0, fake_engine(port_seed=0)), (1, fake_engine(port_seed=0))]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=16,
        )
        assert addr_and_ports[0]["dist_init_addr"] == addr_and_ports[1]["dist_init_addr"]

    def test_init_handles_iteration_pairs_match_addr_dict(self, patch_ray_get):
        """For every (index, engine) pair in server_group.py's loop,
        addr_and_ports[index] must contain all required kwargs."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        new_engines = [(rank, fake_engine(port_seed=0)) for rank in range(2, 6)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=new_engines, num_gpus_per_engine=1, rank_offset=2,
        )
        for index, _engine in new_engines:
            assert index in addr_and_ports, f"missing addr_and_ports for global_rank={index}"
            for key in ("host", "port", "nccl_port", "dist_init_addr"):
                assert key in addr_and_ports[index]

    def test_ports_are_unique_within_a_node(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, fake_engine(port_seed=0)) for rank in range(8)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=1,
        )
        all_ports = []
        for kw in addr_and_ports.values():
            all_ports.extend([kw["port"], kw["nccl_port"], kw["engine_info_bootstrap_port"]])
        assert len(set(all_ports)) == len(all_ports), f"duplicate ports: {all_ports}"
