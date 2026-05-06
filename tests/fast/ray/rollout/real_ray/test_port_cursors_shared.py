"""Two ``ServerGroup``s sharing one ``PortCursors`` must produce disjoint
port allocations across nodes — see ``server_group.py:37`` for the source-level
concern flagged on parallel recover."""

from __future__ import annotations

from miles.ray.rollout.addr_allocator import PortCursors, allocate_rollout_engine_addr_and_ports_normal
from tests.fast.ray.rollout.conftest import fake_engine, make_args


class TestSharedPortCursorsAcrossGroups:
    def test_sequential_groups_share_cursor_and_avoid_overlap(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        cursors = PortCursors.empty()

        engines_a = [(rank, fake_engine(port_seed=0)) for rank in range(4)]
        addrs_a, next_a = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines_a, num_gpus_per_engine=1,
            base_port=cursors.next_base_port(),
        )
        cursors.assign(next_a)

        engines_b = [(rank, fake_engine(port_seed=0)) for rank in range(4, 8)]
        addrs_b, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines_b, num_gpus_per_engine=1,
            base_port=cursors.next_base_port(),
            rank_offset=4,
        )

        ports_a = {addrs_a[r]["port"] for r in addrs_a} | {addrs_a[r]["nccl_port"] for r in addrs_a}
        ports_b = {addrs_b[r]["port"] for r in addrs_b} | {addrs_b[r]["nccl_port"] for r in addrs_b}
        assert ports_a.isdisjoint(ports_b), f"port overlap A={ports_a} B={ports_b}"
