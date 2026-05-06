"""P0.1 — pure-logic UT for miles/ray/rollout/addr_allocator.py.

Covers:
- PortCursors empty/assign/next_base_port
- allocate_rollout_engine_addr_and_ports_normal: single-node 8-card,
  multi-node, prefill, gpus>node_size, rank_offset != 0, mid-rank restart
- allocate_rollout_engine_addr_and_ports_external

Bug history: this module's import path bug fixed in e1c9c8486."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from miles.ray.rollout.addr_allocator import (
    PortCursors,
    allocate_rollout_engine_addr_and_ports_external,
    allocate_rollout_engine_addr_and_ports_normal,
)
from tests.fast.ray.rollout.conftest import make_args


# ----------------------------- PortCursors -----------------------------


class TestPortCursors:
    def test_empty_has_no_values(self):
        c = PortCursors.empty()
        assert c._values == {}

    def test_next_base_port_default_when_empty(self):
        assert PortCursors.empty().next_base_port() == 15000

    def test_next_base_port_returns_max_value(self):
        c = PortCursors(_values={0: 17000, 1: 16500, 2: 18000})
        assert c.next_base_port() == 18000

    def test_assign_copies_values(self):
        a = PortCursors.empty()
        b = PortCursors(_values={0: 19000, 1: 19500})
        a.assign(b)
        assert a._values == {0: 19000, 1: 19500}

    def test_assign_is_decoupled(self):
        """After assign, mutating source must not bleed into target."""
        a = PortCursors.empty()
        b = PortCursors(_values={0: 19000})
        a.assign(b)
        b._values[0] = 99999
        assert a._values == {0: 19000}, "assign must deep-copy the inner dict"


# ----------------------------- normal allocator helpers -----------------------------


def _fake_engine(host: str = "10.0.0.1", port_seed: int = 30000):
    """Build a MagicMock that mimics ``SGLangEngine`` enough for addr_allocator."""
    e = MagicMock()
    e._port_cursor = port_seed

    def _get_addr_port(start_port: int = 15000, consecutive: int = 1):
        # mock ray.get(...) by returning value directly via .remote(...) MagicMock
        port = max(e._port_cursor, start_port)
        e._port_cursor = port + consecutive
        return (host, port)

    # `engine._get_current_node_ip_and_free_port.remote(...)` returns ObjectRef
    # in real code; we monkey-patch the path used after `ray.get(...)`.
    e._get_current_node_ip_and_free_port.remote.side_effect = lambda **kw: _get_addr_port(**kw)
    return e


@pytest.fixture
def patch_ray_get(monkeypatch):
    """ray.get(remote_call(...)) → use the mock's return value directly."""
    import miles.ray.rollout.addr_allocator as mod

    monkeypatch.setattr(mod.ray, "get", lambda x: x)


# ----------------------------- normal allocator -----------------------------


class TestAllocateNormal:
    def test_single_node_8_cards_tp1(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, _fake_engine(host="10.0.0.1", port_seed=30000)) for rank in range(8)]
        addr_and_ports, cursors = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=1, base_port=30000
        )

        assert set(addr_and_ports.keys()) == set(range(8))
        for rank in range(8):
            assert addr_and_ports[rank]["host"] == "10.0.0.1"
            for k in ("port", "nccl_port", "engine_info_bootstrap_port", "dist_init_addr"):
                assert k in addr_and_ports[rank]
        assert isinstance(cursors, PortCursors)

    def test_prefill_worker_gets_disagg_bootstrap_port(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, _fake_engine()) for rank in range(2)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args,
            rollout_engines=engines,
            worker_type="prefill",
            num_gpus_per_engine=1,
        )
        for rank in range(2):
            assert "disaggregation_bootstrap_port" in addr_and_ports[rank]

    def test_regular_worker_does_not_get_disagg_bootstrap_port(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, _fake_engine()) for rank in range(2)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=1
        )
        for rank in range(2):
            assert "disaggregation_bootstrap_port" not in addr_and_ports[rank]

    def test_gpus_per_engine_greater_than_node_shares_dist_init_addr(self, patch_ray_get):
        """When `_gpus_per_engine > num_gpus_per_node`, all ranks of one engine
        share a single ``dist_init_addr`` (multi-node engine)."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        # 2-node engine: 16 gpus total, 8 per node, 2 ranks share dist_init_addr
        engines = [(rank, _fake_engine()) for rank in range(2)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=16
        )
        assert addr_and_ports[0]["dist_init_addr"] == addr_and_ports[1]["dist_init_addr"]

    def test_rank_offset_does_not_break_indexing(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, _fake_engine()) for rank in (4, 5, 6, 7)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=1, rank_offset=4
        )
        for r in (4, 5, 6, 7):
            assert r in addr_and_ports
            assert addr_and_ports[r]["host"]
            assert addr_and_ports[r]["port"]

    def test_mid_rank_restart_fills_remaining_slots_on_node(self, patch_ray_get):
        """Restarting starting from rank 3 on an 8-card node should populate
        addr_and_ports[3..7], i.e. ``num_engines_on_this_node`` accounts for
        the offset within the node."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(3, _fake_engine())]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=1
        )
        for r in range(3, 8):
            assert r in addr_and_ports

    def test_base_port_propagates_into_cursor(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(0, _fake_engine(port_seed=22000))]
        _, cursors = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=1, base_port=22000
        )
        # cursor for node 0 should be > 22000 after allocation
        assert cursors._values[0] > 22000


# ----------------------------- external allocator -----------------------------


class TestAllocateExternal:
    def test_basic_split(self):
        args = make_args(rollout_external_engine_addrs=["10.0.0.1:30000", "10.0.0.2:30001"])
        engines = [(0, MagicMock()), (1, MagicMock())]
        result = allocate_rollout_engine_addr_and_ports_external(args=args, rollout_engines=engines)
        assert result[0] == dict(dist_init_addr="10.0.0.1:30000", nccl_port=None, host="10.0.0.1", port=30000)
        assert result[1] == dict(dist_init_addr="10.0.0.2:30001", nccl_port=None, host="10.0.0.2", port=30001)

    def test_ipv4_addr_split_is_consistent(self):
        args = make_args(rollout_external_engine_addrs=["192.168.1.10:31000"])
        engines = [(0, MagicMock())]
        result = allocate_rollout_engine_addr_and_ports_external(args=args, rollout_engines=engines)
        assert result[0]["host"] == "192.168.1.10"
        assert result[0]["port"] == 31000
