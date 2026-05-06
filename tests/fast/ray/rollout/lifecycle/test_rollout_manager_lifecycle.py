"""P1.3 — RolloutManager lifecycle: start_cell / stop_cell / _get_updatable_server.

Driven against the underlying servers dict directly (rather than the @ray.remote
RolloutManager actor) since we only need to validate the dispatch logic."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import get_cell_indexer_of_id_map
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.rollout.server_group import ServerGroup
from tests.fast.ray.rollout.conftest import fake_actor_handle, make_args


def _build_servers(*, num_servers: int = 1, groups_per_server: int = 1,
                   engines_per_group: int = 2, num_gpus_per_engine: int = 1,
                   updatable: list[bool] | None = None) -> dict[str, RolloutServer]:
    args = make_args(num_gpus_per_node=8)
    updatable = updatable or [True] * num_servers
    servers: dict[str, RolloutServer] = {}
    for s_idx in range(num_servers):
        groups: list[ServerGroup] = []
        for _g in range(groups_per_server):
            engines = [ServerEngine() for _ in range(engines_per_group)]
            for e in engines:
                e.mark_allocated_uninitialized(fake_actor_handle())
                e.mark_alive()
            g = ServerGroup(
                args=args, pg=None, all_engines=engines,
                num_gpus_per_engine=num_gpus_per_engine, has_new_engines=False,
                update_weights=updatable[s_idx],
            )
            groups.append(g)
        servers[f"model_{s_idx}"] = RolloutServer(
            server_groups=groups, model_name=f"model_{s_idx}", update_weights=updatable[s_idx],
        )
    return servers


# ----------------------------- start_cell / stop_cell -----------------------------


class TestCellDispatch:
    def test_cell_indexer_dispatches_single_server(self):
        servers = _build_servers(num_servers=1, groups_per_server=1, engines_per_group=3)
        cells = get_cell_indexer_of_id_map(servers)
        assert len(cells) == 3
        for cell in cells:
            assert cell.srv_key == "model_0"
            assert cell.group_index == 0
            assert len(cell.engine_indices) == 1  # nodes_per_engine=1

    def test_cell_indexer_dispatches_multi_group_continuous_ids(self):
        servers = _build_servers(num_servers=1, groups_per_server=2, engines_per_group=2)
        cells = get_cell_indexer_of_id_map(servers)
        assert len(cells) == 4
        # cells 0,1 → group 0; cells 2,3 → group 1
        assert cells[0].group_index == 0 and cells[1].group_index == 0
        assert cells[2].group_index == 1 and cells[3].group_index == 1

    def test_cell_indexer_orders_servers_by_key(self):
        # build servers in non-alphabetical insertion order; sorted-by-key applies
        servers = _build_servers(num_servers=2, groups_per_server=1, engines_per_group=1)
        cells = get_cell_indexer_of_id_map(servers)
        srv_keys_in_order = [c.srv_key for c in cells]
        assert srv_keys_in_order == sorted(srv_keys_in_order)

    def test_cell_engine_indices_span_nodes_per_engine(self):
        """When num_gpus_per_engine=16 (>num_gpus_per_node=8), each cell maps to
        2 contiguous engine slots."""
        servers = _build_servers(num_servers=1, groups_per_server=1,
                                 engines_per_group=2, num_gpus_per_engine=16)
        cells = get_cell_indexer_of_id_map(servers)
        assert len(cells) == 1  # 2 engine slots → 1 multi-node cell
        assert cells[0].engine_indices == [0, 1]


# ----------------------------- get_updatable_server -----------------------------


class TestGetUpdatableServer:
    """Drives the real free function ``get_updatable_server`` (extracted from
    ``RolloutManager._get_updatable_server`` so it can be tested without
    materializing the @ray.remote actor)."""

    def test_returns_none_when_no_updatable_server(self):
        from miles.ray.rollout.rollout_manager import get_updatable_server
        servers = _build_servers(num_servers=2, updatable=[False, False])
        assert get_updatable_server(servers) is None

    def test_returns_the_one_updatable_server(self):
        from miles.ray.rollout.rollout_manager import get_updatable_server
        servers = _build_servers(num_servers=2, updatable=[False, True])
        srv = get_updatable_server(servers)
        assert srv is not None
        assert srv.model_name == "model_1"

    def test_multiple_updatable_servers_raises_assertion(self):
        from miles.ray.rollout.rollout_manager import get_updatable_server
        servers = _build_servers(num_servers=2, updatable=[True, True])
        with pytest.raises(AssertionError, match="Multiple servers"):
            get_updatable_server(servers)


# ----------------------------- start_rollout_servers ---------------------------


class TestStartRolloutServersHelpers:
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
