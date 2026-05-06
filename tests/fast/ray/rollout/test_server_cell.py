from __future__ import annotations

from unittest.mock import MagicMock

from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import get_cell_indexer_of_id_map
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.rollout.server_group import ServerGroup
from tests.fast.ray.rollout.conftest import fake_actor_handle, make_args


def _build_servers(*, num_servers: int = 1, groups_per_server: int = 1,
                   engines_per_group: int = 2, num_gpus_per_engine: int = 1) -> dict[str, RolloutServer]:
    args = make_args(num_gpus_per_node=8)
    servers: dict[str, RolloutServer] = {}
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
        ``len(all_engines) == len(engines) * nodes_per_engine`` still holds
        (0 == 0 * N)."""
        srv = MagicMock()
        group = MagicMock()
        group.all_engines = []
        group.engines = []
        group.nodes_per_engine = 2
        srv.server_groups = [group]
        out = get_cell_indexer_of_id_map({"only": srv})
        assert out == []

    def test_dispatches_single_server(self):
        servers = _build_servers(num_servers=1, groups_per_server=1, engines_per_group=3)
        cells = get_cell_indexer_of_id_map(servers)
        assert len(cells) == 3
        for cell in cells:
            assert cell.srv_key == "model_0"
            assert cell.group_index == 0
            assert len(cell.engine_indices) == 1

    def test_dispatches_multi_group_continuous_ids(self):
        servers = _build_servers(num_servers=1, groups_per_server=2, engines_per_group=2)
        cells = get_cell_indexer_of_id_map(servers)
        assert len(cells) == 4
        assert cells[0].group_index == 0 and cells[1].group_index == 0
        assert cells[2].group_index == 1 and cells[3].group_index == 1

    def test_orders_servers_by_key(self):
        servers = _build_servers(num_servers=2, groups_per_server=1, engines_per_group=1)
        cells = get_cell_indexer_of_id_map(servers)
        srv_keys_in_order = [c.srv_key for c in cells]
        assert srv_keys_in_order == sorted(srv_keys_in_order)

    def test_engine_indices_span_nodes_per_engine(self):
        """When num_gpus_per_engine=16 (>num_gpus_per_node=8), each cell maps to
        2 contiguous engine slots."""
        servers = _build_servers(num_servers=1, groups_per_server=1,
                                 engines_per_group=2, num_gpus_per_engine=16)
        cells = get_cell_indexer_of_id_map(servers)
        assert len(cells) == 1
        assert cells[0].engine_indices == [0, 1]
