from __future__ import annotations

from unittest.mock import patch

import pytest

from miles.ray.rollout.router_manager import start_router, start_session_server
from tests.fast.ray.rollout.conftest import make_args


class TestRouterManagerErrorPaths:
    def test_pd_disagg_with_miles_router_asserts(self):
        args = make_args(use_miles_router=True, sglang_router_ip=None, sglang_router_port=None)
        with patch("miles.ray.rollout.router_manager.get_host_info", return_value=("h", "127.0.0.1")), \
             patch("miles.ray.rollout.router_manager.find_available_port", return_value=20000):
            with pytest.raises(AssertionError, match="miles router does not support PD"):
                start_router(args, has_pd_disaggregation=True, force_new=False)

    def test_port_conflict_raises_runtime_error(self):
        args = make_args(use_miles_router=False, sglang_router_ip=None, sglang_router_port=None)
        with patch("miles.ray.rollout.router_manager.get_host_info", return_value=("h", "127.0.0.1")), \
             patch("miles.ray.rollout.router_manager.find_available_port", return_value=20000), \
             patch("miles.ray.rollout.router_manager.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="already in use"):
                start_router(args)

    def test_session_server_disabled_returns_silently(self):
        args = make_args(use_session_server=False)
        start_session_server(args)

    def test_session_server_without_hf_checkpoint_raises(self):
        args = make_args(use_session_server=True, hf_checkpoint=None)
        with pytest.raises(ValueError, match="hf-checkpoint"):
            start_session_server(args)

    def test_session_server_port_conflict_raises_runtime_error(self):
        """Symmetric to start_router's port-conflict path: when the configured
        session_server port is already bound, fail loud rather than silently
        re-using the stale process."""
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
