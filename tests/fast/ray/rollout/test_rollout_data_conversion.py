"""Tests for ``miles/ray/rollout/rollout_data_conversion.py``."""

from __future__ import annotations

import pytest

from tests.fast.ray.rollout.conftest import make_args, make_sample


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
