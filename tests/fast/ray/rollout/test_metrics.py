"""Tests for ``miles/ray/rollout/metrics.py``."""

from __future__ import annotations

import pytest

from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped


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
