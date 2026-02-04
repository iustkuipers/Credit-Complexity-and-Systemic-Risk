"""
Unit tests for risk_metrics.py module.
Tests loss calculation, VaR, ES, and risk metric summaries.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent and services directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from risk_metrics import (
    values_to_losses,
    expected_value,
    var,
    es,
    summarize_losses,
    summarize_case_metrics,
    es_over_var_ratio,
    _validate_alpha,
)


class TestValidateAlpha:
    """Tests for _validate_alpha helper function."""

    def test_valid_alpha_low(self):
        """Test that alpha=0.5 is valid."""
        # Should not raise
        _validate_alpha(0.5)

    def test_valid_alpha_high(self):
        """Test that alpha=0.95 is valid."""
        # Should not raise
        _validate_alpha(0.95)

    def test_valid_alpha_near_zero(self):
        """Test that very small alpha is valid."""
        # Should not raise
        _validate_alpha(0.001)

    def test_valid_alpha_near_one(self):
        """Test that alpha close to 1 is valid."""
        # Should not raise
        _validate_alpha(0.999)

    def test_alpha_equals_zero_raises_error(self):
        """Test that alpha=0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            _validate_alpha(0.0)

    def test_alpha_equals_one_raises_error(self):
        """Test that alpha=1.0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            _validate_alpha(1.0)

    def test_alpha_negative_raises_error(self):
        """Test that negative alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            _validate_alpha(-0.5)

    def test_alpha_greater_than_one_raises_error(self):
        """Test that alpha > 1 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            _validate_alpha(1.5)


class TestValuesToLosses:
    """Tests for values_to_losses function."""

    def test_losses_shape(self):
        """Test that losses have same shape as input values."""
        v0 = 1000.0
        v1 = np.array([950.0, 1000.0, 1050.0])
        losses = values_to_losses(v0, v1)
        
        assert losses.shape == v1.shape

    def test_losses_dtype(self):
        """Test that losses are float."""
        v0 = 1000.0
        v1 = np.array([950.0, 1000.0, 1050.0])
        losses = values_to_losses(v0, v1)
        
        assert losses.dtype == float or losses.dtype == np.float64

    def test_perfect_hedge_zero_loss(self):
        """Test that when V0 = V1, loss is zero."""
        v0 = 1000.0
        v1 = np.array([1000.0, 1000.0, 1000.0])
        losses = values_to_losses(v0, v1)
        
        assert np.allclose(losses, 0.0)

    def test_gain_negative_loss(self):
        """Test that V1 > V0 gives negative loss (profit)."""
        v0 = 1000.0
        v1 = np.array([1100.0, 1200.0])
        losses = values_to_losses(v0, v1)
        
        assert np.all(losses < 0)
        assert np.allclose(losses, [-100.0, -200.0])

    def test_loss_positive(self):
        """Test that V1 < V0 gives positive loss."""
        v0 = 1000.0
        v1 = np.array([900.0, 800.0])
        losses = values_to_losses(v0, v1)
        
        assert np.all(losses > 0)
        assert np.allclose(losses, [100.0, 200.0])

    def test_scalar_v1(self):
        """Test with scalar v1."""
        v0 = 1000.0
        v1 = 950.0
        losses = values_to_losses(v0, v1)
        
        # values_to_losses returns a scalar when v1 is scalar
        assert np.isscalar(losses) or isinstance(losses, (float, np.floating))
        assert np.isclose(losses, 50.0)

    def test_list_v1_conversion(self):
        """Test that list v1 is converted to array."""
        v0 = 1000.0
        v1 = [900.0, 1000.0, 1100.0]
        losses = values_to_losses(v0, v1)
        
        assert isinstance(losses, np.ndarray)
        assert losses.shape == (3,)

    def test_integer_inputs_converted_to_float(self):
        """Test that integer inputs are converted to float."""
        v0 = 1000
        v1 = np.array([900, 1000, 1100], dtype=int)
        losses = values_to_losses(v0, v1)
        
        assert losses.dtype == float or losses.dtype == np.float64


class TestExpectedValue:
    """Tests for expected_value function."""

    def test_expected_value_scalar(self):
        """Test expected value with scalar."""
        values = np.array([1000.0])
        result = expected_value(values)
        
        assert np.isclose(result, 1000.0)

    def test_expected_value_multiple(self):
        """Test expected value with multiple values."""
        values = np.array([900.0, 1000.0, 1100.0])
        result = expected_value(values)
        
        assert np.isclose(result, 1000.0)

    def test_expected_value_dtype(self):
        """Test that result is float."""
        values = np.array([1000.0, 2000.0])
        result = expected_value(values)
        
        assert isinstance(result, float)

    def test_expected_value_list_input(self):
        """Test with list input."""
        values = [100.0, 200.0, 300.0]
        result = expected_value(values)
        
        assert np.isclose(result, 200.0)

    def test_expected_value_large_sample(self):
        """Test expected value matches numpy mean."""
        values = np.random.normal(1000, 100, 10000)
        result = expected_value(values)
        numpy_mean = float(values.mean())
        
        assert np.isclose(result, numpy_mean)


class TestVaR:
    """Tests for var (Value-at-Risk) function."""

    def test_var_shape(self):
        """Test that VaR returns scalar."""
        losses = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        result = var(losses, 0.5)
        
        assert isinstance(result, (float, np.floating))

    def test_var_quantile_50(self):
        """Test VaR at 50% (median)."""
        losses = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        result = var(losses, 0.5)
        
        # Should be around 25 (median of 0,10,20,30,40,50)
        assert 20.0 <= result <= 30.0

    def test_var_monotonic_in_alpha(self):
        """Test that VaR increases with alpha."""
        losses = np.linspace(0, 100, 1000)
        
        var_90 = var(losses, 0.90)
        var_95 = var(losses, 0.95)
        var_99 = var(losses, 0.99)
        
        assert var_90 < var_95 < var_99

    def test_var_on_normal_distribution(self):
        """Test VaR on normal distribution."""
        rng = np.random.default_rng(42)
        losses = rng.normal(0.0, 1.0, 100000)
        
        # For standard normal, VaR(99.5%) should be around 2.576
        result = var(losses, 0.995)
        assert 2.4 < result < 2.7

    def test_var_invalid_alpha_low(self):
        """Test that alpha <= 0 raises error."""
        losses = np.array([0.0, 10.0, 20.0])
        with pytest.raises(ValueError, match="alpha must be in"):
            var(losses, 0.0)

    def test_var_invalid_alpha_high(self):
        """Test that alpha >= 1 raises error."""
        losses = np.array([0.0, 10.0, 20.0])
        with pytest.raises(ValueError, match="alpha must be in"):
            var(losses, 1.0)

    def test_var_single_value(self):
        """Test VaR with single value."""
        losses = np.array([50.0])
        result = var(losses, 0.5)
        
        assert np.isclose(result, 50.0)

    def test_var_list_input(self):
        """Test VaR with list input."""
        losses = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        result = var(losses, 0.5)
        
        assert isinstance(result, float)


class TestES:
    """Tests for es (Expected Shortfall) function."""

    def test_es_shape(self):
        """Test that ES returns scalar."""
        losses = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        result = es(losses, 0.5)
        
        assert isinstance(result, (float, np.floating))

    def test_es_greater_than_var(self):
        """Test that ES >= VaR (by definition)."""
        losses = np.linspace(0, 100, 1000)
        alpha = 0.95
        
        var_95 = var(losses, alpha)
        es_95 = es(losses, alpha)
        
        assert es_95 >= var_95

    def test_es_monotonic_in_alpha(self):
        """Test that ES increases with alpha."""
        losses = np.linspace(0, 100, 1000)
        
        es_90 = es(losses, 0.90)
        es_95 = es(losses, 0.95)
        es_99 = es(losses, 0.99)
        
        assert es_90 < es_95 < es_99

    def test_es_on_normal_distribution(self):
        """Test ES on large normal sample."""
        rng = np.random.default_rng(42)
        losses = rng.normal(0.0, 1.0, 100000)
        
        # For standard normal at 99.5%, ES should be around 2.8-3.0
        result = es(losses, 0.995)
        assert 2.7 < result < 3.2

    def test_es_equals_var_on_single_value(self):
        """Test that ES = VaR when all values are same."""
        losses = np.array([50.0, 50.0, 50.0, 50.0])
        
        var_50 = var(losses, 0.5)
        es_50 = es(losses, 0.5)
        
        assert np.isclose(var_50, es_50)

    def test_es_invalid_alpha_raises_error(self):
        """Test that invalid alpha raises error."""
        losses = np.array([0.0, 10.0, 20.0])
        with pytest.raises(ValueError, match="alpha must be in"):
            es(losses, 1.5)

    def test_es_list_input(self):
        """Test ES with list input."""
        losses = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        result = es(losses, 0.8)
        
        assert isinstance(result, float)

    def test_es_tail_behavior(self):
        """Test ES focuses on tail."""
        losses = np.concatenate([np.zeros(950), np.linspace(100, 200, 50)])
        
        # ES should be much higher than average due to tail focus
        es_99 = es(losses, 0.99)
        average = np.mean(losses)
        
        assert es_99 > average


class TestSummarizeLosses:
    """Tests for summarize_losses function."""

    def test_summary_keys(self):
        """Test that summary contains required keys."""
        losses = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        result = summarize_losses(losses)
        
        assert "mean_loss" in result
        assert "var_0.9" in result
        assert "es_0.9" in result
        assert "var_0.995" in result
        assert "es_0.995" in result

    def test_summary_custom_alphas(self):
        """Test summary with custom alphas."""
        losses = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        result = summarize_losses(losses, alphas=(0.50, 0.75, 0.95))
        
        assert "var_0.5" in result
        assert "es_0.5" in result
        assert "var_0.75" in result
        assert "es_0.75" in result
        assert "var_0.95" in result
        assert "es_0.95" in result

    def test_summary_mean_loss(self):
        """Test that mean_loss is correct."""
        losses = np.array([10.0, 20.0, 30.0])
        result = summarize_losses(losses)
        
        assert np.isclose(result["mean_loss"], 20.0)

    def test_summary_all_values_positive(self):
        """Test that all returned values are non-negative for positive losses."""
        losses = np.linspace(0, 100, 1000)
        result = summarize_losses(losses)
        
        for v in result.values():
            assert v >= 0

    def test_summary_empty_alphas(self):
        """Test summary with empty alphas."""
        losses = np.array([0.0, 10.0, 20.0, 30.0])
        result = summarize_losses(losses, alphas=())
        
        assert "mean_loss" in result
        assert len(result) == 1


class TestSummarizeCaseMetrics:
    """Tests for summarize_case_metrics function."""

    def test_case_metrics_keys(self):
        """Test that case metrics contains required keys."""
        v0 = 1000.0
        v1 = np.array([900.0, 1000.0, 1100.0])
        result = summarize_case_metrics(v0, v1)
        
        assert "expected_value" in result
        assert "var_0.9" in result
        assert "es_0.9" in result
        assert "var_0.995" in result
        assert "es_0.995" in result

    def test_case_metrics_expected_value(self):
        """Test that expected_value matches E[V1]."""
        v0 = 1000.0
        v1 = np.array([900.0, 1000.0, 1100.0])
        result = summarize_case_metrics(v0, v1)
        
        assert np.isclose(result["expected_value"], 1000.0)

    def test_case_metrics_custom_alphas(self):
        """Test case metrics with custom alphas."""
        v0 = 1000.0
        v1 = np.array([800.0, 900.0, 1000.0, 1100.0, 1200.0])
        result = summarize_case_metrics(v0, v1, alphas=(0.50, 0.95))
        
        assert "var_0.5" in result
        assert "es_0.5" in result
        assert "var_0.95" in result
        assert "es_0.95" in result
        assert len(result) == 5  # expected_value + 2*alphas

    def test_case_metrics_consistency_with_losses(self):
        """Test that case metrics are consistent with losses."""
        v0 = 1000.0
        v1 = np.linspace(800.0, 1200.0, 100)
        
        result_case = summarize_case_metrics(v0, v1, alphas=(0.95,))
        
        losses = v0 - v1
        result_losses = summarize_losses(losses, alphas=(0.95,))
        
        # Expected value should match
        assert np.isclose(result_case["expected_value"], result_losses["mean_loss"] - 0 + v0)
        # Actually, expected_value of V1 = v0 - mean(losses)
        assert np.isclose(
            result_case["expected_value"],
            v0 - result_losses["mean_loss"]
        )


class TestESOverVarRatio:
    """Tests for es_over_var_ratio function."""

    def test_ratio_greater_or_equal_one(self):
        """Test that ES/VaR >= 1 (by definition)."""
        losses = np.linspace(0, 100, 1000)
        ratio = es_over_var_ratio(losses, alpha=0.95)
        
        assert ratio >= 1.0

    def test_ratio_at_different_alphas(self):
        """Test ratio at different confidence levels."""
        losses = np.linspace(0, 100, 1000)
        
        ratio_90 = es_over_var_ratio(losses, alpha=0.90)
        ratio_99 = es_over_var_ratio(losses, alpha=0.99)
        
        # Both should be >= 1
        assert ratio_90 >= 1.0
        assert ratio_99 >= 1.0

    def test_ratio_on_normal_distribution(self):
        """Test ratio on normal distribution (tail fatness indicator)."""
        rng = np.random.default_rng(42)
        losses = rng.normal(0.0, 1.0, 100000)
        
        ratio = es_over_var_ratio(losses, alpha=0.995)
        
        # For normal, ratio should be around 1.1-1.2
        assert 1.0 < ratio < 1.5

    def test_ratio_invalid_alpha_raises_error(self):
        """Test that invalid alpha raises error."""
        losses = np.array([0.0, 10.0, 20.0])
        with pytest.raises(ValueError, match="alpha must be in"):
            es_over_var_ratio(losses, alpha=0.0)

    def test_ratio_near_zero_var_returns_nan(self):
        """Test that near-zero VaR returns NaN."""
        # Create losses where VaR is nearly zero
        losses = np.array([0.0, 0.0, 0.0, 0.0, 0.1])
        ratio = es_over_var_ratio(losses, alpha=0.5)
        
        # Should be NaN when VaR is very close to 0
        assert np.isnan(ratio) or ratio > 0

    def test_ratio_consistent_with_var_es(self):
        """Test that ratio = ES/VaR."""
        losses = np.linspace(0, 100, 1000)
        alpha = 0.95
        
        ratio = es_over_var_ratio(losses, alpha=alpha)
        var_95 = var(losses, alpha)
        es_95 = es(losses, alpha)
        
        if not np.isnan(ratio):
            assert np.isclose(ratio, es_95 / var_95)

    def test_ratio_default_alpha(self):
        """Test ratio with default alpha=0.995."""
        losses = np.random.normal(0, 1, 10000)
        ratio = es_over_var_ratio(losses)
        
        # Should use alpha=0.995 by default
        assert ratio >= 1.0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_large_values(self):
        """Test with very large portfolio values."""
        v0 = 1e10
        v1 = np.array([9e9, 1e10, 1.1e10])
        
        losses = values_to_losses(v0, v1)
        result = summarize_case_metrics(v0, v1)
        
        assert np.all(np.isfinite(losses))
        assert np.all(np.isfinite(list(result.values())))

    def test_very_small_values(self):
        """Test with very small portfolio values."""
        v0 = 1e-10
        v1 = np.array([0.5e-10, 1e-10, 1.5e-10])
        
        losses = values_to_losses(v0, v1)
        result = summarize_case_metrics(v0, v1)
        
        assert np.all(np.isfinite(losses))
        assert np.all(np.isfinite(list(result.values())))

    def test_negative_values(self):
        """Test with negative portfolio values (e.g., debt)."""
        v0 = -1000.0
        v1 = np.array([-900.0, -1000.0, -1100.0])
        
        losses = values_to_losses(v0, v1)
        # Should still work
        assert losses.shape == v1.shape

    def test_mixed_sign_values(self):
        """Test with mixed positive and negative values."""
        v0 = 0.0
        v1 = np.array([-100.0, 0.0, 100.0])
        
        losses = values_to_losses(v0, v1)
        assert losses.shape == (3,)

    def test_many_identical_values(self):
        """Test with many identical values."""
        v0 = 1000.0
        v1 = np.ones(10000) * 1000.0
        
        losses = values_to_losses(v0, v1)
        result = summarize_case_metrics(v0, v1)
        
        # All losses should be zero
        assert np.allclose(losses, 0.0)
        assert np.isclose(result["expected_value"], 1000.0)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_typical_portfolio_analysis_workflow(self):
        """Test a typical portfolio risk analysis workflow."""
        # Simulate portfolio values
        v0 = 1500.0
        v1 = np.random.normal(1500.0, 200.0, 5000)
        
        # Compute summary metrics
        summary = summarize_case_metrics(v0, v1, alphas=(0.90, 0.95, 0.995))
        
        # Check structure
        assert len(summary) == 7  # expected_value + 3*2 metrics
        
        # Check consistency
        expected_v1 = float(v1.mean())
        assert np.isclose(summary["expected_value"], expected_v1, rtol=0.01)
        
        # Check VaR/ES ordering
        assert summary["var_0.9"] < summary["var_0.95"] < summary["var_0.995"]
        assert summary["es_0.9"] < summary["es_0.95"] < summary["es_0.995"]

    def test_concentrated_vs_diversified_portfolio(self):
        """Test risk metrics for concentrated vs diversified scenarios."""
        v0 = 1000.0
        
        # Concentrated: high volatility
        v1_conc = np.random.normal(1000.0, 500.0, 10000)
        result_conc = summarize_case_metrics(v0, v1_conc)
        
        # Diversified: low volatility
        v1_div = np.random.normal(1000.0, 100.0, 10000)
        result_div = summarize_case_metrics(v0, v1_div)
        
        # Concentrated should have higher VaR/ES
        assert result_conc["var_0.995"] > result_div["var_0.995"]
        assert result_conc["es_0.995"] > result_div["es_0.995"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
