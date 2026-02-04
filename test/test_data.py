"""
Unit tests for data.py module.
Tests ratings, transition matrix, bond values, and portfolio functions.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add parent directory to path to import data module
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from data import (
    RATINGS,
    RATING_TO_IDX,
    START_RATINGS,
    START_RATING_TO_ROW,
    TRANSITION,
    BOND_VALUE_T0,
    BOND_VALUE_T1,
    make_portfolio,
    portfolio_value_t0,
    validate_case_primitives,
)


class TestConstants:
    """Tests for constant definitions."""

    def test_ratings_order(self):
        """Verify ratings are in correct order (best to worst)."""
        assert RATINGS == ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]

    def test_rating_to_idx_mapping(self):
        """Verify rating to index mapping is correct."""
        for i, r in enumerate(RATINGS):
            assert RATING_TO_IDX[r] == i
        assert len(RATING_TO_IDX) == len(RATINGS)

    def test_start_ratings_order(self):
        """Verify start ratings exclude 'D' and are ordered correctly."""
        assert START_RATINGS == ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
        assert "D" not in START_RATINGS
        assert len(START_RATINGS) == 7

    def test_start_rating_to_row_mapping(self):
        """Verify start rating to row index mapping is correct."""
        for i, r in enumerate(START_RATINGS):
            assert START_RATING_TO_ROW[r] == i
        assert len(START_RATING_TO_ROW) == len(START_RATINGS)

    def test_transition_matrix_shape(self):
        """Verify transition matrix has correct shape."""
        assert TRANSITION.shape == (7, 8)

    def test_transition_matrix_rows_sum_to_one(self):
        """Verify each row of transition matrix sums to 1."""
        row_sums = TRANSITION.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_transition_matrix_non_negative(self):
        """Verify all transition probabilities are non-negative."""
        assert np.all(TRANSITION >= 0)

    def test_bond_value_t0_completeness(self):
        """Verify bond values at t=0 are defined for all start ratings."""
        for r in START_RATINGS:
            assert r in BOND_VALUE_T0
        assert "D" not in BOND_VALUE_T0

    def test_bond_value_t1_completeness(self):
        """Verify bond values at t=1 are defined for all ratings."""
        for r in RATINGS:
            assert r in BOND_VALUE_T1

    def test_bond_values_positive(self):
        """Verify bond values are positive."""
        for r, v in BOND_VALUE_T0.items():
            assert v > 0, f"Bond value for {r} at t=0 must be positive"
        for r, v in BOND_VALUE_T1.items():
            assert v > 0, f"Bond value for {r} at t=1 must be positive"

    def test_bond_values_reasonable_range(self):
        """Verify bond values are within reasonable range (typically [0, 100])."""
        for r, v in {**BOND_VALUE_T0, **BOND_VALUE_T1}.items():
            assert 0 < v <= 100, f"Bond value for {r} outside expected range"


class TestMakePortfolio:
    """Tests for make_portfolio function."""

    def test_valid_portfolio(self):
        """Test creating a valid portfolio."""
        p = make_portfolio({"AAA": 0.5, "AA": 0.5})
        assert p == {"AAA": 0.5, "AA": 0.5}

    def test_portfolio_normalizes_when_requested(self):
        """Test portfolio normalization."""
        p = make_portfolio({"AAA": 2.0, "AA": 2.0}, normalize=True)
        assert np.isclose(sum(p.values()), 1.0)
        assert p["AAA"] == 0.5
        assert p["AA"] == 0.5

    def test_portfolio_removes_zero_weights(self):
        """Test that zero weights are removed from portfolio."""
        p = make_portfolio({"AAA": 0.5, "AA": 0.5, "A": 0.0})
        assert "A" not in p
        assert len(p) == 2

    def test_portfolio_rejects_empty(self):
        """Test that empty portfolio is rejected."""
        with pytest.raises(ValueError, match="empty"):
            make_portfolio({})

    def test_portfolio_rejects_all_zero_weights(self):
        """Test that portfolio with all zero weights is rejected."""
        with pytest.raises(ValueError, match="zero total weight"):
            make_portfolio({"AAA": 0.0, "AA": 0.0})

    def test_portfolio_rejects_negative_weights(self):
        """Test that negative weights are rejected."""
        with pytest.raises(ValueError, match="Negative weight"):
            make_portfolio({"AAA": 0.5, "AA": -0.5})

    def test_portfolio_rejects_unknown_rating(self):
        """Test that unknown ratings are rejected."""
        with pytest.raises(ValueError, match="Unknown rating"):
            make_portfolio({"XYZ": 0.5})

    def test_portfolio_rejects_default_rating(self):
        """Test that 'D' (default) rating is rejected."""
        with pytest.raises(ValueError, match="must not include 'D'"):
            make_portfolio({"AAA": 0.5, "D": 0.5})

    def test_portfolio_rejects_non_normalized_without_flag(self):
        """Test that non-normalized portfolio is rejected unless normalize=True."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            make_portfolio({"AAA": 0.5, "AA": 0.3})

    def test_portfolio_accepts_weights_summing_to_one(self):
        """Test that weights summing to 1.0 are accepted."""
        p = make_portfolio({"AAA": 0.6, "AA": 0.3, "A": 0.1})
        assert np.isclose(sum(p.values()), 1.0)

    def test_portfolio_all_start_ratings(self):
        """Test portfolio with all start ratings."""
        weights = {r: 1.0 / len(START_RATINGS) for r in START_RATINGS}
        p = make_portfolio(weights)
        assert len(p) == len(START_RATINGS)
        assert np.isclose(sum(p.values()), 1.0)


class TestPortfolioValueT0:
    """Tests for portfolio_value_t0 function."""

    def test_single_rating_portfolio_value(self):
        """Test value calculation for single-rating portfolio."""
        p = make_portfolio({"AAA": 1.0})
        v = portfolio_value_t0(p, total_notional=100.0)
        expected = 1.0 * 100.0 * (BOND_VALUE_T0["AAA"] / 100.0)
        assert np.isclose(v, expected)

    def test_portfolio_value_with_different_notionals(self):
        """Test that value scales with notional."""
        p = make_portfolio({"AAA": 1.0})
        v1 = portfolio_value_t0(p, total_notional=1000.0)
        v2 = portfolio_value_t0(p, total_notional=2000.0)
        assert np.isclose(v2, 2 * v1)

    def test_multi_rating_portfolio_value(self):
        """Test value calculation for multi-rating portfolio."""
        p = make_portfolio({"AAA": 0.5, "AA": 0.5})
        v = portfolio_value_t0(p, total_notional=1000.0)
        
        expected = (
            0.5 * 1000.0 * (BOND_VALUE_T0["AAA"] / 100.0) +
            0.5 * 1000.0 * (BOND_VALUE_T0["AA"] / 100.0)
        )
        assert np.isclose(v, expected)

    def test_portfolio_value_with_case_example(self):
        """Test with example from case data."""
        p = make_portfolio({"AAA": 0.6, "AA": 0.3, "BBB": 0.1})
        v = portfolio_value_t0(p, total_notional=1500.0)
        
        expected = (
            0.6 * 1500.0 * (BOND_VALUE_T0["AAA"] / 100.0) +
            0.3 * 1500.0 * (BOND_VALUE_T0["AA"] / 100.0) +
            0.1 * 1500.0 * (BOND_VALUE_T0["BBB"] / 100.0)
        )
        assert np.isclose(v, expected)

    def test_portfolio_value_zero_notional(self):
        """Test portfolio value with zero notional."""
        p = make_portfolio({"AAA": 1.0})
        v = portfolio_value_t0(p, total_notional=0.0)
        assert np.isclose(v, 0.0)

    def test_portfolio_value_missing_bond_value(self):
        """Test error handling when bond value is missing."""
        # This should not happen with normal usage, but test the error
        p = {"InvalidRating": 0.5, "AAA": 0.5}
        with pytest.raises(ValueError, match="Missing"):
            portfolio_value_t0(p, total_notional=1000.0)


class TestValidateCasePrimitives:
    """Tests for validate_case_primitives function."""

    def test_validate_case_primitives_passes(self):
        """Test that validation passes for correct data."""
        # Should not raise any exception
        validate_case_primitives()

    def test_transition_matrix_rows_correct_length(self):
        """Verify transition matrix has 7 rows (one per start rating)."""
        assert TRANSITION.shape[0] == len(START_RATINGS)

    def test_transition_matrix_cols_correct_length(self):
        """Verify transition matrix has 8 columns (one per rating)."""
        assert TRANSITION.shape[1] == len(RATINGS)


class TestDataIntegrity:
    """Integration tests for data consistency."""

    def test_ratings_and_indices_consistent(self):
        """Test that rating indices are consistent."""
        for r in RATINGS:
            assert RATING_TO_IDX[r] == RATINGS.index(r)

    def test_bond_values_decreasing_with_risk(self):
        """Verify bond values generally decrease with credit risk (lower ratings)."""
        # AAA should generally be higher valued than CCC at same time
        assert BOND_VALUE_T0["AAA"] > BOND_VALUE_T0["CCC"]
        assert BOND_VALUE_T1["AAA"] > BOND_VALUE_T1["CCC"]

    def test_default_rating_has_recovery_value(self):
        """Verify that default rating has a recovery value at t=1."""
        assert BOND_VALUE_T1["D"] > 0
        assert BOND_VALUE_T1["D"] < BOND_VALUE_T1["CCC"]

    def test_portfolio_weights_type_coercion(self):
        """Test that portfolio accepts weights as ints and strings."""
        p1 = make_portfolio({"AAA": 1})  # int instead of float
        assert p1["AAA"] == 1

    def test_transition_matrix_double_dtype(self):
        """Verify transition matrix uses float dtype."""
        assert TRANSITION.dtype == float or TRANSITION.dtype == np.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
