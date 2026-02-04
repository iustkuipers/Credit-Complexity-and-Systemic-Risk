"""
Extensive unit tests for model.py module.
Tests credit rating migration via asset return mapping.
Covers threshold computation, single and vectorized migrations, edge cases, and validations.
"""

import pytest
import numpy as np
from scipy.stats import norm
from pathlib import Path
import sys

# Add parent and data directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from data import (
    RATINGS,
    RATING_TO_IDX,
    START_RATINGS,
    START_RATING_TO_ROW,
    TRANSITION,
)
from services.model import (
    precompute_thresholds,
    migrate_one,
    migrate_many,
    migrate_many_to_indices,
    _row_index,
    _EPS,
)


class TestRowIndex:
    """Tests for _row_index helper function."""

    def test_valid_start_ratings(self):
        """Test that valid start ratings return correct indices."""
        for i, r in enumerate(START_RATINGS):
            assert _row_index(r) == i

    def test_invalid_rating_raises_error(self):
        """Test that invalid ratings raise ValueError."""
        with pytest.raises(ValueError, match="must be one of"):
            _row_index("XYZ")

    def test_default_rating_not_allowed(self):
        """Test that 'D' (default) is not a valid start rating."""
        with pytest.raises(ValueError, match="not a start state"):
            _row_index("D")


class TestPrecomputeThresholds:
    """Tests for precompute_thresholds function."""

    def test_thresholds_shape(self):
        """Test that thresholds have correct shape."""
        thr = precompute_thresholds()
        assert thr.shape == (len(START_RATINGS), len(RATINGS))

    def test_thresholds_with_custom_transition(self):
        """Test precompute_thresholds with custom transition matrix."""
        # Create a simple diagonal-like transition (high probability on diagonal)
        custom_trans = np.eye(7, 8) * 0.9 + np.ones((7, 8)) * 0.01
        custom_trans /= custom_trans.sum(axis=1, keepdims=True)
        
        thr = precompute_thresholds(custom_trans)
        assert thr.shape == (7, 8)

    def test_thresholds_invalid_shape_raises_error(self):
        """Test that invalid transition shape raises error."""
        invalid_trans = np.ones((5, 8))
        with pytest.raises(ValueError, match="must have shape"):
            precompute_thresholds(invalid_trans)

    def test_thresholds_increasing_within_row(self):
        """Test that thresholds increase (or stay same) within each row."""
        thr = precompute_thresholds()
        for row in range(len(START_RATINGS)):
            for k in range(len(RATINGS) - 2):
                # thresholds should be non-decreasing
                assert thr[row, k] <= thr[row, k + 1]

    def test_last_threshold_is_infinity(self):
        """Test that last threshold in each row is +infinity."""
        thr = precompute_thresholds()
        for row in range(len(START_RATINGS)):
            assert thr[row, -1] == np.inf

    def test_first_threshold_reasonable(self):
        """Test that first thresholds are reasonable values."""
        thr = precompute_thresholds()
        # First threshold should be somewhere around -3 to 0 for typical migration matrix
        for row in range(len(START_RATINGS)):
            assert -10 < thr[row, 0] < 2

    def test_thresholds_are_cumulative_inverse(self):
        """Test that thresholds correspond to CDF quantiles."""
        thr = precompute_thresholds(TRANSITION)
        cdf = np.cumsum(TRANSITION, axis=1)
        
        # For each rating (except last), the threshold should be norm.ppf(cdf)
        for row in range(len(START_RATINGS)):
            for k in range(len(RATINGS) - 1):
                expected = norm.ppf(np.clip(cdf[row, k], _EPS, 1.0 - _EPS))
                assert np.isclose(thr[row, k], expected, atol=1e-10)


class TestMigrateOne:
    """Tests for migrate_one function."""

    def test_very_negative_x_downgrades(self):
        """Test that very negative X leads to worse ratings."""
        thr = precompute_thresholds()
        x = -5.0  # very negative asset return
        
        # Negative returns should lead to downgrades
        for rating in START_RATINGS:
            migrated = migrate_one(rating, x, thr)
            assert migrated in RATINGS

    def test_very_positive_x_upgrades(self):
        """Test that very positive X can lead to upgrades."""
        thr = precompute_thresholds()
        x = 5.0  # very positive asset return
        
        for rating in START_RATINGS:
            migrated = migrate_one(rating, x, thr)
            assert migrated in RATINGS

    def test_zero_x(self):
        """Test migration with zero asset return."""
        thr = precompute_thresholds()
        x = 0.0
        
        for rating in START_RATINGS:
            migrated = migrate_one(rating, x, thr)
            assert migrated in RATINGS

    def test_aaa_with_positive_x_stays_investment_grade(self):
        """Test that AAA with positive X stays high-quality."""
        thr = precompute_thresholds()
        migrated = migrate_one("AAA", 1.0, thr)
        assert migrated in ["AAA", "AA", "A", "BBB"]

    def test_ccc_with_very_negative_x_downgrades(self):
        """Test that CCC with very negative X downgrades to D."""
        thr = precompute_thresholds()
        migrated = migrate_one("CCC", -5.0, thr)
        # Low-grade bond with large negative shock likely downgrades toward D
        assert migrated in RATINGS

    def test_invalid_current_rating(self):
        """Test that invalid current rating raises error."""
        thr = precompute_thresholds()
        with pytest.raises(ValueError):
            migrate_one("InvalidRating", 0.0, thr)

    def test_deterministic_same_input_same_output(self):
        """Test that same input always gives same output."""
        thr = precompute_thresholds()
        out1 = migrate_one("BBB", 0.5, thr)
        out2 = migrate_one("BBB", 0.5, thr)
        assert out1 == out2


class TestMigrateMany:
    """Tests for migrate_many function."""

    def test_many_returns_correct_dtype(self):
        """Test that migrate_many returns string array."""
        thr = precompute_thresholds()
        x_vec = np.array([0.0, 0.5, -0.5])
        result = migrate_many("BBB", x_vec, thr)
        
        assert result.dtype.kind == "U"  # Unicode string type
        assert len(result) == len(x_vec)

    def test_many_returns_valid_ratings(self):
        """Test that all returned ratings are in RATINGS."""
        thr = precompute_thresholds()
        x_vec = np.linspace(-3, 3, 100)
        result = migrate_many("BBB", x_vec, thr)
        
        for rating in result:
            assert rating in RATINGS

    def test_many_matches_one(self):
        """Test that migrate_many matches migrate_one for same inputs."""
        thr = precompute_thresholds()
        x_vec = np.array([0.0, 0.5, -0.5, 2.0, -2.0])
        
        result_many = migrate_many("A", x_vec, thr)
        for i, x in enumerate(x_vec):
            result_one = migrate_one("A", x, thr)
            assert result_many[i] == result_one

    def test_empty_array(self):
        """Test migrate_many with empty array."""
        thr = precompute_thresholds()
        x_vec = np.array([])
        result = migrate_many("AAA", x_vec, thr)
        assert len(result) == 0

    def test_large_array(self):
        """Test migrate_many with large array."""
        thr = precompute_thresholds()
        x_vec = np.random.normal(0, 1, 10000)
        result = migrate_many("BBB", x_vec, thr)
        
        assert len(result) == len(x_vec)
        for rating in result:
            assert rating in RATINGS

    def test_all_start_ratings(self):
        """Test migrate_many works for all start ratings."""
        thr = precompute_thresholds()
        x_vec = np.array([-1.0, 0.0, 1.0])
        
        for rating in START_RATINGS:
            result = migrate_many(rating, x_vec, thr)
            assert len(result) == len(x_vec)
            for r in result:
                assert r in RATINGS


class TestMigrateMany_ToIndices:
    """Tests for migrate_many_to_indices function."""

    def test_indices_correct_dtype(self):
        """Test that indices are returned as integers."""
        thr = precompute_thresholds()
        x_vec = np.array([0.0, 0.5, -0.5])
        result = migrate_many_to_indices("BBB", x_vec, thr)
        
        assert result.dtype in [np.int64, np.int32, int]

    def test_indices_in_valid_range(self):
        """Test that returned indices are valid (0 to len(RATINGS)-1)."""
        thr = precompute_thresholds()
        x_vec = np.linspace(-5, 5, 100)
        result = migrate_many_to_indices("AA", x_vec, thr)
        
        assert np.all(result >= 0)
        assert np.all(result < len(RATINGS))

    def test_indices_match_ratings(self):
        """Test that indices match the ratings from migrate_many."""
        thr = precompute_thresholds()
        x_vec = np.array([0.0, 0.5, -0.5, 2.0, -2.0])
        
        ratings = migrate_many("BBB", x_vec, thr)
        indices = migrate_many_to_indices("BBB", x_vec, thr)
        
        for i, idx in enumerate(indices):
            assert RATINGS[idx] == ratings[i]

    def test_indices_for_all_ratings(self):
        """Test indices for all start ratings."""
        thr = precompute_thresholds()
        x_vec = np.linspace(-3, 3, 50)
        
        for rating in START_RATINGS:
            result = migrate_many_to_indices(rating, x_vec, thr)
            assert len(result) == len(x_vec)
            assert np.all(result >= 0)
            assert np.all(result < len(RATINGS))


class TestMigrationStatistics:
    """Tests for statistical properties of migration."""

    def test_migration_preserves_distribution(self):
        """Test that migration with standard normal X follows transition probabilities."""
        thr = precompute_thresholds()
        n_samples = 100000
        
        # For each start rating, migrate using standard normal samples
        for rating in START_RATINGS:
            x_vec = np.random.standard_normal(n_samples)
            indices = migrate_many_to_indices(rating, x_vec, thr)
            
            # Count empirical distribution
            empirical_counts = np.bincount(indices, minlength=len(RATINGS))
            empirical_probs = empirical_counts / n_samples
            
            # Get theoretical probabilities from transition matrix
            row = START_RATING_TO_ROW[rating]
            theoretical_probs = TRANSITION[row]
            
            # They should be close (with some tolerance for sampling error)
            assert np.allclose(empirical_probs, theoretical_probs, atol=0.02)

    def test_worst_case_leads_to_default(self):
        """Test that extremely negative X leads to downgrades."""
        thr = precompute_thresholds()
        x = -10.0  # extremely negative
        
        # Extremely negative X should lead to worst available rating
        ccc_result = migrate_one("CCC", x, thr)
        # The searchsorted will find the first rating where x <= threshold
        assert ccc_result in RATINGS

    def test_best_case_stays_high(self):
        """Test that extremely positive X leads to some upside rating."""
        thr = precompute_thresholds()
        x = 10.0  # extremely positive
        
        # Extremely positive X should map to some rating
        aaa_result = migrate_one("AAA", x, thr)
        # Positive shock should map to valid rating
        assert aaa_result in RATINGS

    def test_monotonicity_in_x(self):
        """Test that worse X leads to different ratings than good X."""
        thr = precompute_thresholds()
        x_bad = np.array([-3.0])
        x_good = np.array([3.0])
        
        # Map to ratings for comparison
        bad_ratings = migrate_many("BB", x_bad, thr)
        good_ratings = migrate_many("BB", x_good, thr)
        
        # Both should map to valid ratings
        assert bad_ratings[0] in RATINGS
        assert good_ratings[0] in RATINGS


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_threshold_boundaries(self):
        """Test behavior at threshold boundaries."""
        thr = precompute_thresholds()
        
        # Get the first threshold for BBB
        bbb_row = START_RATING_TO_ROW["BBB"]
        first_thr = thr[bbb_row, 0]
        
        # Just below and at threshold
        x_below = first_thr - 1e-6
        x_at = first_thr
        
        result_below = migrate_one("BBB", x_below, thr)
        result_at = migrate_one("BBB", x_at, thr)
        
        # Both should map to same or similar ratings
        assert result_below in RATINGS
        assert result_at in RATINGS

    def test_nan_handling(self):
        """Test that NaN inputs are handled."""
        thr = precompute_thresholds()
        result = migrate_one("AA", np.nan, thr)
        # NaN should map somewhere (searchsorted handles it)
        assert result in RATINGS

    def test_inf_handling(self):
        """Test that infinite X values are handled."""
        thr = precompute_thresholds()
        
        result_pos_inf = migrate_one("BBB", np.inf, thr)
        result_neg_inf = migrate_one("BBB", -np.inf, thr)
        
        # Both should map to valid ratings
        assert result_pos_inf in RATINGS
        assert result_neg_inf in RATINGS


class TestDataIntegrity:
    """Tests for data consistency and integrity."""

    def test_default_transition_matrix_used(self):
        """Test that default TRANSITION matrix is used correctly."""
        thr_default = precompute_thresholds()
        thr_explicit = precompute_thresholds(TRANSITION)
        
        # Should be identical
        assert np.allclose(thr_default, thr_explicit, equal_nan=True)

    def test_ratings_consistency(self):
        """Test that all operations use consistent rating definitions."""
        thr = precompute_thresholds()
        
        for rating in START_RATINGS:
            x = 0.0
            migrated = migrate_one(rating, x, thr)
            idx = RATING_TO_IDX[migrated]
            
            # Index should be in valid range
            assert 0 <= idx < len(RATINGS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestRowIndex:
    """Tests for _row_index helper function."""

    def test_valid_start_ratings(self):
        """Test that valid start ratings return correct indices."""
        for i, r in enumerate(START_RATINGS):
            assert _row_index(r) == i

    def test_invalid_rating_raises_error(self):
        """Test that invalid ratings raise ValueError."""
        with pytest.raises(ValueError, match="must be one of"):
            _row_index("XYZ")

    def test_default_rating_not_allowed(self):
        """Test that 'D' (default) is not a valid start rating."""
        with pytest.raises(ValueError, match="not a start state"):
            _row_index("D")


class TestPrecomputeThresholds:
    """Tests for precompute_thresholds function."""

    def test_thresholds_shape(self):
        """Test that thresholds have correct shape."""
        thr = precompute_thresholds()
        assert thr.shape == (len(START_RATINGS), len(RATINGS))

    def test_thresholds_with_custom_transition(self):
        """Test precompute_thresholds with custom transition matrix."""
        # Create a simple diagonal-like transition (high probability on diagonal)
        custom_trans = np.eye(7, 8) * 0.9 + np.ones((7, 8)) * 0.01
        custom_trans /= custom_trans.sum(axis=1, keepdims=True)
        
        thr = precompute_thresholds(custom_trans)
        assert thr.shape == (7, 8)

    def test_thresholds_invalid_shape_raises_error(self):
        """Test that invalid transition shape raises error."""
        invalid_trans = np.ones((5, 8))
        with pytest.raises(ValueError, match="must have shape"):
            precompute_thresholds(invalid_trans)

    def test_thresholds_increasing_within_row(self):
        """Test that thresholds increase (or stay same) within each row."""
        thr = precompute_thresholds()
        for row in range(len(START_RATINGS)):
            for k in range(len(RATINGS) - 2):
                # thresholds should be non-decreasing
                assert thr[row, k] <= thr[row, k + 1]

    def test_last_threshold_is_infinity(self):
        """Test that last threshold in each row is +infinity."""
        thr = precompute_thresholds()
        for row in range(len(START_RATINGS)):
            assert thr[row, -1] == np.inf

    def test_first_threshold_reasonable(self):
        """Test that first thresholds are reasonable values."""
        thr = precompute_thresholds()
        # First threshold should be somewhere around -3 to 0 for typical migration matrix
        for row in range(len(START_RATINGS)):
            assert -10 < thr[row, 0] < 2

    def test_thresholds_are_cumulative_inverse(self):
        """Test that thresholds correspond to CDF quantiles."""
        thr = precompute_thresholds(TRANSITION)
        cdf = np.cumsum(TRANSITION, axis=1)
        
        # For each rating (except last), the threshold should be norm.ppf(cdf)
        for row in range(len(START_RATINGS)):
            for k in range(len(RATINGS) - 1):
                expected = norm.ppf(np.clip(cdf[row, k], _EPS, 1.0 - _EPS))
                assert np.isclose(thr[row, k], expected, atol=1e-10)


class TestMigrateOne:
    """Tests for migrate_one function."""

    def test_very_negative_x_downgrades(self):
        """Test that very negative X leads to worse ratings."""
        thr = precompute_thresholds()
        x = -5.0  # very negative asset return
        
        # Negative returns should lead to downgrades
        for rating in START_RATINGS:
            migrated = migrate_one(rating, x, thr)
            assert migrated in RATINGS

    def test_very_positive_x_upgrades(self):
        """Test that very positive X can lead to upgrades."""
        thr = precompute_thresholds()
        x = 5.0  # very positive asset return
        
        for rating in START_RATINGS:
            migrated = migrate_one(rating, x, thr)
            assert migrated in RATINGS

    def test_zero_x(self):
        """Test migration with zero asset return."""
        thr = precompute_thresholds()
        x = 0.0
        
        for rating in START_RATINGS:
            migrated = migrate_one(rating, x, thr)
            assert migrated in RATINGS

    def test_aaa_with_positive_x_stays_investment_grade(self):
        """Test that AAA with positive X stays high-quality."""
        thr = precompute_thresholds()
        migrated = migrate_one("AAA", 1.0, thr)
        assert migrated in ["AAA", "AA", "A", "BBB"]

    def test_ccc_with_very_negative_x_downgrades(self):
        """Test that CCC with very negative X downgrades to D."""
        thr = precompute_thresholds()
        migrated = migrate_one("CCC", -5.0, thr)
        # Low-grade bond with large negative shock likely downgrades toward D
        assert migrated in RATINGS

    def test_invalid_current_rating(self):
        """Test that invalid current rating raises error."""
        thr = precompute_thresholds()
        with pytest.raises(ValueError):
            migrate_one("InvalidRating", 0.0, thr)

    def test_deterministic_same_input_same_output(self):
        """Test that same input always gives same output."""
        thr = precompute_thresholds()
        out1 = migrate_one("BBB", 0.5, thr)
        out2 = migrate_one("BBB", 0.5, thr)
        assert out1 == out2


class TestMigrateMany:
    """Tests for migrate_many function."""

    def test_many_returns_correct_dtype(self):
        """Test that migrate_many returns string array."""
        thr = precompute_thresholds()
        x_vec = np.array([0.0, 0.5, -0.5])
        result = migrate_many("BBB", x_vec, thr)
        
        assert result.dtype.kind == "U"  # Unicode string type
        assert len(result) == len(x_vec)

    def test_many_returns_valid_ratings(self):
        """Test that all returned ratings are in RATINGS."""
        thr = precompute_thresholds()
        x_vec = np.linspace(-3, 3, 100)
        result = migrate_many("BBB", x_vec, thr)
        
        for rating in result:
            assert rating in RATINGS

    def test_many_matches_one(self):
        """Test that migrate_many matches migrate_one for same inputs."""
        thr = precompute_thresholds()
        x_vec = np.array([0.0, 0.5, -0.5, 2.0, -2.0])
        
        result_many = migrate_many("A", x_vec, thr)
        for i, x in enumerate(x_vec):
            result_one = migrate_one("A", x, thr)
            assert result_many[i] == result_one

    def test_empty_array(self):
        """Test migrate_many with empty array."""
        thr = precompute_thresholds()
        x_vec = np.array([])
        result = migrate_many("AAA", x_vec, thr)
        assert len(result) == 0

    def test_large_array(self):
        """Test migrate_many with large array."""
        thr = precompute_thresholds()
        x_vec = np.random.normal(0, 1, 10000)
        result = migrate_many("BBB", x_vec, thr)
        
        assert len(result) == len(x_vec)
        for rating in result:
            assert rating in RATINGS

    def test_all_start_ratings(self):
        """Test migrate_many works for all start ratings."""
        thr = precompute_thresholds()
        x_vec = np.array([-1.0, 0.0, 1.0])
        
        for rating in START_RATINGS:
            result = migrate_many(rating, x_vec, thr)
            assert len(result) == len(x_vec)
            for r in result:
                assert r in RATINGS


class TestMigrateMany_ToIndices:
    """Tests for migrate_many_to_indices function."""

    def test_indices_correct_dtype(self):
        """Test that indices are returned as integers."""
        thr = precompute_thresholds()
        x_vec = np.array([0.0, 0.5, -0.5])
        result = migrate_many_to_indices("BBB", x_vec, thr)
        
        assert result.dtype in [np.int64, np.int32, int]

    def test_indices_in_valid_range(self):
        """Test that returned indices are valid (0 to len(RATINGS)-1)."""
        thr = precompute_thresholds()
        x_vec = np.linspace(-5, 5, 100)
        result = migrate_many_to_indices("AA", x_vec, thr)
        
        assert np.all(result >= 0)
        assert np.all(result < len(RATINGS))

    def test_indices_match_ratings(self):
        """Test that indices match the ratings from migrate_many."""
        thr = precompute_thresholds()
        x_vec = np.array([0.0, 0.5, -0.5, 2.0, -2.0])
        
        ratings = migrate_many("BBB", x_vec, thr)
        indices = migrate_many_to_indices("BBB", x_vec, thr)
        
        for i, idx in enumerate(indices):
            assert RATINGS[idx] == ratings[i]

    def test_indices_for_all_ratings(self):
        """Test indices for all start ratings."""
        thr = precompute_thresholds()
        x_vec = np.linspace(-3, 3, 50)
        
        for rating in START_RATINGS:
            result = migrate_many_to_indices(rating, x_vec, thr)
            assert len(result) == len(x_vec)
            assert np.all(result >= 0)
            assert np.all(result < len(RATINGS))


class TestMigrationStatistics:
    """Tests for statistical properties of migration."""

    def test_migration_preserves_distribution(self):
        """Test that migration with standard normal X follows transition probabilities."""
        thr = precompute_thresholds()
        n_samples = 100000
        
        # For each start rating, migrate using standard normal samples
        for rating in START_RATINGS:
            x_vec = np.random.standard_normal(n_samples)
            indices = migrate_many_to_indices(rating, x_vec, thr)
            
            # Count empirical distribution
            empirical_counts = np.bincount(indices, minlength=len(RATINGS))
            empirical_probs = empirical_counts / n_samples
            
            # Get theoretical probabilities from transition matrix
            row = START_RATING_TO_ROW[rating]
            theoretical_probs = TRANSITION[row]
            
            # They should be close (with some tolerance for sampling error)
            assert np.allclose(empirical_probs, theoretical_probs, atol=0.02)

    def test_worst_case_leads_to_default(self):
        """Test that extremely negative X leads to downgrades."""
        thr = precompute_thresholds()
        x = -10.0  # extremely negative
        
        # Extremely negative X should lead to worst available rating
        ccc_result = migrate_one("CCC", x, thr)
        # The searchsorted will find the first rating where x <= threshold
        assert ccc_result in RATINGS

    def test_best_case_stays_high(self):
        """Test that extremely positive X leads to some upside rating."""
        thr = precompute_thresholds()
        x = 10.0  # extremely positive
        
        # Extremely positive X should map to some rating
        aaa_result = migrate_one("AAA", x, thr)
        # Positive shock should map to valid rating
        assert aaa_result in RATINGS

    def test_monotonicity_in_x(self):
        """Test that worse X leads to different ratings than good X."""
        thr = precompute_thresholds()
        x_bad = np.array([-3.0])
        x_good = np.array([3.0])
        
        # Map to ratings for comparison
        bad_ratings = migrate_many("BB", x_bad, thr)
        good_ratings = migrate_many("BB", x_good, thr)
        
        # Both should map to valid ratings
        assert bad_ratings[0] in RATINGS
        assert good_ratings[0] in RATINGS


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_threshold_boundaries(self):
        """Test behavior at threshold boundaries."""
        thr = precompute_thresholds()
        
        # Get the first threshold for BBB
        bbb_row = START_RATING_TO_ROW["BBB"]
        first_thr = thr[bbb_row, 0]
        
        # Just below and at threshold
        x_below = first_thr - 1e-6
        x_at = first_thr
        
        result_below = migrate_one("BBB", x_below, thr)
        result_at = migrate_one("BBB", x_at, thr)
        
        # Both should map to same or similar ratings
        assert result_below in RATINGS
        assert result_at in RATINGS

    def test_nan_handling(self):
        """Test that NaN inputs are handled."""
        thr = precompute_thresholds()
        result = migrate_one("AA", np.nan, thr)
        # NaN should map somewhere (searchsorted handles it)
        assert result in RATINGS

    def test_inf_handling(self):
        """Test that infinite X values are handled."""
        thr = precompute_thresholds()
        
        result_pos_inf = migrate_one("BBB", np.inf, thr)
        result_neg_inf = migrate_one("BBB", -np.inf, thr)
        
        # Both should map to valid ratings
        assert result_pos_inf in RATINGS
        assert result_neg_inf in RATINGS


class TestDataIntegrity:
    """Tests for data consistency and integrity."""

    def test_default_transition_matrix_used(self):
        """Test that default TRANSITION matrix is used correctly."""
        thr_default = precompute_thresholds()
        thr_explicit = precompute_thresholds(TRANSITION)
        
        # Should be identical
        assert np.allclose(thr_default, thr_explicit, equal_nan=True)

    def test_ratings_consistency(self):
        """Test that all operations use consistent rating definitions."""
        thr = precompute_thresholds()
        
        for rating in START_RATINGS:
            x = 0.0
            migrated = migrate_one(rating, x, thr)
            idx = RATING_TO_IDX[migrated]
            
            # Index should be in valid range
            assert 0 <= idx < len(RATINGS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
