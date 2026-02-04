"""
Unit tests for simulation.py module.
Tests Monte Carlo simulation of portfolio values under credit migration.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent and services directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from data import (
    RATINGS,
    START_RATINGS,
    BOND_VALUE_T1,
    make_portfolio,
)
from model import precompute_thresholds
from simulation import (
    _bond_values_t1_vector,
    simulate_one_factor_asset_returns,
    simulate_portfolio_values_t1,
)


class TestBondValuesT1Vector:
    """Tests for _bond_values_t1_vector helper function."""

    def test_bond_vector_shape(self):
        """Test that bond vector has correct shape."""
        vec = _bond_values_t1_vector()
        assert vec.shape == (len(RATINGS),)

    def test_bond_vector_values(self):
        """Test that bond vector values match BOND_VALUE_T1."""
        vec = _bond_values_t1_vector()
        for i, r in enumerate(RATINGS):
            assert np.isclose(vec[i], BOND_VALUE_T1[r])

    def test_bond_vector_dtype(self):
        """Test that bond vector is float dtype."""
        vec = _bond_values_t1_vector()
        assert vec.dtype == float or vec.dtype == np.float64

    def test_bond_vector_all_positive(self):
        """Test that all bond values are positive."""
        vec = _bond_values_t1_vector()
        assert np.all(vec > 0)


class TestSimulateOneFactorAssetReturns:
    """Tests for simulate_one_factor_asset_returns function."""

    def test_asset_returns_shape(self):
        """Test that returned array has correct shape."""
        n = 1000
        x = simulate_one_factor_asset_returns(n, rho=0.5, rng=np.random.default_rng(42))
        assert x.shape == (n,)

    def test_asset_returns_dtype(self):
        """Test that returned array is float."""
        x = simulate_one_factor_asset_returns(100, rho=0.5, rng=np.random.default_rng(42))
        assert x.dtype in [np.float64, float]

    def test_zero_correlation_iid(self):
        """Test that rho=0 generates approximately iid normal."""
        rng = np.random.default_rng(42)
        x = simulate_one_factor_asset_returns(10000, rho=0.0, rng=rng)
        
        # Check mean and std are approximately 0 and 1
        assert np.abs(np.mean(x)) < 0.05
        assert np.abs(np.std(x) - 1.0) < 0.05

    def test_full_correlation(self):
        """Test that rho=1 makes all returns identical."""
        rng = np.random.default_rng(42)
        x = simulate_one_factor_asset_returns(100, rho=1.0, rng=rng)
        
        # All elements should be equal
        assert np.allclose(x, x[0])

    def test_intermediate_correlation(self):
        """Test that 0 < rho < 1 produces correlated but distinct returns."""
        rng = np.random.default_rng(42)
        x = simulate_one_factor_asset_returns(100, rho=0.5, rng=rng)
        
        # Not all equal (with high probability)
        assert not np.allclose(x, x[0])
        # But should have reasonable variation
        assert np.std(x) > 0

    def test_negative_rho_raises_error(self):
        """Test that negative rho raises ValueError."""
        with pytest.raises(ValueError, match="rho must be in"):
            simulate_one_factor_asset_returns(100, rho=-0.1, rng=np.random.default_rng(42))

    def test_rho_greater_than_one_raises_error(self):
        """Test that rho > 1 raises ValueError."""
        with pytest.raises(ValueError, match="rho must be in"):
            simulate_one_factor_asset_returns(100, rho=1.5, rng=np.random.default_rng(42))

    def test_reproducibility_with_seed(self):
        """Test that same seed produces identical results."""
        x1 = simulate_one_factor_asset_returns(100, rho=0.5, rng=np.random.default_rng(123))
        x2 = simulate_one_factor_asset_returns(100, rho=0.5, rng=np.random.default_rng(123))
        
        assert np.allclose(x1, x2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        x1 = simulate_one_factor_asset_returns(100, rho=0.5, rng=np.random.default_rng(123))
        x2 = simulate_one_factor_asset_returns(100, rho=0.5, rng=np.random.default_rng(456))
        
        assert not np.allclose(x1, x2)


class TestSimulatePortfolioValuesT1:
    """Tests for simulate_portfolio_values_t1 function."""

    def test_portfolio_values_shape(self):
        """Test that returned array has correct shape."""
        portfolio = make_portfolio({"AAA": 1.0})
        v1 = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1000.0,
            n_issuers_per_rating=10,
            rho=0.5,
            N=100,
            seed=42,
        )
        assert v1.shape == (100,)

    def test_portfolio_values_dtype(self):
        """Test that portfolio values are float."""
        portfolio = make_portfolio({"AAA": 0.5, "AA": 0.5})
        v1 = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1000.0,
            n_issuers_per_rating=10,
            rho=0.5,
            N=50,
            seed=42,
        )
        assert v1.dtype == float or v1.dtype == np.float64

    def test_single_rating_deterministic_mean(self):
        """Test that single-rating portfolio has sensible mean value."""
        portfolio = make_portfolio({"AAA": 1.0})
        v1 = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1000.0,
            n_issuers_per_rating=100,
            rho=0.0,
            N=1000,
            seed=42,
        )
        
        # Mean should be close to bond value at t=1
        expected = 1000.0 * (BOND_VALUE_T1["AAA"] / 100.0)
        assert np.abs(np.mean(v1) - expected) < 0.1 * expected

    def test_multi_rating_portfolio(self):
        """Test multi-rating portfolio simulation."""
        portfolio = make_portfolio({"AAA": 0.4, "AA": 0.3, "A": 0.3})
        v1 = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=2000.0,
            n_issuers_per_rating=50,
            rho=0.3,
            N=100,
            seed=42,
        )
        
        assert v1.shape == (100,)
        assert np.all(np.isfinite(v1))

    def test_zero_n_raises_error(self):
        """Test that N=0 raises ValueError."""
        portfolio = make_portfolio({"AAA": 1.0})
        with pytest.raises(ValueError, match="N must be positive"):
            simulate_portfolio_values_t1(
                portfolio=portfolio,
                total_notional=1000.0,
                n_issuers_per_rating=10,
                rho=0.5,
                N=0,
                seed=42,
            )

    def test_negative_n_raises_error(self):
        """Test that negative N raises ValueError."""
        portfolio = make_portfolio({"AAA": 1.0})
        with pytest.raises(ValueError, match="N must be positive"):
            simulate_portfolio_values_t1(
                portfolio=portfolio,
                total_notional=1000.0,
                n_issuers_per_rating=10,
                rho=0.5,
                N=-10,
                seed=42,
            )

    def test_zero_n_issuers_raises_error(self):
        """Test that n_issuers_per_rating=0 raises ValueError."""
        portfolio = make_portfolio({"AAA": 1.0})
        with pytest.raises(ValueError, match="n_issuers_per_rating must be positive"):
            simulate_portfolio_values_t1(
                portfolio=portfolio,
                total_notional=1000.0,
                n_issuers_per_rating=0,
                rho=0.5,
                N=100,
                seed=42,
            )

    def test_negative_notional_raises_error(self):
        """Test that negative notional raises ValueError."""
        portfolio = make_portfolio({"AAA": 1.0})
        with pytest.raises(ValueError, match="total_notional must be positive"):
            simulate_portfolio_values_t1(
                portfolio=portfolio,
                total_notional=-1000.0,
                n_issuers_per_rating=10,
                rho=0.5,
                N=100,
                seed=42,
            )

    def test_invalid_rho_raises_error(self):
        """Test that rho outside [0,1] raises ValueError."""
        portfolio = make_portfolio({"AAA": 1.0})
        with pytest.raises(ValueError, match="rho must be in"):
            simulate_portfolio_values_t1(
                portfolio=portfolio,
                total_notional=1000.0,
                n_issuers_per_rating=10,
                rho=1.5,
                N=100,
                seed=42,
            )

    def test_invalid_rating_in_portfolio_raises_error(self):
        """Test that invalid ratings in portfolio raise ValueError."""
        # Manually create invalid portfolio (bypass make_portfolio validation)
        invalid_portfolio = {"D": 1.0}
        with pytest.raises(ValueError, match="not in start ratings"):
            simulate_portfolio_values_t1(
                portfolio=invalid_portfolio,
                total_notional=1000.0,
                n_issuers_per_rating=10,
                rho=0.5,
                N=100,
                seed=42,
            )

    def test_reproducibility_with_seed(self):
        """Test that same seed produces identical results."""
        portfolio = make_portfolio({"AAA": 0.5, "BB": 0.5})
        
        v1_a = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1500.0,
            n_issuers_per_rating=20,
            rho=0.4,
            N=100,
            seed=999,
        )
        
        v1_b = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1500.0,
            n_issuers_per_rating=20,
            rho=0.4,
            N=100,
            seed=999,
        )
        
        assert np.allclose(v1_a, v1_b)

    def test_all_positive_values(self):
        """Test that all simulated values are positive."""
        portfolio = make_portfolio({"AAA": 0.6, "AA": 0.4})
        v1 = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1000.0,
            n_issuers_per_rating=10,
            rho=0.5,
            N=200,
            seed=42,
        )
        
        assert np.all(v1 > 0)

    def test_finite_values(self):
        """Test that all simulated values are finite."""
        portfolio = make_portfolio({"AAA": 1.0})
        v1 = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1000.0,
            n_issuers_per_rating=10,
            rho=0.5,
            N=100,
            seed=42,
        )
        
        assert np.all(np.isfinite(v1))


class TestPortfolioValueDistribution:
    """Tests for statistical properties of portfolio value distributions."""

    def test_concentrated_vs_diversified(self):
        """Test that diversified portfolio has lower volatility."""
        portfolio = make_portfolio({"BBB": 1.0})
        
        v1_conc = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1000.0,
            n_issuers_per_rating=1,  # concentrated
            rho=0.5,
            N=500,
            seed=42,
        )
        
        v1_div = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1000.0,
            n_issuers_per_rating=100,  # diversified
            rho=0.5,
            N=500,
            seed=42,
        )
        
        # Diversified should have lower std (reduced idiosyncratic risk)
        assert np.std(v1_div) < np.std(v1_conc)

    def test_correlation_effect(self):
        """Test that higher correlation increases portfolio variance."""
        portfolio = make_portfolio({"A": 1.0})
        
        v1_low_rho = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1000.0,
            n_issuers_per_rating=50,
            rho=0.1,
            N=500,
            seed=42,
        )
        
        v1_high_rho = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1000.0,
            n_issuers_per_rating=50,
            rho=0.9,
            N=500,
            seed=42,
        )
        
        # Higher correlation => higher systematic risk => higher variance
        assert np.std(v1_high_rho) > np.std(v1_low_rho)

    def test_notional_scaling(self):
        """Test that portfolio value scales with notional."""
        portfolio = make_portfolio({"AA": 1.0})
        
        v1_small = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1000.0,
            n_issuers_per_rating=100,
            rho=0.5,
            N=200,
            seed=42,
        )
        
        v1_large = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=2000.0,
            n_issuers_per_rating=100,
            rho=0.5,
            N=200,
            seed=42,
        )
        
        # Means should scale proportionally
        ratio = np.mean(v1_large) / np.mean(v1_small)
        assert np.isclose(ratio, 2.0, rtol=0.05)

    def test_portfolio_mix_effect(self):
        """Test that lower-quality portfolio has lower expected value."""
        portfolio_high = make_portfolio({"AAA": 1.0})
        portfolio_low = make_portfolio({"CCC": 1.0})
        
        v1_high = simulate_portfolio_values_t1(
            portfolio=portfolio_high,
            total_notional=1000.0,
            n_issuers_per_rating=100,
            rho=0.0,  # no correlation for cleaner comparison
            N=500,
            seed=42,
        )
        
        v1_low = simulate_portfolio_values_t1(
            portfolio=portfolio_low,
            total_notional=1000.0,
            n_issuers_per_rating=100,
            rho=0.0,
            N=500,
            seed=42,
        )
        
        # High-quality portfolio should have higher expected value
        assert np.mean(v1_high) > np.mean(v1_low)


class TestPrecomputedThresholds:
    """Tests for using precomputed thresholds."""

    def test_with_explicit_thresholds(self):
        """Test that providing explicit thresholds works."""
        portfolio = make_portfolio({"BBB": 1.0})
        thresholds = precompute_thresholds()
        
        v1 = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1000.0,
            n_issuers_per_rating=10,
            rho=0.5,
            N=50,
            seed=42,
            thresholds=thresholds,
        )
        
        assert v1.shape == (50,)
        assert np.all(np.isfinite(v1))

    def test_thresholds_vs_none(self):
        """Test that explicit vs None thresholds produce same distribution."""
        portfolio = make_portfolio({"A": 1.0})
        thresholds = precompute_thresholds()
        
        v1_explicit = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1000.0,
            n_issuers_per_rating=50,
            rho=0.4,
            N=200,
            seed=123,
            thresholds=thresholds,
        )
        
        v1_computed = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1000.0,
            n_issuers_per_rating=50,
            rho=0.4,
            N=200,
            seed=123,
            thresholds=None,
        )
        
        # Should be identical
        assert np.allclose(v1_explicit, v1_computed)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_scenario(self):
        """Test simulation with N=1."""
        portfolio = make_portfolio({"AAA": 1.0})
        v1 = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1000.0,
            n_issuers_per_rating=10,
            rho=0.5,
            N=1,
            seed=42,
        )
        
        assert v1.shape == (1,)
        assert v1[0] > 0

    def test_large_n_scenarios(self):
        """Test simulation with large N."""
        portfolio = make_portfolio({"A": 1.0})
        v1 = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1000.0,
            n_issuers_per_rating=20,
            rho=0.3,
            N=5000,
            seed=42,
        )
        
        assert v1.shape == (5000,)
        assert np.all(np.isfinite(v1))

    def test_all_start_ratings_in_portfolio(self):
        """Test portfolio with all start ratings equally weighted."""
        weights = {r: 1.0 / len(START_RATINGS) for r in START_RATINGS}
        portfolio = make_portfolio(weights, normalize=True)
        
        v1 = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=1500.0,
            n_issuers_per_rating=10,
            rho=0.5,
            N=100,
            seed=42,
        )
        
        assert v1.shape == (100,)
        assert np.all(v1 > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
