import pytest
import math
import pandas as pd
import numpy as np
from services.q2_exact.stripping import (
    StrippingDiagnosticsRow,
    StripResult,
    _to_decimal_spread,
    _build_segments_for_tenor,
    _solve_root,
    strip_forward_hazards_iterative,
)
from services.instruments.curves import PiecewiseConstantHazardCurve


# ============================================================
# Helper function tests
# ============================================================

class TestToDecimalSpread:
    """Test basis points to decimal conversion."""

    def test_100_bps_to_decimal(self):
        """100 bps should be 0.01."""
        assert _to_decimal_spread(100) == pytest.approx(0.01)

    def test_zero_bps(self):
        """0 bps should be 0.0."""
        assert _to_decimal_spread(0) == pytest.approx(0.0)

    def test_1_bps(self):
        """1 bps should be 0.0001."""
        assert _to_decimal_spread(1) == pytest.approx(0.0001)

    def test_10000_bps(self):
        """10000 bps should be 1.0."""
        assert _to_decimal_spread(10000) == pytest.approx(1.0)

    def test_negative_bps(self):
        """Negative bps should convert correctly."""
        assert _to_decimal_spread(-100) == pytest.approx(-0.01)


# ============================================================
# Build segments tests
# ============================================================

class TestBuildSegmentsForTenor:
    """Test segment building from hazard rates."""

    def test_single_interval_single_hazard(self):
        """Single interval with one hazard rate."""
        segments = _build_segments_for_tenor(
            tenor_boundaries=[0.0, 1.0],
            forward_hazards=[0.02],
            horizon=1.0,
        )
        assert len(segments) == 1
        assert segments[0] == (0.0, 1.0, 0.02)

    def test_multiple_intervals(self):
        """Multiple intervals with multiple hazard rates."""
        segments = _build_segments_for_tenor(
            tenor_boundaries=[0.0, 1.0, 3.0, 5.0],
            forward_hazards=[0.01, 0.03, 0.02],
            horizon=5.0,
        )
        assert len(segments) == 3
        assert segments[0] == (0.0, 1.0, 0.01)
        assert segments[1] == (1.0, 3.0, 0.03)
        assert segments[2] == (3.0, 5.0, 0.02)

    def test_horizon_shorter_than_last_boundary(self):
        """Horizon shorter than last boundary should truncate."""
        segments = _build_segments_for_tenor(
            tenor_boundaries=[0.0, 1.0, 3.0, 5.0],
            forward_hazards=[0.01, 0.03, 0.02],
            horizon=2.5,
        )
        assert len(segments) == 2
        assert segments[0] == (0.0, 1.0, 0.01)
        assert segments[1] == (1.0, 2.5, 0.03)

    def test_tenor_boundaries_not_starting_at_zero_raises(self):
        """tenor_boundaries must start at 0."""
        with pytest.raises(ValueError, match="tenor_boundaries must start at 0"):
            _build_segments_for_tenor(
                tenor_boundaries=[0.5, 1.0, 3.0],
                forward_hazards=[0.01, 0.03],
                horizon=3.0,
            )

    def test_horizon_not_covered_raises(self):
        """Horizon not covered by boundaries should raise."""
        with pytest.raises(ValueError, match="Failed to build segments covering horizon"):
            _build_segments_for_tenor(
                tenor_boundaries=[0.0, 1.0, 3.0],
                forward_hazards=[0.01, 0.03],
                horizon=5.0,
            )

    def test_standard_case_2_boundaries(self):
        """Test with Case 2 standard boundaries [0, 1, 3, 5, 7, 10]."""
        segments = _build_segments_for_tenor(
            tenor_boundaries=[0.0, 1.0, 3.0, 5.0, 7.0, 10.0],
            forward_hazards=[0.01, 0.03, 0.02, 0.025, 0.015],
            horizon=10.0,
        )
        assert len(segments) == 5
        assert segments[-1][1] == pytest.approx(10.0)


# ============================================================
# Root solve tests
# ============================================================

class TestSolveRoot:
    """Test robust root finding."""

    def test_solve_linear_root(self):
        """Solve simple linear f(x) = 2*x - 1 = 0."""
        def f(x):
            return 2 * x - 1
        
        root = _solve_root(f, low=0.1, high=1.0)
        assert root == pytest.approx(0.5, rel=1e-6)

    def test_solve_quadratic_root(self):
        """Solve f(x) = x^2 - 4 = 0 for x > 0."""
        def f(x):
            return x * x - 4
        
        root = _solve_root(f, low=0.1, high=5.0)
        assert root == pytest.approx(2.0, rel=1e-6)

    def test_solve_exponential_root(self):
        """Solve f(x) = exp(x) - 2 = 0."""
        def f(x):
            return math.exp(x) - 2
        
        root = _solve_root(f, low=0.1, high=2.0)
        assert root == pytest.approx(math.log(2), rel=1e-6)

    def test_bracket_expansion(self):
        """Test automatic bracket expansion."""
        def f(x):
            return x - 0.5
        
        # Initial bracket [0.01, 0.1] does not bracket the root 0.5
        # Should expand automatically
        root = _solve_root(f, low=0.01, high=0.1, expand_factor=2.0)
        assert root == pytest.approx(0.5, rel=1e-6)

    def test_zero_bracket_raises(self):
        """Zero or negative bracket should raise."""
        def f(x):
            return x - 0.5
        
        with pytest.raises(ValueError, match="bracket must be positive"):
            _solve_root(f, low=-0.1, high=1.0)

    def test_no_sign_change_raises(self):
        """No sign change in bracket and max reached should raise."""
        def f(x):
            return x * x + 1  # Always positive
        
        with pytest.raises(ValueError, match="Root not bracketed"):
            _solve_root(f, low=0.01, high=2.0, max_high=10.0)


# ============================================================
# Data structure tests
# ============================================================

class TestStrippingDiagnosticsRow:
    """Test StrippingDiagnosticsRow dataclass."""

    def test_creation(self):
        """Should create row with all fields."""
        row = StrippingDiagnosticsRow(
            maturity_years=1.0,
            cds_spread_bps=100.0,
            cds_spread=0.01,
            solved_forward_hazard=0.02,
            premium_leg_pv=50.0,
            protection_leg_pv=45.0,
            npv=5.0,
        )
        assert row.maturity_years == 1.0
        assert row.cds_spread_bps == 100.0
        assert row.solved_forward_hazard == 0.02

    def test_frozen(self):
        """StrippingDiagnosticsRow should be immutable."""
        row = StrippingDiagnosticsRow(
            maturity_years=1.0,
            cds_spread_bps=100.0,
            cds_spread=0.01,
            solved_forward_hazard=0.02,
            premium_leg_pv=50.0,
            protection_leg_pv=45.0,
            npv=5.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            row.maturity_years = 2.0


class TestStripResult:
    """Test StripResult dataclass."""

    def test_creation(self):
        """Should create StripResult with hazard curve and diagnostics."""
        hazard_curve = PiecewiseConstantHazardCurve(segments=((0.0, 1.0, 0.02),))
        diag_df = pd.DataFrame({
            "maturity_years": [1.0],
            "solved_forward_hazard": [0.02],
        })
        result = StripResult(
            forward_hazards=[0.02],
            hazard_curve=hazard_curve,
            diagnostics=diag_df,
        )
        assert len(result.forward_hazards) == 1
        assert result.hazard_curve is hazard_curve
        assert len(result.diagnostics) == 1

    def test_frozen(self):
        """StripResult should be immutable."""
        hazard_curve = PiecewiseConstantHazardCurve(segments=((0.0, 1.0, 0.02),))
        diag_df = pd.DataFrame()
        result = StripResult(
            forward_hazards=[0.02],
            hazard_curve=hazard_curve,
            diagnostics=diag_df,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.forward_hazards = [0.03]


# ============================================================
# Main stripping algorithm tests
# ============================================================

class TestStripForwardHazardsIterative:
    """Test the main iterative stripping function."""

    @pytest.fixture
    def standard_cds_quotes(self):
        """Standard Case 2 CDS quotes."""
        return pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })

    def test_returns_strip_result(self, standard_cds_quotes):
        """Should return a StripResult object."""
        result = strip_forward_hazards_iterative(
            standard_cds_quotes,
            r=0.03,
            lgd=0.40,
            payments_per_year=4,
            verbose=False,
        )
        assert isinstance(result, StripResult)

    def test_forward_hazards_list(self, standard_cds_quotes):
        """Forward hazards should be a list of positive floats."""
        result = strip_forward_hazards_iterative(
            standard_cds_quotes,
            r=0.03,
            lgd=0.40,
            payments_per_year=4,
            verbose=False,
        )
        assert len(result.forward_hazards) == 5
        assert all(isinstance(x, float) for x in result.forward_hazards)
        assert all(x > 0 for x in result.forward_hazards)

    def test_hazard_curve_valid(self, standard_cds_quotes):
        """Hazard curve should be PiecewiseConstantHazardCurve."""
        result = strip_forward_hazards_iterative(
            standard_cds_quotes,
            r=0.03,
            lgd=0.40,
            payments_per_year=4,
            verbose=False,
        )
        assert isinstance(result.hazard_curve, PiecewiseConstantHazardCurve)

    def test_diagnostics_dataframe(self, standard_cds_quotes):
        """Diagnostics should be a DataFrame with expected columns."""
        result = strip_forward_hazards_iterative(
            standard_cds_quotes,
            r=0.03,
            lgd=0.40,
            payments_per_year=4,
            verbose=False,
        )
        assert isinstance(result.diagnostics, pd.DataFrame)
        assert len(result.diagnostics) == 5
        
        expected_cols = ["maturity_years", "cds_spread_bps", "cds_spread",
                        "solved_forward_hazard", "premium_leg_pv", 
                        "protection_leg_pv", "npv"]
        for col in expected_cols:
            assert col in result.diagnostics.columns

    def test_missing_required_columns_raises(self):
        """Missing required columns should raise ValueError."""
        bad_quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5],
            # missing cds_spread_bps
        })
        with pytest.raises(ValueError, match="missing columns"):
            strip_forward_hazards_iterative(
                bad_quotes,
                r=0.03,
                lgd=0.40,
                verbose=False,
            )

    def test_non_increasing_maturities_raises(self):
        """Non-increasing maturities should raise."""
        bad_quotes = pd.DataFrame({
            "maturity_years": [1, 3, 3, 5],
            "cds_spread_bps": [100, 110, 115, 120],
        })
        with pytest.raises(ValueError, match="strictly increasing"):
            strip_forward_hazards_iterative(
                bad_quotes,
                r=0.03,
                lgd=0.40,
                verbose=False,
            )

    def test_different_risk_free_rates(self, standard_cds_quotes):
        """Stripping should work with different risk-free rates."""
        result_low = strip_forward_hazards_iterative(
            standard_cds_quotes,
            r=0.01,
            lgd=0.40,
            verbose=False,
        )
        result_high = strip_forward_hazards_iterative(
            standard_cds_quotes,
            r=0.05,
            lgd=0.40,
            verbose=False,
        )
        
        # Different rates should produce different hazards
        assert result_low.forward_hazards != result_high.forward_hazards

    def test_different_lgd_values(self, standard_cds_quotes):
        """Stripping should work with different LGD values."""
        result_low = strip_forward_hazards_iterative(
            standard_cds_quotes,
            r=0.03,
            lgd=0.30,
            verbose=False,
        )
        result_high = strip_forward_hazards_iterative(
            standard_cds_quotes,
            r=0.03,
            lgd=0.50,
            verbose=False,
        )
        
        # Different LGDs should produce different hazards
        assert result_low.forward_hazards != result_high.forward_hazards

    def test_verbose_mode_does_not_fail(self, standard_cds_quotes):
        """Verbose mode should complete without errors."""
        result = strip_forward_hazards_iterative(
            standard_cds_quotes,
            r=0.03,
            lgd=0.40,
            verbose=True,
        )
        assert isinstance(result, StripResult)

    def test_annual_payments(self, standard_cds_quotes):
        """Should work with annual payments."""
        result = strip_forward_hazards_iterative(
            standard_cds_quotes,
            r=0.03,
            lgd=0.40,
            payments_per_year=1,
            verbose=False,
        )
        assert len(result.forward_hazards) == 5

    def test_semi_annual_payments(self, standard_cds_quotes):
        """Should work with semi-annual payments."""
        result = strip_forward_hazards_iterative(
            standard_cds_quotes,
            r=0.03,
            lgd=0.40,
            payments_per_year=2,
            verbose=False,
        )
        assert len(result.forward_hazards) == 5

    def test_monthly_payments(self, standard_cds_quotes):
        """Should work with monthly payments."""
        result = strip_forward_hazards_iterative(
            standard_cds_quotes,
            r=0.03,
            lgd=0.40,
            payments_per_year=12,
            verbose=False,
        )
        assert len(result.forward_hazards) == 5

    def test_hazard_curve_can_evaluate_survival(self, standard_cds_quotes):
        """Hazard curve should allow survival probability calculations."""
        result = strip_forward_hazards_iterative(
            standard_cds_quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        # Should be able to compute survival at various times
        for t in [0.5, 1.0, 3.0, 5.0, 7.0, 10.0]:
            survival = result.hazard_curve.survival(t)
            assert 0 <= survival <= 1

    def test_survival_decreasing(self, standard_cds_quotes):
        """Survival probabilities should be decreasing with time."""
        result = strip_forward_hazards_iterative(
            standard_cds_quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        times = [1.0, 3.0, 5.0, 7.0, 10.0]
        survivals = [result.hazard_curve.survival(t) for t in times]
        
        for i in range(len(survivals) - 1):
            assert survivals[i] > survivals[i + 1]

    def test_npv_close_to_zero_after_solve(self, standard_cds_quotes):
        """NPV should be very close to zero after solving for each tenor."""
        result = strip_forward_hazards_iterative(
            standard_cds_quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        # NPVs should be very small (close to zero, within numerical tolerance)
        for npv in result.diagnostics["npv"]:
            assert abs(npv) < 1e-6


# ============================================================
# Integration tests
# ============================================================

class TestStrippingIntegration:
    """Integration tests for the stripping workflow."""

    def test_stripping_with_flat_spreads(self):
        """Test stripping with flat CDS curve."""
        flat_quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 100, 100, 100, 100],  # Flat
        })
        
        result = strip_forward_hazards_iterative(
            flat_quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        assert len(result.forward_hazards) == 5
        assert all(x > 0 for x in result.forward_hazards)

    def test_stripping_with_increasing_spreads(self):
        """Test stripping with increasing CDS curve."""
        increasing_quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [50, 100, 150, 200, 250],  # Increasing
        })
        
        result = strip_forward_hazards_iterative(
            increasing_quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        assert len(result.forward_hazards) == 5
        # Later hazards should generally be higher for increasing curve
        assert all(x > 0 for x in result.forward_hazards)

    def test_stripping_with_very_small_spreads(self):
        """Test stripping with very small spreads."""
        small_quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [1, 2, 3, 4, 5],  # Very small
        })
        
        result = strip_forward_hazards_iterative(
            small_quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        # Should still solve successfully with small spreads
        assert len(result.forward_hazards) == 5
        assert all(x > 0 for x in result.forward_hazards)

    def test_stripping_with_moderate_high_spreads(self):
        """Test stripping with moderately high spreads."""
        high_quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [200, 300, 350, 375, 400],  # Moderately high
        })
        
        result = strip_forward_hazards_iterative(
            high_quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        # Should still solve with high spreads
        assert len(result.forward_hazards) == 5
        assert all(x > 0 for x in result.forward_hazards)

    def test_stripping_consistency_same_input(self):
        """Same input should produce consistent results."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        result1 = strip_forward_hazards_iterative(
            quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        result2 = strip_forward_hazards_iterative(
            quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        # Results should be identical for same input
        for h1, h2 in zip(result1.forward_hazards, result2.forward_hazards):
            assert h1 == pytest.approx(h2, rel=1e-10)
