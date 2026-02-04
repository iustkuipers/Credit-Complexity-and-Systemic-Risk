import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from services.q3_audit.audit_7y import audit_7y, Audit7YResult
from services.q2_exact.stripping import strip_forward_hazards_iterative, StripResult
from services.instruments.curves import PiecewiseConstantHazardCurve


# ============================================================
# Audit7YResult dataclass tests
# ============================================================

class TestAudit7YResult:
    """Test Audit7YResult container."""

    def test_creation(self):
        """Audit7YResult should be created with all required fields."""
        result = Audit7YResult(
            maturity_years=7.0,
            cds_spread_bps=120.0,
            cds_spread=0.012,
            premium_leg_pv=0.1,
            protection_leg_pv=0.1,
            npv=1e-10,
            abs_npv=1e-10,
            tolerance=1e-6,
            passed=True,
        )
        
        assert result.maturity_years == 7.0
        assert result.cds_spread_bps == 120.0
        assert result.passed is True

    def test_frozen(self):
        """Audit7YResult should be frozen (immutable)."""
        result = Audit7YResult(
            maturity_years=7.0,
            cds_spread_bps=120.0,
            cds_spread=0.012,
            premium_leg_pv=0.1,
            protection_leg_pv=0.1,
            npv=1e-10,
            abs_npv=1e-10,
            tolerance=1e-6,
            passed=True,
        )
        
        with pytest.raises(AttributeError):
            result.passed = False


# ============================================================
# Basic audit tests
# ============================================================

class TestAudit7YBasic:
    """Test basic audit_7y functionality."""

    @pytest.fixture
    def standard_quotes(self):
        """Standard CDS quotes including 7Y."""
        return pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })

    def test_returns_audit_result(self, standard_quotes):
        """audit_7y should return Audit7YResult."""
        result = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            verbose=False,
        )
        assert isinstance(result, Audit7YResult)

    def test_maturity_is_7y(self, standard_quotes):
        """Result maturity should be 7.0 years."""
        result = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            verbose=False,
        )
        assert result.maturity_years == 7.0

    def test_spread_extracted_correctly(self, standard_quotes):
        """Spread should be extracted from 7Y row."""
        result = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            verbose=False,
        )
        assert result.cds_spread_bps == 120.0
        assert result.cds_spread == pytest.approx(0.012)

    def test_legs_pv_positive(self, standard_quotes):
        """Premium and protection leg PVs should be positive."""
        result = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            verbose=False,
        )
        assert result.premium_leg_pv > 0
        assert result.protection_leg_pv > 0

    def test_npv_close_to_zero(self, standard_quotes):
        """NPV should be close to zero (within tolerance)."""
        result = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            tolerance=1e-6,
            verbose=False,
        )
        assert abs(result.npv) <= result.tolerance
        assert result.passed is True

    def test_abs_npv_is_absolute_value(self, standard_quotes):
        """abs_npv should equal absolute value of npv."""
        result = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            verbose=False,
        )
        assert result.abs_npv == pytest.approx(abs(result.npv))


# ============================================================
# Strip result integration tests
# ============================================================

class TestAudit7YWithStripResult:
    """Test audit_7y with provided StripResult."""

    @pytest.fixture
    def standard_quotes(self):
        """Standard CDS quotes."""
        return pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })

    def test_with_provided_strip_result(self, standard_quotes):
        """Should work with provided StripResult."""
        # First strip the hazards
        strip_result = strip_forward_hazards_iterative(
            standard_quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        # Then audit with provided result
        result = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            strip_result=strip_result,
            verbose=False,
        )
        
        assert isinstance(result, Audit7YResult)
        assert result.passed is True

    def test_with_and_without_strip_result_same(self, standard_quotes):
        """Results should be identical with or without provided StripResult."""
        # With internal stripping
        result1 = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            strip_result=None,
            verbose=False,
        )
        
        # With provided stripping
        strip_result = strip_forward_hazards_iterative(
            standard_quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        result2 = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            strip_result=strip_result,
            verbose=False,
        )
        
        # Results should be numerically identical
        assert result1.npv == pytest.approx(result2.npv)
        assert result1.premium_leg_pv == pytest.approx(result2.premium_leg_pv)
        assert result1.protection_leg_pv == pytest.approx(result2.protection_leg_pv)


# ============================================================
# Parameter variation tests
# ============================================================

class TestAudit7YParameterVariation:
    """Test audit_7y with different parameter combinations."""

    @pytest.fixture
    def standard_quotes(self):
        """Standard CDS quotes."""
        return pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })

    def test_different_risk_free_rates(self, standard_quotes):
        """Should work with different risk-free rates."""
        for r in [0.01, 0.03, 0.05]:
            result = audit_7y(
                cds_quotes=standard_quotes,
                risk_free_rate=r,
                lgd=0.40,
                verbose=False,
            )
            assert isinstance(result, Audit7YResult)
            assert result.maturity_years == 7.0

    def test_different_lgd_values(self, standard_quotes):
        """Should work with different LGD values."""
        for lgd in [0.20, 0.40, 0.60, 0.80]:
            result = audit_7y(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=lgd,
                verbose=False,
            )
            assert isinstance(result, Audit7YResult)
            assert abs(result.npv) <= result.tolerance

    def test_different_premium_frequencies(self, standard_quotes):
        """Should work with different premium frequencies."""
        for freq in [1, 2, 4, 12]:
            result = audit_7y(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                premium_frequency=freq,
                verbose=False,
            )
            assert isinstance(result, Audit7YResult)

    def test_different_tolerances(self, standard_quotes):
        """Tolerance affects pass/fail status."""
        result_tight = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            tolerance=1e-12,
            verbose=False,
        )
        
        result_loose = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            tolerance=1e-2,
            verbose=False,
        )
        
        # Tight tolerance is more likely to fail than loose
        assert result_loose.passed  # Loose tolerance should pass
        # (tight might fail or pass depending on numerical precision)

    def test_verbose_mode_true(self, standard_quotes, capsys):
        """Verbose mode should print progress."""
        result = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            verbose=True,
        )
        
        captured = capsys.readouterr()
        assert "[Q3]" in captured.out

    def test_verbose_mode_false(self, standard_quotes, capsys):
        """Non-verbose mode should not print."""
        result = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        captured = capsys.readouterr()
        assert "[Q3]" not in captured.out


# ============================================================
# Error handling tests
# ============================================================

class TestAudit7YErrorHandling:
    """Test error handling in audit_7y."""

    def test_missing_7y_quote_raises(self):
        """Should raise error if 7Y quote is missing."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 10],  # No 7Y
            "cds_spread_bps": [100, 110, 120, 125],
        })
        
        with pytest.raises(ValueError, match="Could not find 7Y"):
            audit_7y(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                verbose=False,
            )

    def test_missing_maturity_column_raises(self):
        """Should raise error for missing maturity_years column."""
        quotes = pd.DataFrame({
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        with pytest.raises(ValueError, match="missing columns"):
            audit_7y(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                verbose=False,
            )

    def test_missing_spread_column_raises(self):
        """Should raise error for missing cds_spread_bps column."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
        })
        
        with pytest.raises(ValueError, match="missing columns"):
            audit_7y(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                verbose=False,
            )

    def test_invalid_lgd_zero_raises(self):
        """Should raise error for zero LGD."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        with pytest.raises((ValueError, ZeroDivisionError)):
            audit_7y(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=0.0,
                verbose=False,
            )

    def test_invalid_lgd_negative_raises(self):
        """Should raise error for negative LGD."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        with pytest.raises(ValueError):
            audit_7y(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=-0.40,
                verbose=False,
            )


# ============================================================
# Quote scenarios tests
# ============================================================

class TestAudit7YQuoteScenarios:
    """Test audit_7y with different quote scenarios."""

    def test_7y_at_start(self):
        """Should work with 7Y as first quote."""
        quotes = pd.DataFrame({
            "maturity_years": [7, 10],
            "cds_spread_bps": [120, 125],
        })
        
        result = audit_7y(
            cds_quotes=quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        assert result.maturity_years == 7.0
        assert result.cds_spread_bps == 120.0

    def test_7y_at_end(self):
        """Should work with 7Y as last quote."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7],
            "cds_spread_bps": [100, 110, 120, 120],
        })
        
        result = audit_7y(
            cds_quotes=quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        assert result.maturity_years == 7.0

    def test_7y_middle(self):
        """Should work with 7Y in the middle."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 7, 10],
            "cds_spread_bps": [100, 120, 125],
        })
        
        result = audit_7y(
            cds_quotes=quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        assert result.maturity_years == 7.0

    def test_exact_7y_float_match(self):
        """Should match 7.0 exactly in floating point."""
        quotes = pd.DataFrame({
            "maturity_years": [7.0, 10.0],
            "cds_spread_bps": [120.0, 125.0],
        })
        
        result = audit_7y(
            cds_quotes=quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        assert result.maturity_years == 7.0

    def test_case_2_standard_quotes(self):
        """Test with standard Case 2 quotes."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        result = audit_7y(
            cds_quotes=quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            premium_frequency=4,
            verbose=False,
        )
        
        # Should pass validation
        assert result.passed is True
        assert result.abs_npv <= result.tolerance


# ============================================================
# Numerical properties tests
# ============================================================

class TestAudit7YNumericalProperties:
    """Test numerical properties of audit results."""

    @pytest.fixture
    def standard_quotes(self):
        """Standard CDS quotes."""
        return pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })

    def test_npv_equals_premium_minus_protection(self, standard_quotes):
        """NPV should equal premium_leg - protection_leg."""
        result = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        computed_npv = result.premium_leg_pv - result.protection_leg_pv
        assert result.npv == pytest.approx(computed_npv)

    def test_abs_npv_non_negative(self, standard_quotes):
        """abs_npv should be non-negative."""
        result = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        assert result.abs_npv >= 0

    def test_tolerance_non_negative(self, standard_quotes):
        """Tolerance should be non-negative."""
        result = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            tolerance=1e-6,
            verbose=False,
        )
        
        assert result.tolerance >= 0

    def test_passed_iff_abs_npv_within_tolerance(self, standard_quotes):
        """Passed should be True iff abs_npv <= tolerance."""
        result = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            tolerance=1e-6,
            verbose=False,
        )
        
        expected_passed = result.abs_npv <= result.tolerance
        assert result.passed == expected_passed

    def test_consistency_across_runs(self, standard_quotes):
        """Repeated audits should give identical results."""
        result1 = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        result2 = audit_7y(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        assert result1.npv == pytest.approx(result2.npv)
        assert result1.premium_leg_pv == pytest.approx(result2.premium_leg_pv)
        assert result1.protection_leg_pv == pytest.approx(result2.protection_leg_pv)
        assert result1.passed == result2.passed


# ============================================================
# Integration tests
# ============================================================

class TestAudit7YIntegration:
    """Integration tests for audit_7y."""

    def test_audit_after_q2_stripping(self):
        """Should audit successfully after Q2 stripping."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        # Simulate Q2
        strip_result = strip_forward_hazards_iterative(
            quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        # Q3: Audit with Q2 results
        result = audit_7y(
            cds_quotes=quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            strip_result=strip_result,
            verbose=False,
        )
        
        assert result.passed is True

    def test_workflow_q2_q3_consistency(self):
        """Q3 results should be consistent with Q2 calculations."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        # Q2: Strip hazards
        strip_result = strip_forward_hazards_iterative(
            quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        # Q3: Audit 7Y (should pass since it's the NPV ~ 0 tenor)
        audit_result = audit_7y(
            cds_quotes=quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            strip_result=strip_result,
            verbose=False,
        )
        
        # For the 7Y tenor that was stripped, NPV should be ~ 0
        assert audit_result.passed is True
        assert audit_result.abs_npv <= audit_result.tolerance
