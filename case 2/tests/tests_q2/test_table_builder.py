import pytest
import math
import pandas as pd
import numpy as np
from services.q2_exact.table_builder import (
    _validate_quotes,
    _integrated_hazard_piecewise,
    build_q2_results_table,
    build_q2_submission_table,
    _REQUIRED_QUOTE_COLS,
)
from services.q2_exact.stripping import StripResult, strip_forward_hazards_iterative
from services.instruments.curves import PiecewiseConstantHazardCurve


# ============================================================
# Quote validation tests
# ============================================================

class TestValidateQuotes:
    """Test CDS quotes validation."""

    def test_valid_quotes_dataframe(self):
        """Valid DataFrame should be returned sorted."""
        quotes = pd.DataFrame({
            "maturity_years": [5, 1, 3],  # Not sorted
            "cds_spread_bps": [120, 100, 110],
        })
        result = _validate_quotes(quotes)
        assert isinstance(result, pd.DataFrame)
        assert list(result["maturity_years"]) == [1, 3, 5]

    def test_returns_copy_not_original(self):
        """Should return a copy, not modify original."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5],
            "cds_spread_bps": [100, 110, 120],
        })
        result = _validate_quotes(quotes)
        result.loc[0, "maturity_years"] = 999
        assert quotes.loc[0, "maturity_years"] == 1

    def test_not_dataframe_raises(self):
        """Non-DataFrame input should raise TypeError."""
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            _validate_quotes([1, 2, 3])

    def test_missing_maturity_column_raises(self):
        """Missing maturity_years column should raise."""
        quotes = pd.DataFrame({
            "cds_spread_bps": [100, 110, 120],
        })
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_quotes(quotes)

    def test_missing_spread_column_raises(self):
        """Missing cds_spread_bps column should raise."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5],
        })
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_quotes(quotes)

    def test_empty_dataframe_raises(self):
        """Empty DataFrame should raise."""
        quotes = pd.DataFrame({
            "maturity_years": [],
            "cds_spread_bps": [],
        })
        with pytest.raises(ValueError, match="empty"):
            _validate_quotes(quotes)

    def test_non_increasing_maturities_raises(self):
        """Non-strictly-increasing maturities should raise."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 3, 5],
            "cds_spread_bps": [100, 110, 115, 120],
        })
        with pytest.raises(ValueError, match="strictly increasing"):
            _validate_quotes(quotes)

    def test_decreasing_maturities_sorted(self):
        """Decreasing maturities should be sorted automatically."""
        quotes = pd.DataFrame({
            "maturity_years": [5, 3, 1],
            "cds_spread_bps": [120, 110, 100],
        })
        result = _validate_quotes(quotes)
        assert list(result["maturity_years"]) == [1, 3, 5]
        assert list(result["cds_spread_bps"]) == [100, 110, 120]

    def test_extra_columns_ignored(self):
        """Extra columns should be ignored but preserved."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5],
            "cds_spread_bps": [100, 110, 120],
            "extra_col": [10, 20, 30],
        })
        result = _validate_quotes(quotes)
        assert "extra_col" in result.columns


# ============================================================
# Integrated hazard computation tests
# ============================================================

class TestIntegratedHazardPiecewise:
    """Test integrated hazard calculation."""

    def test_single_interval(self):
        """Single interval integral."""
        integ = _integrated_hazard_piecewise(
            tenor_boundaries=[0.0, 1.0],
            forward_hazards=[0.02],
            T=1.0,
        )
        assert integ == pytest.approx(0.02)

    def test_multiple_intervals_full(self):
        """Multiple intervals, evaluate at final maturity."""
        integ = _integrated_hazard_piecewise(
            tenor_boundaries=[0.0, 1.0, 3.0, 5.0],
            forward_hazards=[0.01, 0.03, 0.02],
            T=5.0,
        )
        # = 0.01*1 + 0.03*2 + 0.02*2 = 0.01 + 0.06 + 0.04 = 0.11
        assert integ == pytest.approx(0.11)

    def test_partial_integration(self):
        """Evaluate at time before final maturity."""
        integ = _integrated_hazard_piecewise(
            tenor_boundaries=[0.0, 1.0, 3.0, 5.0],
            forward_hazards=[0.01, 0.03, 0.02],
            T=2.0,
        )
        # = 0.01*1 + 0.03*1 = 0.04
        assert integ == pytest.approx(0.04)

    def test_at_boundary(self):
        """Evaluate exactly at a boundary."""
        integ = _integrated_hazard_piecewise(
            tenor_boundaries=[0.0, 1.0, 3.0],
            forward_hazards=[0.01, 0.03],
            T=3.0,
        )
        # = 0.01*1 + 0.03*2 = 0.07
        assert integ == pytest.approx(0.07)

    def test_case_2_standard(self):
        """Test with Case 2 standard boundaries."""
        integ = _integrated_hazard_piecewise(
            tenor_boundaries=[0.0, 1.0, 3.0, 5.0, 7.0, 10.0],
            forward_hazards=[0.01, 0.03, 0.02, 0.025, 0.015],
            T=10.0,
        )
        # = 0.01*1 + 0.03*2 + 0.02*2 + 0.025*2 + 0.015*3
        # = 0.01 + 0.06 + 0.04 + 0.05 + 0.045 = 0.205
        assert integ == pytest.approx(0.205)

    def test_tenor_boundaries_not_at_zero_raises(self):
        """tenor_boundaries must start at 0.0."""
        with pytest.raises(ValueError, match="tenor_boundaries must start at 0"):
            _integrated_hazard_piecewise(
                tenor_boundaries=[0.5, 1.0, 3.0],
                forward_hazards=[0.01, 0.03],
                T=3.0,
            )

    def test_hazard_length_mismatch_raises(self):
        """Hazard length must match intervals."""
        with pytest.raises(ValueError, match="forward_hazards length must match intervals"):
            _integrated_hazard_piecewise(
                tenor_boundaries=[0.0, 1.0, 3.0],
                forward_hazards=[0.01],  # Wrong length
                T=3.0,
            )

    def test_zero_time(self):
        """Integral at T=0 should be 0."""
        integ = _integrated_hazard_piecewise(
            tenor_boundaries=[0.0, 1.0, 3.0],
            forward_hazards=[0.01, 0.03],
            T=0.0,
        )
        assert integ == pytest.approx(0.0)


# ============================================================
# Q2 results table builder tests
# ============================================================

class TestBuildQ2ResultsTable:
    """Test Q2 results table building."""

    @pytest.fixture
    def standard_setup(self):
        """Standard setup with CDS quotes and strip result."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        strip_result = strip_forward_hazards_iterative(
            quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        return quotes, strip_result

    def test_returns_dataframe(self, standard_setup):
        """Should return a DataFrame."""
        quotes, strip_result = standard_setup
        result = build_q2_results_table(quotes, strip_result)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_required_columns(self, standard_setup):
        """Output should have all required columns."""
        quotes, strip_result = standard_setup
        result = build_q2_results_table(quotes, strip_result)
        
        required = ["maturity_years", "cds_spread_bps", "cds_spread",
                   "avg_hazard", "fwd_hazard", "fwd_default_prob",
                   "survival_T", "cum_default_prob"]
        for col in required:
            assert col in result.columns

    def test_output_length_matches_input(self, standard_setup):
        """Output rows should match input quotes."""
        quotes, strip_result = standard_setup
        result = build_q2_results_table(quotes, strip_result)
        assert len(result) == len(quotes)

    def test_maturities_match(self, standard_setup):
        """Maturities in output should match input."""
        quotes, strip_result = standard_setup
        result = build_q2_results_table(quotes, strip_result)
        assert list(result["maturity_years"]) == list(quotes["maturity_years"])

    def test_spreads_conversion(self, standard_setup):
        """Spreads should be converted from bps to decimal."""
        quotes, strip_result = standard_setup
        result = build_q2_results_table(quotes, strip_result)
        
        for i, row in result.iterrows():
            expected_spread = row["cds_spread_bps"] / 10_000.0
            assert row["cds_spread"] == pytest.approx(expected_spread)

    def test_survival_decreasing(self, standard_setup):
        """Survival probabilities should be strictly decreasing."""
        quotes, strip_result = standard_setup
        result = build_q2_results_table(quotes, strip_result)
        
        survivals = result["survival_T"].tolist()
        for i in range(len(survivals) - 1):
            assert survivals[i] > survivals[i + 1]

    def test_cum_default_prob_increasing(self, standard_setup):
        """Cumulative default prob should be strictly increasing."""
        quotes, strip_result = standard_setup
        result = build_q2_results_table(quotes, strip_result)
        
        cum_probs = result["cum_default_prob"].tolist()
        for i in range(len(cum_probs) - 1):
            assert cum_probs[i] < cum_probs[i + 1]

    def test_cum_default_prob_is_one_minus_survival(self, standard_setup):
        """cum_default_prob should equal 1 - survival_T."""
        quotes, strip_result = standard_setup
        result = build_q2_results_table(quotes, strip_result)
        
        for idx, row in result.iterrows():
            expected = 1.0 - row["survival_T"]
            assert row["cum_default_prob"] == pytest.approx(expected)

    def test_fwd_default_prob_sum_to_cum(self, standard_setup):
        """Forward default probs should sum to cumulative."""
        quotes, strip_result = standard_setup
        result = build_q2_results_table(quotes, strip_result)
        
        cum_at_last = result["cum_default_prob"].iloc[-1]
        sum_fwd = result["fwd_default_prob"].sum()
        assert sum_fwd == pytest.approx(cum_at_last)

    def test_avg_hazard_positive(self, standard_setup):
        """Average hazard should be positive."""
        quotes, strip_result = standard_setup
        result = build_q2_results_table(quotes, strip_result)
        
        assert (result["avg_hazard"] > 0).all()

    def test_fwd_hazard_positive(self, standard_setup):
        """Forward hazard should be positive."""
        quotes, strip_result = standard_setup
        result = build_q2_results_table(quotes, strip_result)
        
        assert (result["fwd_hazard"] > 0).all()

    def test_invalid_quotes_raises(self, standard_setup):
        """Invalid quotes should raise."""
        _, strip_result = standard_setup
        bad_quotes = pd.DataFrame({
            "maturity_years": [1, 3, 3],  # Duplicate
            "cds_spread_bps": [100, 110, 120],
        })
        with pytest.raises(ValueError):
            build_q2_results_table(bad_quotes, strip_result)

    def test_mismatched_hazard_count_raises(self, standard_setup):
        """Wrong number of forward hazards should raise."""
        quotes, strip_result = standard_setup
        
        # Create a new strip result with wrong number of hazards
        bad_result = StripResult(
            forward_hazards=[0.01, 0.03],  # Only 2 instead of 5
            hazard_curve=strip_result.hazard_curve,
            diagnostics=strip_result.diagnostics,
        )
        
        with pytest.raises(ValueError, match="Expected .* forward hazards"):
            build_q2_results_table(quotes, bad_result)

    def test_avg_hazard_relationship_to_survival(self, standard_setup):
        """Average hazard should relate to survival via S(T) = exp(-avg_hazard * T)."""
        quotes, strip_result = standard_setup
        result = build_q2_results_table(quotes, strip_result)
        
        for idx, row in result.iterrows():
            T = row["maturity_years"]
            avg_lam = row["avg_hazard"]
            S_T = row["survival_T"]
            
            expected_S = math.exp(-avg_lam * T)
            assert S_T == pytest.approx(expected_S, rel=1e-10)


# ============================================================
# Q2 submission table builder tests
# ============================================================

class TestBuildQ2SubmissionTable:
    """Test Q2 submission table building."""

    @pytest.fixture
    def sample_full_table(self):
        """Create a sample full results table."""
        return pd.DataFrame({
            "maturity_years": [1.0, 3.0, 5.0, 7.0, 10.0],
            "cds_spread_bps": [100.0, 110.0, 120.0, 120.0, 125.0],
            "avg_hazard": [0.025, 0.027, 0.029, 0.030, 0.031],
            "fwd_hazard": [0.025, 0.03, 0.035, 0.03, 0.034],
            "fwd_default_prob": [0.024, 0.055, 0.060, 0.050, 0.079],
            "survival_T": [0.975, 0.921, 0.861, 0.811, 0.732],
            "cum_default_prob": [0.025, 0.079, 0.139, 0.189, 0.268],
        })

    def test_returns_dataframe(self, sample_full_table):
        """Should return a DataFrame."""
        result = build_q2_submission_table(sample_full_table)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_required_columns(self, sample_full_table):
        """Output should have submission-required columns."""
        result = build_q2_submission_table(sample_full_table)
        
        required = ["Maturity", "CDS Rate (in bps)",
                   "(Average) Hazard Rate",
                   "Forward Hazard Rate (between T_{i-1} and T_i)",
                   "Forward Default Probability (between T_{i-1} and T_i)"]
        for col in required:
            assert col in result.columns

    def test_output_length_matches_input(self, sample_full_table):
        """Output rows should match input."""
        result = build_q2_submission_table(sample_full_table)
        assert len(result) == len(sample_full_table)

    def test_maturity_labels_format(self, sample_full_table):
        """Maturity should be formatted as 'NY' strings."""
        result = build_q2_submission_table(sample_full_table)
        
        expected = ["1Y", "3Y", "5Y", "7Y", "10Y"]
        assert list(result["Maturity"]) == expected

    def test_cds_spread_renamed(self, sample_full_table):
        """cds_spread_bps should be renamed to 'CDS Rate (in bps)'."""
        result = build_q2_submission_table(sample_full_table)
        
        assert "CDS Rate (in bps)" in result.columns
        assert "cds_spread_bps" not in result.columns
        assert list(result["CDS Rate (in bps)"]) == [100.0, 110.0, 120.0, 120.0, 125.0]

    def test_avg_hazard_renamed(self, sample_full_table):
        """avg_hazard should be renamed."""
        result = build_q2_submission_table(sample_full_table)
        
        assert "(Average) Hazard Rate" in result.columns
        assert "avg_hazard" not in result.columns

    def test_fwd_hazard_renamed(self, sample_full_table):
        """fwd_hazard should be renamed."""
        result = build_q2_submission_table(sample_full_table)
        
        assert "Forward Hazard Rate (between T_{i-1} and T_i)" in result.columns
        assert "fwd_hazard" not in result.columns

    def test_fwd_default_prob_renamed(self, sample_full_table):
        """fwd_default_prob should be renamed."""
        result = build_q2_submission_table(sample_full_table)
        
        expected_name = "Forward Default Probability (between T_{i-1} and T_i)"
        assert expected_name in result.columns
        assert "fwd_default_prob" not in result.columns

    def test_extra_columns_removed(self, sample_full_table):
        """Extra columns like survival_T, cum_default_prob should be removed."""
        result = build_q2_submission_table(sample_full_table)
        
        assert "survival_T" not in result.columns
        assert "cum_default_prob" not in result.columns

    def test_missing_required_columns_raises(self):
        """Missing required columns should raise."""
        bad_table = pd.DataFrame({
            "maturity_years": [1.0, 3.0],
            # Missing other required columns
        })
        with pytest.raises(ValueError, match="missing columns needed for submission"):
            build_q2_submission_table(bad_table)

    def test_maturity_with_decimal(self):
        """Non-integer maturity should format correctly."""
        table = pd.DataFrame({
            "maturity_years": [1.5, 2.25],
            "cds_spread_bps": [100.0, 110.0],
            "avg_hazard": [0.025, 0.027],
            "fwd_hazard": [0.025, 0.03],
            "fwd_default_prob": [0.024, 0.055],
        })
        result = build_q2_submission_table(table)
        
        # Should preserve non-integer maturities
        assert result["Maturity"].iloc[0] == "1.5Y"
        assert result["Maturity"].iloc[1] == "2.25Y"

    def test_maturity_rounding_tolerance(self):
        """Values very close to integer should round."""
        table = pd.DataFrame({
            "maturity_years": [1.0 + 1e-13, 3.0 - 1e-13],
            "cds_spread_bps": [100.0, 110.0],
            "avg_hazard": [0.025, 0.027],
            "fwd_hazard": [0.025, 0.03],
            "fwd_default_prob": [0.024, 0.055],
        })
        result = build_q2_submission_table(table)
        
        # Should round very close values to integer
        assert result["Maturity"].iloc[0] == "1Y"
        assert result["Maturity"].iloc[1] == "3Y"


# ============================================================
# Integration tests
# ============================================================

class TestTableBuilderIntegration:
    """Integration tests for the full table building workflow."""

    def test_full_workflow_results_to_submission(self):
        """Test building results table then submission table."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        strip_result = strip_forward_hazards_iterative(
            quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        results = build_q2_results_table(quotes, strip_result)
        submission = build_q2_submission_table(results)
        
        # Verify both tables are valid
        assert len(results) == 5
        assert len(submission) == 5
        assert "Maturity" in submission.columns

    def test_consistency_across_different_spreads(self):
        """Results should be consistent for different spread curves."""
        for spreads in [[100, 110, 120, 120, 125],
                       [50, 60, 70, 80, 90],
                       [150, 160, 170, 180, 190]]:
            quotes = pd.DataFrame({
                "maturity_years": [1, 3, 5, 7, 10],
                "cds_spread_bps": spreads,
            })
            
            strip_result = strip_forward_hazards_iterative(
                quotes,
                r=0.03,
                lgd=0.40,
                verbose=False,
            )
            
            results = build_q2_results_table(quotes, strip_result)
            submission = build_q2_submission_table(results)
            
            # All should produce valid outputs
            assert len(results) == 5
            assert len(submission) == 5
            assert (results["survival_T"] > 0).all()

    def test_mathematical_properties_preserved(self):
        """Mathematical properties should hold across transformation."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        strip_result = strip_forward_hazards_iterative(
            quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        results = build_q2_results_table(quotes, strip_result)
        submission = build_q2_submission_table(results)
        
        # Verify survival is in (0,1) and decreasing
        results_survival = results["survival_T"].tolist()
        assert all(0 < s < 1 for s in results_survival)
        assert all(results_survival[i] > results_survival[i+1] 
                  for i in range(len(results_survival)-1))
