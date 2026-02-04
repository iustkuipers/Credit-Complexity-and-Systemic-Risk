# tests/tests_q4/test_orchestrator.py

import pandas as pd
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from services.q4_stress.orchestrator import (
    Q4Outputs,
    run_q4,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def standard_quotes():
    """Standard CDS quotes for testing."""
    return pd.DataFrame({
        "maturity_years": [1.0, 3.0, 5.0, 7.0, 10.0],
        "cds_spread_bps": [100.0, 110.0, 120.0, 120.0, 125.0],
    })


@pytest.fixture
def narrow_quotes():
    """Narrow maturity range quotes."""
    return pd.DataFrame({
        "maturity_years": [1.0, 2.0],
        "cds_spread_bps": [100.0, 105.0],
    })


@pytest.fixture
def wide_quotes():
    """Wide maturity range quotes."""
    return pd.DataFrame({
        "maturity_years": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        "cds_spread_bps": [80.0, 100.0, 110.0, 120.0, 125.0, 130.0],
    })


# ============================================================================
# TestRunQ4Basic
# ============================================================================

class TestRunQ4Basic:
    """Basic execution tests for run_q4()"""

    def test_basic_execution(self, standard_quotes):
        """Should execute without error."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=False,
            save=False,
        )
        assert isinstance(result, Q4Outputs)

    def test_outputs_structure(self, standard_quotes):
        """Should return Q4Outputs with required fields."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=False,
            save=False,
        )
        assert result.baseline is not None
        assert result.stressed is not None
        assert result.deltas is not None
        assert isinstance(result.deltas, dict)

    def test_baseline_scenarios_differ(self, standard_quotes):
        """Baseline and stressed scenarios should produce different results."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            baseline_r=0.03,
            stressed_r=0.10,
            verbose=False,
            save=False,
        )
        # Q2 results should differ (rate-dependent)
        assert not result.baseline.q2_table_full.equals(result.stressed.q2_table_full)

    def test_q1_results_identical_across_scenarios(self, standard_quotes):
        """Q1 results should be identical (rate-independent)."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            baseline_r=0.03,
            stressed_r=0.10,
            verbose=False,
            save=False,
        )
        # Q1 should be identical
        pd.testing.assert_frame_equal(
            result.baseline.q1_table,
            result.stressed.q1_table,
            check_exact=False,
            rtol=1e-10
        )

    def test_default_rate_values(self, standard_quotes):
        """Should use default rate values if not provided."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=False,
            save=False,
        )
        # Default baseline_r = 0.03, stressed_r = 0.10
        assert result.baseline.risk_free_rate == 0.03
        assert result.stressed.risk_free_rate == 0.10

    def test_custom_rate_values(self, standard_quotes):
        """Should accept custom baseline and stressed rates."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            baseline_r=0.01,
            stressed_r=0.05,
            verbose=False,
            save=False,
        )
        assert result.baseline.risk_free_rate == 0.01
        assert result.stressed.risk_free_rate == 0.05

    def test_premium_frequency_parameter(self, standard_quotes):
        """Should accept custom premium frequency."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            premium_frequency=2,
            verbose=False,
            save=False,
        )
        assert isinstance(result, Q4Outputs)

    def test_verbose_false_no_print(self, standard_quotes, capsys):
        """Should not print when verbose=False."""
        run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=False,
            save=False,
        )
        captured = capsys.readouterr()
        assert "[Q4]" not in captured.out

    def test_verbose_true_prints(self, standard_quotes, capsys):
        """Should print progress when verbose=True."""
        run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=True,
            save=False,
        )
        captured = capsys.readouterr()
        assert "[Q4]" in captured.out
        assert "baseline" in captured.out.lower()
        assert "stressed" in captured.out.lower()


# ============================================================================
# TestQ4Outputs
# ============================================================================

class TestQ4Outputs:
    """Tests for Q4Outputs dataclass"""

    def test_q4_outputs_creation(self, standard_quotes):
        """Should create Q4Outputs instance."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=False,
            save=False,
        )
        assert isinstance(result, Q4Outputs)
        assert result.baseline is not None
        assert result.stressed is not None

    def test_q4_outputs_frozen(self, standard_quotes):
        """Q4Outputs should be frozen (immutable)."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=False,
            save=False,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            result.baseline = None

    def test_output_dir_none_when_not_saved(self, standard_quotes):
        """output_dir should be None when save=False."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=False,
            save=False,
        )
        assert result.output_dir is None


# ============================================================================
# TestDeltaTables
# ============================================================================

class TestDeltaTables:
    """Tests for delta tables in Q4Outputs"""

    def test_deltas_structure(self, standard_quotes):
        """Delta tables dict should have expected keys."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=False,
            save=False,
        )
        assert "delta_q1" in result.deltas
        assert "delta_q2" in result.deltas
        assert "delta_gap" in result.deltas

    def test_delta_q1_near_zero(self, standard_quotes):
        """Q1 deltas should be near zero (rate-independent)."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            baseline_r=0.03,
            stressed_r=0.10,
            verbose=False,
            save=False,
        )
        delta_q1 = result.deltas["delta_q1"]
        assert (delta_q1["avg_hazard_delta"].abs() < 1e-10).all()

    def test_delta_q2_nonzero(self, standard_quotes):
        """Q2 deltas should be nonzero (rate-dependent)."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            baseline_r=0.03,
            stressed_r=0.10,
            verbose=False,
            save=False,
        )
        delta_q2 = result.deltas["delta_q2"]
        # At least some deltas should be nonzero
        assert (delta_q2["avg_hazard_delta"].abs() > 1e-10).any()

    def test_delta_gap_has_widening(self, standard_quotes):
        """Delta gap should have widening column."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            baseline_r=0.03,
            stressed_r=0.10,
            verbose=False,
            save=False,
        )
        delta_gap = result.deltas["delta_gap"]
        assert "gap_abs_widening" in delta_gap.columns

    def test_delta_row_counts_match_input(self, standard_quotes):
        """Delta tables should have same number of rows as input."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=False,
            save=False,
        )
        n = len(standard_quotes)
        assert len(result.deltas["delta_q1"]) == n
        assert len(result.deltas["delta_q2"]) == n
        assert len(result.deltas["delta_gap"]) == n


# ============================================================================
# TestScenarioComparison
# ============================================================================

class TestScenarioComparison:
    """Tests comparing baseline and stressed scenarios"""

    def test_baseline_rate_lower_than_stressed(self, standard_quotes):
        """Baseline rate should be lower than stressed (by default)."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=False,
            save=False,
        )
        assert result.baseline.risk_free_rate < result.stressed.risk_free_rate

    def test_same_quotes_used_in_both(self, standard_quotes):
        """Both scenarios should use same CDS quotes."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=False,
            save=False,
        )
        baseline_maturities = sorted(result.baseline.q1_table["maturity_years"].values)
        stressed_maturities = sorted(result.stressed.q1_table["maturity_years"].values)
        assert baseline_maturities == stressed_maturities

    def test_both_scenarios_have_complete_outputs(self, standard_quotes):
        """Both scenarios should have q1, q2, and comparison tables."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=False,
            save=False,
        )
        for scenario in [result.baseline, result.stressed]:
            assert isinstance(scenario.q1_table, pd.DataFrame)
            assert isinstance(scenario.q2_table_full, pd.DataFrame)
            assert isinstance(scenario.compare_table, pd.DataFrame)
            assert len(scenario.q1_table) > 0
            assert len(scenario.q2_table_full) > 0
            assert len(scenario.compare_table) > 0


# ============================================================================
# TestParameterVariation
# ============================================================================

class TestParameterVariation:
    """Tests with different parameter combinations"""

    def test_extreme_lgd_low(self, standard_quotes):
        """Should handle very low LGD."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.05,
            verbose=False,
            save=False,
        )
        assert isinstance(result, Q4Outputs)

    def test_extreme_lgd_high(self, standard_quotes):
        """Should handle very high LGD."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.95,
            verbose=False,
            save=False,
        )
        assert isinstance(result, Q4Outputs)

    def test_extreme_rates(self, standard_quotes):
        """Should handle very low and very high rates."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            baseline_r=0.001,
            stressed_r=0.20,
            verbose=False,
            save=False,
        )
        assert isinstance(result, Q4Outputs)

    def test_narrow_rate_difference(self, standard_quotes):
        """Should handle small difference between baseline and stressed."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            baseline_r=0.03,
            stressed_r=0.031,
            verbose=False,
            save=False,
        )
        assert isinstance(result, Q4Outputs)

    def test_different_quote_sets(self):
        """Should work with different quote sets."""
        q1 = pd.DataFrame({
            "maturity_years": [1.0],
            "cds_spread_bps": [100.0],
        })
        q2 = pd.DataFrame({
            "maturity_years": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cds_spread_bps": [100.0, 105.0, 110.0, 115.0, 120.0],
        })
        
        for quotes in [q1, q2]:
            result = run_q4(
                cds_quotes=quotes,
                lgd=0.4,
                verbose=False,
                save=False,
            )
            assert isinstance(result, Q4Outputs)


# ============================================================================
# TestFileOperations
# ============================================================================

class TestFileOperations:
    """Tests for file I/O operations"""

    def test_save_false_no_output_dir(self, standard_quotes):
        """When save=False, output_dir should be None."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=False,
            save=False,
        )
        assert result.output_dir is None

    def test_save_true_creates_output_dir(self, standard_quotes):
        """When save=True, should create output directory."""
        with TemporaryDirectory() as tmpdir:
            result = run_q4(
                cds_quotes=standard_quotes,
                lgd=0.4,
                output_root=tmpdir,
                verbose=False,
                save=True,
            )
            assert result.output_dir is not None
            assert result.output_dir.exists()

    def test_save_creates_files(self, standard_quotes):
        """When save=True, should create output files."""
        with TemporaryDirectory() as tmpdir:
            result = run_q4(
                cds_quotes=standard_quotes,
                lgd=0.4,
                output_root=tmpdir,
                verbose=False,
                save=True,
            )
            # Check that output directory was created
            assert result.output_dir is not None
            assert result.output_dir.exists()

    def test_custom_output_root(self, standard_quotes):
        """Should use provided output_root path."""
        with TemporaryDirectory() as tmpdir:
            result = run_q4(
                cds_quotes=standard_quotes,
                lgd=0.4,
                output_root=tmpdir,
                verbose=False,
                save=True,
            )
            # Output directory should be under tmpdir
            assert str(result.output_dir).startswith(tmpdir)


# ============================================================================
# TestErrorHandling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling"""

    def test_empty_quotes_raises(self):
        """Should raise error for empty quotes."""
        empty = pd.DataFrame({
            "maturity_years": [],
            "cds_spread_bps": [],
        })
        with pytest.raises(ValueError):
            run_q4(
                cds_quotes=empty,
                lgd=0.4,
                verbose=False,
                save=False,
            )

    def test_missing_maturity_column_raises(self, standard_quotes):
        """Should raise error if maturity_years column missing."""
        bad = standard_quotes.drop(columns=["maturity_years"])
        with pytest.raises(ValueError):
            run_q4(
                cds_quotes=bad,
                lgd=0.4,
                verbose=False,
                save=False,
            )

    def test_missing_spread_column_raises(self, standard_quotes):
        """Should raise error if cds_spread_bps column missing."""
        bad = standard_quotes.drop(columns=["cds_spread_bps"])
        with pytest.raises(ValueError):
            run_q4(
                cds_quotes=bad,
                lgd=0.4,
                verbose=False,
                save=False,
            )


# ============================================================================
# TestOutputConsistency
# ============================================================================

class TestOutputConsistency:
    """Tests for consistency of outputs"""

    def test_scenario_tables_have_matching_lengths(self, standard_quotes):
        """Q1, Q2, and compare tables should have same length in each scenario."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=False,
            save=False,
        )
        n = len(standard_quotes)
        
        for scenario in [result.baseline, result.stressed]:
            assert len(scenario.q1_table) == n
            assert len(scenario.q2_table_full) == n
            assert len(scenario.compare_table) == n

    def test_baseline_and_stressed_same_shape(self, standard_quotes):
        """Baseline and stressed should have same table shapes."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=False,
            save=False,
        )
        assert result.baseline.q1_table.shape == result.stressed.q1_table.shape
        assert result.baseline.q2_table_full.shape == result.stressed.q2_table_full.shape
        assert result.baseline.compare_table.shape == result.stressed.compare_table.shape

    def test_deltas_have_consistent_columns(self, standard_quotes):
        """Delta tables should have expected columns."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            verbose=False,
            save=False,
        )
        
        # Check delta_q1
        delta_q1 = result.deltas["delta_q1"]
        assert "maturity_years" in delta_q1.columns
        assert "avg_hazard_delta" in delta_q1.columns
        
        # Check delta_q2
        delta_q2 = result.deltas["delta_q2"]
        assert "maturity_years" in delta_q2.columns
        assert "avg_hazard_delta" in delta_q2.columns
        assert "survival_T_delta" in delta_q2.columns
        
        # Check delta_gap
        delta_gap = result.deltas["delta_gap"]
        assert "maturity_years" in delta_gap.columns
        assert "gap_abs_widening" in delta_gap.columns


# ============================================================================
# TestIntegration
# ============================================================================

class TestIntegration:
    """Integration tests"""

    def test_full_q4_workflow(self, standard_quotes):
        """Should support complete Q4 workflow."""
        result = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            baseline_r=0.03,
            stressed_r=0.10,
            premium_frequency=4,
            verbose=False,
            save=False,
        )
        
        # All components present
        assert result.baseline is not None
        assert result.stressed is not None
        assert len(result.deltas) == 3
        
        # Can access nested data
        assert len(result.baseline.q1_table) > 0
        assert len(result.stressed.q2_table_full) > 0
        assert len(result.deltas["delta_gap"]) > 0

    def test_with_file_operations(self, standard_quotes):
        """Should support complete workflow with file operations."""
        with TemporaryDirectory() as tmpdir:
            result = run_q4(
                cds_quotes=standard_quotes,
                lgd=0.4,
                output_root=tmpdir,
                verbose=False,
                save=True,
            )
            
            assert result.output_dir is not None
            assert result.output_dir.exists()
            assert isinstance(result, Q4Outputs)

    def test_multiple_runs_independent(self, standard_quotes):
        """Multiple runs should be independent."""
        r1 = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            baseline_r=0.02,
            stressed_r=0.08,
            verbose=False,
            save=False,
        )
        r2 = run_q4(
            cds_quotes=standard_quotes,
            lgd=0.4,
            baseline_r=0.03,
            stressed_r=0.10,
            verbose=False,
            save=False,
        )
        
        # Different rate scenarios should produce different deltas
        assert not r1.deltas["delta_q2"].equals(r2.deltas["delta_q2"])

    def test_different_quote_sets_consistent(self):
        """Different quote sets should produce consistent outputs."""
        q1 = pd.DataFrame({
            "maturity_years": [1.0, 5.0, 10.0],
            "cds_spread_bps": [100.0, 120.0, 125.0],
        })
        q2 = pd.DataFrame({
            "maturity_years": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
            "cds_spread_bps": [100.0, 105.0, 110.0, 115.0, 120.0, 125.0],
        })
        
        for quotes in [q1, q2]:
            result = run_q4(
                cds_quotes=quotes,
                lgd=0.4,
                verbose=False,
                save=False,
            )
            
            # Should have same number of rows as input
            n = len(quotes)
            assert len(result.baseline.q1_table) == n
            assert len(result.deltas["delta_q1"]) == n
