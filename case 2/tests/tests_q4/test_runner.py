# tests/tests_q4/test_runner.py

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from services.q4_stress.runner import (
    ScenarioOutputs,
    _validate_quotes,
    _merge_q1_q2,
    run_single_scenario,
    build_delta_tables,
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
def standard_q1_table():
    """Sample Q1 simple model output."""
    return pd.DataFrame({
        "maturity_years": [1.0, 3.0, 5.0, 7.0, 10.0],
        "cds_spread_bps": [100.0, 110.0, 120.0, 120.0, 125.0],
        "cds_spread": [0.01, 0.011, 0.012, 0.012, 0.0125],
        "avg_hazard": [0.025, 0.0275, 0.03, 0.03, 0.03125],
        "fwd_hazard": [0.025, 0.02875, 0.03375, 0.03, 0.034167],
        "fwd_default_prob": [0.0246, 0.0545, 0.0601, 0.0501, 0.0790],
        "survival_T": [0.9753, 0.9208, 0.8607, 0.8106, 0.7316],
        "cum_default_prob": [0.0247, 0.0792, 0.1393, 0.1894, 0.2684],
    })


@pytest.fixture
def standard_q2_table():
    """Sample Q2 exact model output."""
    return pd.DataFrame({
        "maturity_years": [1.0, 3.0, 5.0, 7.0, 10.0],
        "cds_spread_bps": [100.0, 110.0, 120.0, 120.0, 125.0],
        "cds_spread": [0.01, 0.011, 0.012, 0.012, 0.0125],
        "avg_hazard": [0.02491, 0.02747, 0.03018, 0.03010, 0.03160],
        "fwd_hazard": [0.02491, 0.02875, 0.03424, 0.02989, 0.03512],
        "fwd_default_prob": [0.02460, 0.05451, 0.06095, 0.04990, 0.08101],
        "survival_T": [0.97540, 0.92089, 0.85994, 0.81004, 0.72903],
        "cum_default_prob": [0.02460, 0.07911, 0.14006, 0.18996, 0.27097],
    })


# ============================================================================
# TestValidateQuotes
# ============================================================================

class TestValidateQuotes:
    """Tests for _validate_quotes()"""

    def test_valid_quotes(self, standard_quotes):
        """Should pass validation for correctly formatted quotes."""
        result = _validate_quotes(standard_quotes)
        assert isinstance(result, pd.DataFrame)
        assert "maturity_years" in result.columns
        assert "cds_spread_bps" in result.columns

    def test_quotes_sorted_by_maturity(self, standard_quotes):
        """Should sort by maturity_years."""
        unsorted = standard_quotes.iloc[[4, 0, 2, 1, 3]]  # Scramble order
        result = _validate_quotes(unsorted)
        expected_maturities = [1.0, 3.0, 5.0, 7.0, 10.0]
        assert result["maturity_years"].tolist() == expected_maturities

    def test_quotes_index_reset(self, standard_quotes):
        """Should reset index to 0, 1, 2, ..."""
        unsorted = standard_quotes.iloc[[4, 0, 2, 1, 3]]
        result = _validate_quotes(unsorted)
        assert result.index.tolist() == [0, 1, 2, 3, 4]

    def test_missing_maturity_column(self, standard_quotes):
        """Should raise ValueError if maturity_years missing."""
        bad = standard_quotes.drop(columns=["maturity_years"])
        with pytest.raises(ValueError, match="missing columns"):
            _validate_quotes(bad)

    def test_missing_spread_column(self, standard_quotes):
        """Should raise ValueError if cds_spread_bps missing."""
        bad = standard_quotes.drop(columns=["cds_spread_bps"])
        with pytest.raises(ValueError, match="missing columns"):
            _validate_quotes(bad)

    def test_empty_quotes(self):
        """Should raise ValueError for empty DataFrame."""
        empty = pd.DataFrame({
            "maturity_years": [],
            "cds_spread_bps": [],
        })
        with pytest.raises(ValueError, match="empty"):
            _validate_quotes(empty)

    def test_missing_multiple_columns(self, standard_quotes):
        """Should report all missing columns."""
        bad = standard_quotes.drop(columns=["maturity_years", "cds_spread_bps"])
        with pytest.raises(ValueError, match="missing columns"):
            _validate_quotes(bad)

    def test_validates_copy_not_original(self, standard_quotes):
        """Should return a copy, not modify original."""
        original_id = id(standard_quotes)
        result = _validate_quotes(standard_quotes)
        assert id(result) != original_id


# ============================================================================
# TestMergeQ1Q2
# ============================================================================

class TestMergeQ1Q2:
    """Tests for _merge_q1_q2()"""

    def test_basic_merge(self, standard_q1_table, standard_q2_table):
        """Should merge Q1 and Q2 tables on maturity and spread."""
        result = _merge_q1_q2(standard_q1_table, standard_q2_table)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_column_renaming(self, standard_q1_table, standard_q2_table):
        """Should rename columns to distinguish Q1 and Q2 versions."""
        result = _merge_q1_q2(standard_q1_table, standard_q2_table)
        expected_cols = [
            "avg_hazard_q1_simple",
            "avg_hazard_q2_exact",
            "fwd_hazard_q1_simple",
            "fwd_hazard_q2_exact",
            "fwd_default_prob_q1_simple",
            "fwd_default_prob_q2_exact",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_gap_columns_created(self, standard_q1_table, standard_q2_table):
        """Should create gap columns (Q2 minus Q1)."""
        result = _merge_q1_q2(standard_q1_table, standard_q2_table)
        assert "avg_hazard_gap_q2_minus_q1" in result.columns
        assert "avg_hazard_gap_abs" in result.columns

    def test_gap_calculations(self, standard_q1_table, standard_q2_table):
        """Gap should equal Q2 avg_hazard minus Q1 avg_hazard."""
        result = _merge_q1_q2(standard_q1_table, standard_q2_table)
        q1_hazards = standard_q1_table["avg_hazard"].values
        q2_hazards = standard_q2_table["avg_hazard"].values
        expected_gap = q2_hazards - q1_hazards
        np.testing.assert_array_almost_equal(
            result["avg_hazard_gap_q2_minus_q1"].values, expected_gap
        )

    def test_gap_abs_is_absolute(self, standard_q1_table, standard_q2_table):
        """Gap abs should be absolute value of gap."""
        result = _merge_q1_q2(standard_q1_table, standard_q2_table)
        assert (result["avg_hazard_gap_abs"] >= 0).all()
        np.testing.assert_array_almost_equal(
            result["avg_hazard_gap_abs"].values,
            result["avg_hazard_gap_q2_minus_q1"].abs().values
        )

    def test_missing_avg_hazard_q1(self, standard_q2_table):
        """Should raise ValueError if Q1 table missing avg_hazard."""
        bad_q1 = pd.DataFrame({
            "maturity_years": [1.0, 3.0],
            "cds_spread_bps": [100.0, 110.0],
            "fwd_hazard": [0.025, 0.0275],
        })
        with pytest.raises(ValueError, match="missing 'avg_hazard'"):
            _merge_q1_q2(bad_q1, standard_q2_table)

    def test_missing_avg_hazard_q2(self, standard_q1_table):
        """Should raise ValueError if Q2 table missing avg_hazard."""
        bad_q2 = pd.DataFrame({
            "maturity_years": [1.0, 3.0],
            "cds_spread_bps": [100.0, 110.0],
            "fwd_hazard": [0.02491, 0.02875],
        })
        with pytest.raises(ValueError, match="missing 'avg_hazard'"):
            _merge_q1_q2(standard_q1_table, bad_q2)

    def test_missing_join_key_maturity(self, standard_q1_table, standard_q2_table):
        """Should raise ValueError if join key maturity_years missing."""
        bad_q1 = standard_q1_table.drop(columns=["maturity_years"])
        with pytest.raises(ValueError, match="Missing join key"):
            _merge_q1_q2(bad_q1, standard_q2_table)

    def test_missing_join_key_spread(self, standard_q1_table, standard_q2_table):
        """Should raise ValueError if join key cds_spread_bps missing."""
        bad_q2 = standard_q2_table.drop(columns=["cds_spread_bps"])
        with pytest.raises(ValueError, match="Missing join key"):
            _merge_q1_q2(standard_q1_table, bad_q2)

    def test_column_order(self, standard_q1_table, standard_q2_table):
        """Should have consistent column ordering."""
        result = _merge_q1_q2(standard_q1_table, standard_q2_table)
        col_order = [
            "maturity_years",
            "cds_spread_bps",
            "avg_hazard_q1_simple",
            "avg_hazard_q2_exact",
            "avg_hazard_gap_q2_minus_q1",
            "avg_hazard_gap_abs",
        ]
        for i, col in enumerate(col_order):
            assert result.columns[i] == col

    def test_inner_join_behavior(self, standard_q1_table):
        """Should use inner join (matching maturities only)."""
        q2_partial = pd.DataFrame({
            "maturity_years": [1.0, 3.0],  # Missing 5, 7, 10
            "cds_spread_bps": [100.0, 110.0],
            "avg_hazard": [0.02491, 0.02747],
            "fwd_hazard": [0.02491, 0.02875],
            "fwd_default_prob": [0.02460, 0.05451],
            "survival_T": [0.97540, 0.92089],
            "cum_default_prob": [0.02460, 0.07911],
        })
        result = _merge_q1_q2(standard_q1_table, q2_partial)
        assert len(result) == 2  # Only rows in both tables


# ============================================================================
# TestRunSingleScenario
# ============================================================================

class TestRunSingleScenario:
    """Tests for run_single_scenario()"""

    def test_basic_execution(self, standard_quotes):
        """Should execute without error."""
        result = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        assert isinstance(result, ScenarioOutputs)

    def test_outputs_structure(self, standard_quotes):
        """Should return ScenarioOutputs with required fields."""
        result = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        assert result.risk_free_rate == 0.03
        assert isinstance(result.q1_table, pd.DataFrame)
        assert isinstance(result.q2_table_full, pd.DataFrame)
        assert isinstance(result.compare_table, pd.DataFrame)

    def test_q1_table_has_required_columns(self, standard_quotes):
        """Q1 table should have required columns."""
        result = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        required = ["maturity_years", "avg_hazard", "survival_T"]
        for col in required:
            assert col in result.q1_table.columns

    def test_q2_table_has_required_columns(self, standard_quotes):
        """Q2 table should have required columns."""
        result = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        required = ["maturity_years", "avg_hazard", "survival_T"]
        for col in required:
            assert col in result.q2_table_full.columns

    def test_compare_table_has_gap_columns(self, standard_quotes):
        """Compare table should have gap columns."""
        result = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        assert "avg_hazard_gap_q2_minus_q1" in result.compare_table.columns
        assert "avg_hazard_gap_abs" in result.compare_table.columns

    def test_different_rates_produce_different_q2_results(self, standard_quotes):
        """Different risk-free rates should affect Q2 model."""
        r1 = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.01,
            verbose=False,
        )
        r2 = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.05,
            verbose=False,
        )
        # Q2 results should differ
        assert not r1.q2_table_full.equals(r2.q2_table_full)

    def test_q1_results_independent_of_rate(self, standard_quotes):
        """Q1 model should produce identical results regardless of rate."""
        r1 = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.01,
            verbose=False,
        )
        r2 = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.05,
            verbose=False,
        )
        # Q1 results should be identical
        assert_frame_equal(r1.q1_table, r2.q1_table, check_exact=False, rtol=1e-10)

    def test_premium_frequency_parameter(self, standard_quotes):
        """Should accept premium_frequency parameter."""
        result = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            premium_frequency=2,  # Semi-annual instead of quarterly
            verbose=False,
        )
        assert isinstance(result, ScenarioOutputs)

    def test_verbose_false_no_print(self, standard_quotes, capsys):
        """Should not print when verbose=False."""
        run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        captured = capsys.readouterr()
        assert "[Q4]" not in captured.out

    def test_verbose_true_prints(self, standard_quotes, capsys):
        """Should print progress when verbose=True."""
        run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=True,
        )
        captured = capsys.readouterr()
        assert "[Q4]" in captured.out

    def test_invalid_quotes_raises(self, standard_quotes):
        """Should raise error for invalid quotes."""
        bad = standard_quotes.drop(columns=["maturity_years"])
        with pytest.raises(ValueError):
            run_single_scenario(
                cds_quotes=bad,
                lgd=0.4,
                risk_free_rate=0.03,
                verbose=False,
            )

    def test_row_counts_match(self, standard_quotes):
        """All output tables should have same row count."""
        result = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        assert len(result.q1_table) == len(standard_quotes)
        assert len(result.q2_table_full) == len(standard_quotes)
        assert len(result.compare_table) == len(standard_quotes)

    def test_maturities_preserved(self, standard_quotes):
        """Output maturities should match input quotes."""
        result = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        input_maturities = sorted(standard_quotes["maturity_years"].values)
        output_maturities = sorted(result.q1_table["maturity_years"].values)
        np.testing.assert_array_almost_equal(input_maturities, output_maturities)

    def test_extreme_lgd_values(self, standard_quotes):
        """Should handle extreme LGD values."""
        # LGD = 0.1
        r1 = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.1,
            risk_free_rate=0.03,
            verbose=False,
        )
        assert isinstance(r1, ScenarioOutputs)
        
        # LGD = 0.95
        r2 = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.95,
            risk_free_rate=0.03,
            verbose=False,
        )
        assert isinstance(r2, ScenarioOutputs)

    def test_extreme_rate_values(self, standard_quotes):
        """Should handle extreme risk-free rates."""
        # r = 0.001
        r1 = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.001,
            verbose=False,
        )
        assert isinstance(r1, ScenarioOutputs)
        
        # r = 0.10
        r2 = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.10,
            verbose=False,
        )
        assert isinstance(r2, ScenarioOutputs)


# ============================================================================
# TestBuildDeltaTables
# ============================================================================

class TestBuildDeltaTables:
    """Tests for build_delta_tables()"""

    def test_basic_execution(self, standard_quotes):
        """Should execute without error."""
        baseline = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        stressed = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.05,
            verbose=False,
        )
        result = build_delta_tables(baseline=baseline, stressed=stressed)
        assert isinstance(result, dict)

    def test_returns_three_tables(self, standard_quotes):
        """Should return dict with delta_q1, delta_q2, delta_gap."""
        baseline = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        stressed = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.05,
            verbose=False,
        )
        result = build_delta_tables(baseline=baseline, stressed=stressed)
        assert "delta_q1" in result
        assert "delta_q2" in result
        assert "delta_gap" in result

    def test_delta_q1_structure(self, standard_quotes):
        """delta_q1 should have expected columns."""
        baseline = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        stressed = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.05,
            verbose=False,
        )
        result = build_delta_tables(baseline=baseline, stressed=stressed)
        delta_q1 = result["delta_q1"]
        assert "maturity_years" in delta_q1.columns
        assert "avg_hazard_delta" in delta_q1.columns
        assert "simple_changed_flag" in delta_q1.columns

    def test_delta_q1_should_be_near_zero(self, standard_quotes):
        """Q1 deltas should be ~0 (model is rate-independent)."""
        baseline = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        stressed = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.05,
            verbose=False,
        )
        result = build_delta_tables(baseline=baseline, stressed=stressed)
        delta_q1 = result["delta_q1"]
        # All deltas should be near zero (within tolerance)
        assert (delta_q1["avg_hazard_delta"].abs() < 1e-10).all()

    def test_delta_q2_structure(self, standard_quotes):
        """delta_q2 should have expected columns."""
        baseline = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        stressed = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.05,
            verbose=False,
        )
        result = build_delta_tables(baseline=baseline, stressed=stressed)
        delta_q2 = result["delta_q2"]
        assert "maturity_years" in delta_q2.columns
        assert "avg_hazard_delta" in delta_q2.columns
        assert "survival_T_delta" in delta_q2.columns
        assert "cum_default_prob_delta" in delta_q2.columns

    def test_delta_q2_should_be_nonzero(self, standard_quotes):
        """Q2 deltas should be nonzero (model depends on rate)."""
        baseline = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        stressed = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.05,
            verbose=False,
        )
        result = build_delta_tables(baseline=baseline, stressed=stressed)
        delta_q2 = result["delta_q2"]
        # At least some deltas should be nonzero
        assert (delta_q2["avg_hazard_delta"].abs() > 1e-10).any()

    def test_delta_gap_structure(self, standard_quotes):
        """delta_gap should have expected columns."""
        baseline = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        stressed = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.05,
            verbose=False,
        )
        result = build_delta_tables(baseline=baseline, stressed=stressed)
        delta_gap = result["delta_gap"]
        assert "maturity_years" in delta_gap.columns
        assert "gap_q2_minus_q1_baseline" in delta_gap.columns
        assert "gap_q2_minus_q1_stressed" in delta_gap.columns
        assert "gap_abs_baseline" in delta_gap.columns
        assert "gap_abs_stressed" in delta_gap.columns
        assert "gap_abs_widening" in delta_gap.columns

    def test_gap_widening_calculation(self, standard_quotes):
        """gap_abs_widening should equal stressed minus baseline."""
        baseline = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        stressed = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.05,
            verbose=False,
        )
        result = build_delta_tables(baseline=baseline, stressed=stressed)
        delta_gap = result["delta_gap"]
        expected_widening = delta_gap["gap_abs_stressed"] - delta_gap["gap_abs_baseline"]
        np.testing.assert_array_almost_equal(
            delta_gap["gap_abs_widening"].values,
            expected_widening.values
        )

    def test_row_counts_match(self, standard_quotes):
        """All delta tables should have same row count."""
        baseline = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        stressed = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.05,
            verbose=False,
        )
        result = build_delta_tables(baseline=baseline, stressed=stressed)
        n = len(standard_quotes)
        assert len(result["delta_q1"]) == n
        assert len(result["delta_q2"]) == n
        assert len(result["delta_gap"]) == n

    def test_custom_tolerance(self, standard_quotes):
        """Should accept custom tolerance for Q1 changed flag."""
        baseline = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        stressed = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.05,
            verbose=False,
        )
        result = build_delta_tables(baseline=baseline, stressed=stressed, tol=1e-6)
        delta_q1 = result["delta_q1"]
        assert "simple_changed_flag" in delta_q1.columns

    def test_identical_scenarios_zero_deltas(self, standard_quotes):
        """Delta of identical scenarios should be ~0."""
        scenario = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        result = build_delta_tables(baseline=scenario, stressed=scenario)
        delta_q1 = result["delta_q1"]
        delta_q2 = result["delta_q2"]
        delta_gap = result["delta_gap"]
        
        # All deltas should be near zero
        assert (delta_q1["avg_hazard_delta"].abs() < 1e-10).all()
        assert (delta_q2["avg_hazard_delta"].abs() < 1e-10).all()
        assert (delta_gap["gap_abs_widening"].abs() < 1e-10).all()

    def test_maturities_preserved_in_deltas(self, standard_quotes):
        """Maturities should be preserved in all delta tables."""
        baseline = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        stressed = run_single_scenario(
            cds_quotes=standard_quotes,
            lgd=0.4,
            risk_free_rate=0.05,
            verbose=False,
        )
        result = build_delta_tables(baseline=baseline, stressed=stressed)
        
        input_maturities = sorted(standard_quotes["maturity_years"].values)
        for table_name in ["delta_q1", "delta_q2", "delta_gap"]:
            output_maturities = sorted(result[table_name]["maturity_years"].values)
            np.testing.assert_array_almost_equal(input_maturities, output_maturities)


# ============================================================================
# TestScenarioOutputs
# ============================================================================

class TestScenarioOutputs:
    """Tests for ScenarioOutputs dataclass"""

    def test_scenario_outputs_creation(self, standard_q1_table, standard_q2_table):
        """Should create ScenarioOutputs instance."""
        compare = pd.DataFrame({"maturity_years": [1.0]})
        outputs = ScenarioOutputs(
            risk_free_rate=0.03,
            q1_table=standard_q1_table,
            q2_table_full=standard_q2_table,
            compare_table=compare,
        )
        assert outputs.risk_free_rate == 0.03
        assert isinstance(outputs.q1_table, pd.DataFrame)

    def test_scenario_outputs_frozen(self, standard_q1_table, standard_q2_table):
        """ScenarioOutputs should be frozen (immutable)."""
        compare = pd.DataFrame({"maturity_years": [1.0]})
        outputs = ScenarioOutputs(
            risk_free_rate=0.03,
            q1_table=standard_q1_table,
            q2_table_full=standard_q2_table,
            compare_table=compare,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            outputs.risk_free_rate = 0.04


# ============================================================================
# Integration Tests
# ============================================================================

class TestRunnerIntegration:
    """Integration tests for the runner module"""

    def test_full_stress_test_workflow(self, standard_quotes):
        """Should support full stress test workflow."""
        # Define rate scenarios
        rates = [0.01, 0.03, 0.05]
        scenarios = {}
        
        for rate in rates:
            scenarios[rate] = run_single_scenario(
                cds_quotes=standard_quotes,
                lgd=0.4,
                risk_free_rate=rate,
                verbose=False,
            )
        
        # Build delta tables for adjacent pairs
        baseline = scenarios[0.01]
        stressed = scenarios[0.05]
        deltas = build_delta_tables(baseline=baseline, stressed=stressed)
        
        assert len(deltas["delta_q1"]) > 0
        assert len(deltas["delta_q2"]) > 0
        assert len(deltas["delta_gap"]) > 0

    def test_multiple_lgd_scenarios(self, standard_quotes):
        """Should run multiple LGD scenarios."""
        lgds = [0.2, 0.4, 0.6]
        
        for lgd in lgds:
            result = run_single_scenario(
                cds_quotes=standard_quotes,
                lgd=lgd,
                risk_free_rate=0.03,
                verbose=False,
            )
            assert isinstance(result, ScenarioOutputs)

    def test_narrow_quote_range(self):
        """Should handle quotes with narrow range."""
        narrow = pd.DataFrame({
            "maturity_years": [1.0, 2.0],
            "cds_spread_bps": [100.0, 101.0],
        })
        result = run_single_scenario(
            cds_quotes=narrow,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        assert isinstance(result, ScenarioOutputs)
        assert len(result.q1_table) == 2

    def test_wide_quote_range(self):
        """Should handle quotes with wide maturity range."""
        wide = pd.DataFrame({
            "maturity_years": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
            "cds_spread_bps": [80.0, 100.0, 110.0, 120.0, 125.0, 130.0],
        })
        result = run_single_scenario(
            cds_quotes=wide,
            lgd=0.4,
            risk_free_rate=0.03,
            verbose=False,
        )
        assert isinstance(result, ScenarioOutputs)
        assert len(result.q1_table) == 6
