import math
import pytest
import pandas as pd
from services.q1_simple.simple_model import (
    Q1Outputs,
    _validate_cds_quotes_df,
    _to_decimal_spread,
    _survival_from_avg_hazard,
    compute_q1_table,
    build_q1_answer_text,
    run_q1_simple,
    REQUIRED_COLUMNS,
)


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


class TestSurvivalFromAvgHazard:
    """Test survival probability calculation."""

    def test_zero_time(self):
        """Survival at time 0 should be 1.0."""
        assert _survival_from_avg_hazard(lam_avg=0.05, T=0) == pytest.approx(1.0)

    def test_zero_hazard(self):
        """With zero hazard, survival should be 1.0."""
        assert _survival_from_avg_hazard(lam_avg=0.0, T=5.0) == pytest.approx(1.0)

    def test_basic_hazard(self):
        """Test basic hazard calculation."""
        lam = 0.02
        T = 3.0
        expected = math.exp(-lam * T)
        assert _survival_from_avg_hazard(lam, T) == pytest.approx(expected)

    def test_survival_decreases_with_time(self):
        """Survival should decrease with increasing time."""
        s1 = _survival_from_avg_hazard(0.05, 1.0)
        s2 = _survival_from_avg_hazard(0.05, 2.0)
        s3 = _survival_from_avg_hazard(0.05, 3.0)
        assert s1 > s2 > s3


# ============================================================
# Validation tests
# ============================================================

class TestValidateCdsQuotesDF:
    """Test CDS quotes DataFrame validation."""

    def test_valid_dataframe(self):
        """Valid DataFrame should not raise."""
        df = pd.DataFrame({
            "maturity_years": [1, 3, 5],
            "cds_spread_bps": [100, 110, 120]
        })
        _validate_cds_quotes_df(df)  # Should not raise

    def test_not_dataframe_raises(self):
        """Non-DataFrame input should raise TypeError."""
        with pytest.raises(TypeError):
            _validate_cds_quotes_df([1, 2, 3])

    def test_missing_columns_raises(self):
        """Missing required columns should raise ValueError."""
        df = pd.DataFrame({"maturity_years": [1, 3, 5]})
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_cds_quotes_df(df)

    def test_empty_dataframe_raises(self):
        """Empty DataFrame should raise ValueError."""
        df = pd.DataFrame({"maturity_years": [], "cds_spread_bps": []})
        with pytest.raises(ValueError, match="empty"):
            _validate_cds_quotes_df(df)

    def test_non_positive_maturity_raises(self):
        """Non-positive maturities should raise ValueError."""
        df = pd.DataFrame({
            "maturity_years": [0, 1, 3],
            "cds_spread_bps": [100, 110, 120]
        })
        with pytest.raises(ValueError, match="maturities must be > 0"):
            _validate_cds_quotes_df(df)

    def test_negative_spread_raises(self):
        """Negative CDS spreads should raise ValueError."""
        df = pd.DataFrame({
            "maturity_years": [1, 3, 5],
            "cds_spread_bps": [100, -110, 120]
        })
        with pytest.raises(ValueError, match="CDS spreads must be >= 0"):
            _validate_cds_quotes_df(df)

    def test_non_increasing_maturities_raises(self):
        """Non-strictly-increasing maturities should raise ValueError."""
        df = pd.DataFrame({
            "maturity_years": [1, 3, 3, 5],
            "cds_spread_bps": [100, 110, 115, 120]
        })
        with pytest.raises(ValueError, match="strictly increasing"):
            _validate_cds_quotes_df(df)


# ============================================================
# Main computation tests
# ============================================================

class TestComputeQ1Table:
    """Test Q1 table computation."""

    @pytest.fixture
    def simple_cds_quotes(self):
        """Simple 2-row CDS quotes for testing."""
        return pd.DataFrame({
            "maturity_years": [1.0, 3.0],
            "cds_spread_bps": [100.0, 110.0]
        })

    @pytest.fixture
    def standard_lgd(self):
        """Standard LGD value."""
        return 0.40

    def test_output_is_dataframe(self, simple_cds_quotes, standard_lgd):
        """Output should be a DataFrame."""
        result = compute_q1_table(simple_cds_quotes, standard_lgd)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_required_columns(self, simple_cds_quotes, standard_lgd):
        """Output should have all required columns."""
        result = compute_q1_table(simple_cds_quotes, standard_lgd)
        required = ["maturity_years", "cds_spread_bps", "cds_spread", "avg_hazard",
                   "fwd_hazard", "fwd_default_prob", "survival_T", "cum_default_prob"]
        for col in required:
            assert col in result.columns

    def test_output_length_matches_input(self, simple_cds_quotes, standard_lgd):
        """Output DataFrame should have same length as input."""
        result = compute_q1_table(simple_cds_quotes, standard_lgd)
        assert len(result) == len(simple_cds_quotes)

    def test_spread_conversion(self, simple_cds_quotes, standard_lgd):
        """CDS spread should be correctly converted from bps."""
        result = compute_q1_table(simple_cds_quotes, standard_lgd)
        assert result["cds_spread"].iloc[0] == pytest.approx(0.01)
        assert result["cds_spread"].iloc[1] == pytest.approx(0.011)

    def test_avg_hazard_calculation(self, simple_cds_quotes, standard_lgd):
        """Average hazard should equal spread / LGD."""
        result = compute_q1_table(simple_cds_quotes, standard_lgd)
        assert result["avg_hazard"].iloc[0] == pytest.approx(0.01 / 0.40)
        assert result["avg_hazard"].iloc[1] == pytest.approx(0.011 / 0.40)

    def test_survival_is_decreasing(self, simple_cds_quotes, standard_lgd):
        """Survival probabilities should be strictly decreasing."""
        result = compute_q1_table(simple_cds_quotes, standard_lgd)
        survivals = result["survival_T"].tolist()
        for i in range(len(survivals) - 1):
            assert survivals[i] > survivals[i + 1]

    def test_cum_default_prob_is_increasing(self, simple_cds_quotes, standard_lgd):
        """Cumulative default probability should be strictly increasing."""
        result = compute_q1_table(simple_cds_quotes, standard_lgd)
        cum_probs = result["cum_default_prob"].tolist()
        for i in range(len(cum_probs) - 1):
            assert cum_probs[i] < cum_probs[i + 1]

    def test_cum_default_prob_is_one_minus_survival(self, simple_cds_quotes, standard_lgd):
        """Cumulative default prob should equal 1 - survival."""
        result = compute_q1_table(simple_cds_quotes, standard_lgd)
        for idx, row in result.iterrows():
            expected = 1.0 - row["survival_T"]
            assert row["cum_default_prob"] == pytest.approx(expected)

    def test_forward_default_probs_sum_to_cum(self, simple_cds_quotes, standard_lgd):
        """Forward default probs should sum to cumulative."""
        result = compute_q1_table(simple_cds_quotes, standard_lgd)
        cum_at_last = result["cum_default_prob"].iloc[-1]
        sum_fwd = result["fwd_default_prob"].sum()
        assert sum_fwd == pytest.approx(cum_at_last)

    def test_invalid_lgd_raises(self, simple_cds_quotes):
        """Invalid LGD values should raise."""
        with pytest.raises(ValueError, match="lgd must be in"):
            compute_q1_table(simple_cds_quotes, 0.0)
        with pytest.raises(ValueError, match="lgd must be in"):
            compute_q1_table(simple_cds_quotes, 1.5)


# ============================================================
# Q1Outputs dataclass tests
# ============================================================

class TestQ1Outputs:
    """Test Q1Outputs dataclass."""

    def test_q1_outputs_creation(self):
        """Q1Outputs should be created successfully."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        text = "Answer text"
        output = Q1Outputs(table=df, answer_text=text)
        assert output.table is df
        assert output.answer_text == text

    def test_q1_outputs_immutable(self):
        """Q1Outputs should be frozen (immutable)."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        output = Q1Outputs(table=df, answer_text="text")
        with pytest.raises((AttributeError, TypeError)):
            output.table = pd.DataFrame()


# ============================================================
# Answer text tests
# ============================================================

class TestBuildQ1AnswerText:
    """Test Q1 answer text generation."""

    def test_answer_text_is_string(self):
        """Answer text should be a string."""
        text = build_q1_answer_text()
        assert isinstance(text, str)

    def test_answer_text_not_empty(self):
        """Answer text should not be empty."""
        text = build_q1_answer_text()
        assert len(text) > 0

    def test_answer_text_contains_headers(self):
        """Answer text should contain markdown headers."""
        text = build_q1_answer_text()
        assert "Q1" in text
        assert "#" in text

    def test_answer_text_contains_key_concepts(self):
        """Answer text should mention key concepts."""
        text = build_q1_answer_text()
        assert "survival" in text.lower() or "S(T)" in text
        assert "hazard" in text.lower()


# ============================================================
# Integration tests
# ============================================================

class TestRunQ1Simple:
    """Test the main Q1 runner function."""

    @pytest.fixture
    def standard_data(self):
        """Standard test data."""
        cds_quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125]
        })
        lgd = 0.40
        return cds_quotes, lgd

    def test_run_q1_simple_returns_q1outputs(self, standard_data):
        """run_q1_simple should return a Q1Outputs object."""
        cds_quotes, lgd = standard_data
        result = run_q1_simple(cds_quotes, lgd)
        assert isinstance(result, Q1Outputs)

    def test_run_q1_simple_table_computed(self, standard_data):
        """Returned table should be properly computed."""
        cds_quotes, lgd = standard_data
        result = run_q1_simple(cds_quotes, lgd)
        assert len(result.table) == len(cds_quotes)
        assert "survival_T" in result.table.columns

    def test_run_q1_simple_answer_text_provided(self, standard_data):
        """Answer text should be provided and non-empty."""
        cds_quotes, lgd = standard_data
        result = run_q1_simple(cds_quotes, lgd)
        assert isinstance(result.answer_text, str)
        assert len(result.answer_text) > 0

    def test_run_q1_simple_full_workflow(self, standard_data):
        """Full workflow should complete without errors."""
        cds_quotes, lgd = standard_data
        output = run_q1_simple(cds_quotes, lgd)
        
        # Verify table structure
        assert isinstance(output.table, pd.DataFrame)
        assert output.table.shape[0] > 0
        
        # Verify answer text
        assert isinstance(output.answer_text, str)
        assert "Q1" in output.answer_text
