import pytest
import pandas as pd
from data.data import (
    CASE_NAME,
    COUNTERPARTY,
    RISK_FREE_RATE,
    LGD,
    PREMIUM_FREQUENCY,
    cds_quotes,
)


class TestCaseMetadata:
    """Test case metadata constants."""

    def test_case_name(self):
        """Test that CASE_NAME is correctly defined."""
        assert isinstance(CASE_NAME, str)
        assert "Case 2" in CASE_NAME
        assert "CDS Stripping" in CASE_NAME

    def test_counterparty(self):
        """Test that COUNTERPARTY is correctly defined."""
        assert COUNTERPARTY == "C"
        assert isinstance(COUNTERPARTY, str)


class TestMarketAssumptions:
    """Test market assumption constants."""

    def test_risk_free_rate(self):
        """Test that risk-free rate is valid."""
        assert isinstance(RISK_FREE_RATE, (int, float))
        assert 0 <= RISK_FREE_RATE <= 1
        assert RISK_FREE_RATE == 0.03

    def test_lgd(self):
        """Test that LGD (Loss Given Default) is valid."""
        assert isinstance(LGD, (int, float))
        assert 0 <= LGD <= 1
        assert LGD == 0.40

    def test_premium_frequency(self):
        """Test that premium frequency is valid."""
        assert isinstance(PREMIUM_FREQUENCY, int)
        assert PREMIUM_FREQUENCY > 0
        assert PREMIUM_FREQUENCY == 4  # quarterly


class TestCDSQuotes:
    """Test CDS market data."""

    def test_cds_quotes_is_dataframe(self):
        """Test that cds_quotes is a pandas DataFrame."""
        assert isinstance(cds_quotes, pd.DataFrame)

    def test_cds_quotes_columns(self):
        """Test that cds_quotes has all required columns."""
        required_columns = ["maturity_years", "cds_spread_bps", "formula", "cds_spread"]
        for col in required_columns:
            assert col in cds_quotes.columns

    def test_cds_quotes_length(self):
        """Test that cds_quotes has expected number of rows."""
        assert len(cds_quotes) == 5

    def test_maturity_years(self):
        """Test maturity years are correct and increasing."""
        expected_maturities = [1, 3, 5, 7, 10]
        assert cds_quotes["maturity_years"].tolist() == expected_maturities

    def test_cds_spread_bps(self):
        """Test CDS spreads in basis points."""
        expected_spreads_bps = [100, 110, 120, 120, 125]
        assert cds_quotes["cds_spread_bps"].tolist() == expected_spreads_bps

    def test_cds_spread_conversion(self):
        """Test that cds_spread is correctly converted from basis points."""
        # Verify conversion: bps / 10,000
        for idx, row in cds_quotes.iterrows():
            expected_spread = row["cds_spread_bps"] / 10_000
            assert row["cds_spread"] == expected_spread

    def test_cds_spread_decimals(self):
        """Test specific CDS spread decimal values."""
        expected_spreads_decimal = [0.01, 0.011, 0.012, 0.012, 0.0125]
        assert cds_quotes["cds_spread"].tolist() == expected_spreads_decimal

    def test_formula_column(self):
        """Test formula column has expected values."""
        expected_formulas = ["R(1)", "R(3)", "R(5)", "R(7)", "R(10)"]
        assert cds_quotes["formula"].tolist() == expected_formulas

    def test_no_null_values(self):
        """Test that there are no null values in cds_quotes."""
        assert not cds_quotes.isnull().any().any()

    def test_spread_values_positive(self):
        """Test that all CDS spreads are positive."""
        assert (cds_quotes["cds_spread"] > 0).all()
        assert (cds_quotes["cds_spread_bps"] > 0).all()
