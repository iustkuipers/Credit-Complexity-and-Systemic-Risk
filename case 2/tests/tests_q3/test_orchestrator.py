import pytest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from services.q3_audit.orchestrator import run_q3, Q3Outputs
from services.q3_audit.audit_7y import Audit7YResult
from services.q2_exact.stripping import strip_forward_hazards_iterative, StripResult


# ============================================================
# Q3Outputs dataclass tests
# ============================================================

class TestQ3Outputs:
    """Test Q3Outputs container."""

    def test_creation(self):
        """Q3Outputs should be created with result_7y and output_dir."""
        audit_result = Audit7YResult(
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
        
        outputs = Q3Outputs(result_7y=audit_result, output_dir=None)
        
        assert outputs.result_7y == audit_result
        assert outputs.output_dir is None

    def test_creation_with_output_dir(self):
        """Q3Outputs can have an output directory."""
        audit_result = Audit7YResult(
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
        
        out_dir = Path("/tmp/output/q3")
        outputs = Q3Outputs(result_7y=audit_result, output_dir=out_dir)
        
        assert outputs.output_dir == out_dir

    def test_frozen(self):
        """Q3Outputs should be frozen (immutable)."""
        audit_result = Audit7YResult(
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
        
        outputs = Q3Outputs(result_7y=audit_result, output_dir=None)
        
        with pytest.raises(AttributeError):
            outputs.result_7y = None


# ============================================================
# Basic orchestration tests
# ============================================================

class TestRunQ3Basic:
    """Test basic Q3 orchestration."""

    @pytest.fixture
    def standard_quotes(self):
        """Standard CDS quotes including 7Y."""
        return pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })

    def test_returns_q3outputs(self, standard_quotes):
        """run_q3 should return Q3Outputs object."""
        result = run_q3(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        assert isinstance(result, Q3Outputs)

    def test_q3outputs_has_audit_result(self, standard_quotes):
        """Q3Outputs should contain an Audit7YResult."""
        result = run_q3(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        assert isinstance(result.result_7y, Audit7YResult)

    def test_audit_result_7y_maturity(self, standard_quotes):
        """Audit result should be for 7Y."""
        result = run_q3(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        assert result.result_7y.maturity_years == 7.0

    def test_audit_result_passed_by_default(self, standard_quotes):
        """Audit should pass by default (NPV ~ 0)."""
        result = run_q3(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            tolerance=1e-6,
            save=False,
            verbose=False,
        )
        assert result.result_7y.passed is True

    def test_output_dir_none_when_save_false(self, standard_quotes):
        """output_dir should be None when save=False."""
        result = run_q3(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        assert result.output_dir is None

    def test_npv_within_tolerance(self, standard_quotes):
        """NPV should be within default tolerance."""
        result = run_q3(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            tolerance=1e-6,
            save=False,
            verbose=False,
        )
        assert result.result_7y.abs_npv <= result.result_7y.tolerance


# ============================================================
# Parameter variation tests
# ============================================================

class TestRunQ3ParameterVariation:
    """Test run_q3 with different parameter combinations."""

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
            result = run_q3(
                cds_quotes=standard_quotes,
                risk_free_rate=r,
                lgd=0.40,
                save=False,
                verbose=False,
            )
            assert isinstance(result, Q3Outputs)

    def test_different_lgd_values(self, standard_quotes):
        """Should work with different LGD values."""
        for lgd in [0.20, 0.40, 0.60, 0.80]:
            result = run_q3(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=lgd,
                save=False,
                verbose=False,
            )
            assert isinstance(result, Q3Outputs)

    def test_different_premium_frequencies(self, standard_quotes):
        """Should work with different premium frequencies."""
        for freq in [1, 2, 4, 12]:
            result = run_q3(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                premium_frequency=freq,
                save=False,
                verbose=False,
            )
            assert isinstance(result, Q3Outputs)

    def test_different_tolerances(self, standard_quotes):
        """Should work with different tolerance values."""
        for tol in [1e-12, 1e-9, 1e-6, 1e-3]:
            result = run_q3(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                tolerance=tol,
                save=False,
                verbose=False,
            )
            assert isinstance(result, Q3Outputs)

    def test_verbose_mode_true(self, standard_quotes, capsys):
        """Verbose mode should print progress."""
        result = run_q3(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=True,
        )
        
        captured = capsys.readouterr()
        assert "[Q3]" in captured.out
        assert "Starting Q3 orchestration" in captured.out

    def test_verbose_mode_false(self, standard_quotes, capsys):
        """Non-verbose mode should not print progress."""
        result = run_q3(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        captured = capsys.readouterr()
        assert "[Q3]" not in captured.out


# ============================================================
# File saving tests
# ============================================================

class TestRunQ3FileOperations:
    """Test file saving operations in run_q3."""

    @pytest.fixture
    def standard_quotes(self):
        """Standard CDS quotes."""
        return pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })

    def test_save_false_no_files_created(self, standard_quotes):
        """With save=False, no files should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q3(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=False,
                verbose=False,
            )
            
            q3_dir = Path(tmpdir) / "q3"
            assert not q3_dir.exists()

    def test_save_true_creates_output_directory(self, standard_quotes):
        """With save=True, output directory should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q3(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            q3_dir = Path(tmpdir) / "q3"
            assert q3_dir.exists()
            assert q3_dir.is_dir()

    def test_save_true_creates_csv(self, standard_quotes):
        """With save=True, q3_audit_7y.csv should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q3(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            csv_path = Path(tmpdir) / "q3" / "q3_audit_7y.csv"
            assert csv_path.exists()

    def test_save_true_creates_xlsx(self, standard_quotes):
        """With save=True, q3_audit_7y.xlsx should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q3(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            xlsx_path = Path(tmpdir) / "q3" / "q3_audit_7y.xlsx"
            assert xlsx_path.exists()

    def test_save_true_creates_markdown(self, standard_quotes):
        """With save=True, q3_audit_7y.md should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q3(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            md_path = Path(tmpdir) / "q3" / "q3_audit_7y.md"
            assert md_path.exists()

    def test_saved_csv_loadable(self, standard_quotes):
        """Saved CSV should be loadable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q3(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            csv_path = Path(tmpdir) / "q3" / "q3_audit_7y.csv"
            loaded = pd.read_csv(csv_path)
            
            assert not loaded.empty
            assert "maturity_years" in loaded.columns
            assert loaded.iloc[0]["maturity_years"] == 7.0

    def test_saved_markdown_readable(self, standard_quotes):
        """Saved markdown should be readable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q3(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            md_path = Path(tmpdir) / "q3" / "q3_audit_7y.md"
            content = md_path.read_text()
            
            assert "Q3 Audit" in content or "7Y" in content

    def test_output_dir_returned_when_save_true(self, standard_quotes):
        """output_dir should be set when save=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q3(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            assert result.output_dir is not None
            assert result.output_dir.exists()


# ============================================================
# Strip result integration tests
# ============================================================

class TestRunQ3WithStripResult:
    """Test run_q3 with provided StripResult."""

    @pytest.fixture
    def standard_quotes(self):
        """Standard CDS quotes."""
        return pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })

    def test_with_provided_strip_result(self, standard_quotes):
        """Should work with provided StripResult."""
        strip_result = strip_forward_hazards_iterative(
            standard_quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        
        result = run_q3(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            strip_result=strip_result,
            save=False,
            verbose=False,
        )
        
        assert isinstance(result, Q3Outputs)
        assert result.result_7y.passed is True

    def test_with_and_without_strip_result_same(self, standard_quotes):
        """Results should be identical with or without provided StripResult."""
        # With internal stripping
        result1 = run_q3(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            strip_result=None,
            save=False,
            verbose=False,
        )
        
        # With provided stripping
        strip_result = strip_forward_hazards_iterative(
            standard_quotes,
            r=0.03,
            lgd=0.40,
            verbose=False,
        )
        result2 = run_q3(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            strip_result=strip_result,
            save=False,
            verbose=False,
        )
        
        # Results should be identical
        assert result1.result_7y.npv == pytest.approx(result2.result_7y.npv)
        assert result1.result_7y.premium_leg_pv == pytest.approx(result2.result_7y.premium_leg_pv)


# ============================================================
# Error handling tests
# ============================================================

class TestRunQ3ErrorHandling:
    """Test error handling in run_q3."""

    def test_missing_7y_quote_raises(self):
        """Should raise error if 7Y quote is missing."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 10],  # No 7Y
            "cds_spread_bps": [100, 110, 120, 125],
        })
        
        with pytest.raises(ValueError, match="Could not find 7Y"):
            run_q3(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                save=False,
                verbose=False,
            )

    def test_missing_columns_raises(self):
        """Should raise error for missing required columns."""
        quotes = pd.DataFrame({
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        with pytest.raises(ValueError):
            run_q3(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                save=False,
                verbose=False,
            )

    def test_invalid_lgd_zero_raises(self):
        """Should raise error for zero LGD."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        with pytest.raises((ValueError, ZeroDivisionError)):
            run_q3(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=0.0,
                save=False,
                verbose=False,
            )


# ============================================================
# Output consistency tests
# ============================================================

class TestRunQ3OutputConsistency:
    """Test consistency of outputs from run_q3."""

    @pytest.fixture
    def standard_quotes(self):
        """Standard CDS quotes."""
        return pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })

    def test_repeated_runs_same_input_same_output(self, standard_quotes):
        """Running twice with same input should give same output."""
        result1 = run_q3(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        result2 = run_q3(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        assert result1.result_7y.npv == pytest.approx(result2.result_7y.npv)
        assert result1.result_7y.passed == result2.result_7y.passed

    def test_result_7y_data_types(self, standard_quotes):
        """Result should have correct data types."""
        result = run_q3(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        audit = result.result_7y
        assert isinstance(audit.maturity_years, float)
        assert isinstance(audit.cds_spread_bps, float)
        assert isinstance(audit.passed, bool)
        assert isinstance(audit.npv, float)


# ============================================================
# Integration tests
# ============================================================

class TestRunQ3Integration:
    """Integration tests combining Q2 and Q3."""

    def test_full_q2_q3_workflow(self):
        """Should audit successfully after Q2 stripping."""
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
        
        # Q3: Audit
        result = run_q3(
            cds_quotes=quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            strip_result=strip_result,
            save=False,
            verbose=False,
        )
        
        # Verification
        assert result.result_7y.maturity_years == 7.0
        assert result.result_7y.passed is True

    def test_workflow_with_save_and_verify(self):
        """Complete workflow with file I/O."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q3(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            # Verify files exist
            q3_dir = Path(tmpdir) / "q3"
            assert (q3_dir / "q3_audit_7y.csv").exists()
            assert (q3_dir / "q3_audit_7y.xlsx").exists()
            assert (q3_dir / "q3_audit_7y.md").exists()
            
            # Verify content
            loaded = pd.read_csv(q3_dir / "q3_audit_7y.csv")
            assert loaded.iloc[0]["passed"] == result.result_7y.passed

    def test_case_2_standard_workflow(self):
        """Test with standard Case 2 data."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        result = run_q3(
            cds_quotes=quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            premium_frequency=4,
            tolerance=1e-6,
            save=False,
            verbose=False,
        )
        
        # Should pass validation
        assert result.result_7y.passed is True
        assert result.result_7y.maturity_years == 7.0
