import pytest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from services.q2_exact.orchestrator import run_q2, Q2Outputs
from services.q2_exact.stripping import StripResult


# ============================================================
# Q2Outputs dataclass tests
# ============================================================

class TestQ2Outputs:
    """Test Q2Outputs container."""

    def test_creation(self):
        """Q2Outputs should be created with strip_result and table_full."""
        strip_result = StripResult(
            forward_hazards=[0.01, 0.02, 0.03],
            hazard_curve=None,
            diagnostics=pd.DataFrame(),
        )
        table_full = pd.DataFrame({
            "maturity_years": [1, 3, 5],
            "cds_spread_bps": [100, 110, 120],
        })
        
        outputs = Q2Outputs(strip_result=strip_result, table_full=table_full)
        
        assert outputs.strip_result == strip_result
        assert outputs.table_full.equals(table_full)

    def test_frozen(self):
        """Q2Outputs should be frozen (immutable)."""
        strip_result = StripResult(
            forward_hazards=[0.01, 0.02],
            hazard_curve=None,
            diagnostics=pd.DataFrame(),
        )
        table_full = pd.DataFrame({"col": [1, 2]})
        
        outputs = Q2Outputs(strip_result=strip_result, table_full=table_full)
        
        with pytest.raises(AttributeError):
            outputs.strip_result = None


# ============================================================
# Basic orchestration tests
# ============================================================

class TestRunQ2Basic:
    """Test basic Q2 orchestration."""

    @pytest.fixture
    def standard_quotes(self):
        """Standard CDS quotes for testing."""
        return pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })

    def test_returns_q2outputs(self, standard_quotes):
        """run_q2 should return Q2Outputs object."""
        result = run_q2(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        assert isinstance(result, Q2Outputs)

    def test_q2outputs_has_strip_result(self, standard_quotes):
        """Q2Outputs should contain a StripResult."""
        result = run_q2(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        assert isinstance(result.strip_result, StripResult)

    def test_q2outputs_has_table_full(self, standard_quotes):
        """Q2Outputs should contain a full results table."""
        result = run_q2(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        assert isinstance(result.table_full, pd.DataFrame)

    def test_table_full_has_required_columns(self, standard_quotes):
        """Full table should have all required Q2 columns."""
        result = run_q2(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        required = ["maturity_years", "cds_spread_bps", "cds_spread",
                   "avg_hazard", "fwd_hazard", "fwd_default_prob",
                   "survival_T", "cum_default_prob"]
        for col in required:
            assert col in result.table_full.columns

    def test_table_full_rows_match_quotes(self, standard_quotes):
        """Full table rows should match input quotes."""
        result = run_q2(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        assert len(result.table_full) == len(standard_quotes)

    def test_forward_hazards_count_matches_tenors(self, standard_quotes):
        """Forward hazards count should match number of quotes."""
        result = run_q2(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        # 5 tenors -> 5 forward hazards (one per tenor, piecewise constant)
        assert len(result.strip_result.forward_hazards) == len(standard_quotes)

    def test_forward_hazards_all_positive(self, standard_quotes):
        """Forward hazards should all be positive."""
        result = run_q2(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        assert all(h > 0 for h in result.strip_result.forward_hazards)


# ============================================================
# Parameter variation tests
# ============================================================

class TestRunQ2ParameterVariation:
    """Test run_q2 with different parameter combinations."""

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
            result = run_q2(
                cds_quotes=standard_quotes,
                risk_free_rate=r,
                lgd=0.40,
                save=False,
                verbose=False,
            )
            assert isinstance(result, Q2Outputs)
            assert len(result.strip_result.forward_hazards) > 0

    def test_different_lgd_values(self, standard_quotes):
        """Should work with different LGD values."""
        for lgd in [0.20, 0.40, 0.60, 0.80]:
            result = run_q2(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=lgd,
                save=False,
                verbose=False,
            )
            assert isinstance(result, Q2Outputs)
            assert len(result.strip_result.forward_hazards) > 0

    def test_different_premium_frequencies(self, standard_quotes):
        """Should work with different premium frequencies."""
        for freq in [1, 2, 4, 12]:
            result = run_q2(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                premium_frequency=freq,
                save=False,
                verbose=False,
            )
            assert isinstance(result, Q2Outputs)
            assert len(result.strip_result.forward_hazards) > 0

    def test_verbose_mode_true(self, standard_quotes, capsys):
        """Verbose mode should print progress messages."""
        result = run_q2(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=True,
        )
        
        captured = capsys.readouterr()
        assert "[Q2]" in captured.out
        assert "Starting Q2 orchestration" in captured.out

    def test_verbose_mode_false(self, standard_quotes, capsys):
        """Non-verbose mode should not print progress."""
        result = run_q2(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        captured = capsys.readouterr()
        assert "[Q2]" not in captured.out


# ============================================================
# File saving tests
# ============================================================

class TestRunQ2FileOperations:
    """Test file saving operations in run_q2."""

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
            result = run_q2(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=False,
                verbose=False,
            )
            
            q2_dir = Path(tmpdir) / "q2"
            assert not q2_dir.exists()

    def test_save_true_creates_output_directory(self, standard_quotes):
        """With save=True, output directory should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q2(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            q2_dir = Path(tmpdir) / "q2"
            assert q2_dir.exists()
            assert q2_dir.is_dir()

    def test_save_true_creates_full_results_csv(self, standard_quotes):
        """With save=True, q2_results_full.csv should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q2(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            csv_path = Path(tmpdir) / "q2" / "q2_results_full.csv"
            assert csv_path.exists()

    def test_save_true_creates_full_results_xlsx(self, standard_quotes):
        """With save=True, q2_results_full.xlsx should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q2(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            xlsx_path = Path(tmpdir) / "q2" / "q2_results_full.xlsx"
            assert xlsx_path.exists()

    def test_save_true_creates_submission_csv(self, standard_quotes):
        """With save=True, q2_results_submission.csv should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q2(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            csv_path = Path(tmpdir) / "q2" / "q2_results_submission.csv"
            assert csv_path.exists()

    def test_save_true_creates_submission_xlsx(self, standard_quotes):
        """With save=True, q2_results_submission.xlsx should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q2(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            xlsx_path = Path(tmpdir) / "q2" / "q2_results_submission.xlsx"
            assert xlsx_path.exists()

    def test_save_true_creates_diagnostics_csv(self, standard_quotes):
        """With save=True, q2_pricing_diagnostics.csv should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q2(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            csv_path = Path(tmpdir) / "q2" / "q2_pricing_diagnostics.csv"
            assert csv_path.exists()

    def test_save_true_creates_diagnostics_xlsx(self, standard_quotes):
        """With save=True, q2_pricing_diagnostics.xlsx should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q2(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            xlsx_path = Path(tmpdir) / "q2" / "q2_pricing_diagnostics.xlsx"
            assert xlsx_path.exists()

    def test_saved_full_csv_loadable(self, standard_quotes):
        """Saved full CSV should be loadable and match output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q2(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            csv_path = Path(tmpdir) / "q2" / "q2_results_full.csv"
            loaded = pd.read_csv(csv_path)
            
            pd.testing.assert_frame_equal(loaded, result.table_full)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q2(
                cds_quotes=standard_quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            csv_path = Path(tmpdir) / "q2" / "q2_pricing_diagnostics.csv"
            loaded = pd.read_csv(csv_path)
            
            pd.testing.assert_frame_equal(loaded, result.strip_result.diagnostics)
# Different quote scenarios
# ============================================================

class TestRunQ2QuoteScenarios:
    """Test run_q2 with different CDS quote scenarios."""

    def test_minimal_quotes_two_tenors(self):
        """Should work with minimum quotes (2 tenors)."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 5],
            "cds_spread_bps": [100, 120],
        })
        
        result = run_q2(
            cds_quotes=quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        assert len(result.strip_result.forward_hazards) == 2
        assert len(result.table_full) == 2

    def test_flat_spreads(self):
        """Should work with flat (constant) spreads."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 100, 100, 100, 100],
        })
        
        result = run_q2(
            cds_quotes=quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        assert isinstance(result, Q2Outputs)
        # Forward hazards should be relatively similar for flat spreads
        assert all(h > 0 for h in result.strip_result.forward_hazards)

    def test_upward_sloping_spreads(self):
        """Should work with upward sloping spread curve."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [50, 80, 110, 140, 170],
        })
        
        result = run_q2(
            cds_quotes=quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        assert isinstance(result, Q2Outputs)
        assert all(h > 0 for h in result.strip_result.forward_hazards)

    def test_downward_sloping_spreads(self):
        """Downward sloping spreads may not converge (credit improvement)."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [170, 140, 110, 80, 50],
        })
        
        # Downward slopes can fail root solving if hazard rates would need to be negative
        try:
            result = run_q2(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                save=False,
                verbose=False,
            )
            assert isinstance(result, Q2Outputs)
        except ValueError as e:
            # Root bracketing may fail for downward slopes (expected)
            assert "Root not bracketed" in str(e)

    def test_very_small_spreads(self):
        """Should work with very small spreads."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [10, 15, 20, 25, 30],
        })
        
        result = run_q2(
            cds_quotes=quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        assert isinstance(result, Q2Outputs)
        assert all(h > 0 for h in result.strip_result.forward_hazards)

    def test_case_2_standard_quotes(self):
        """Test with standard Case 2 CDS quotes."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        result = run_q2(
            cds_quotes=quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        # Verify mathematical properties
        assert result.table_full["survival_T"].is_monotonic_decreasing
        assert result.table_full["cum_default_prob"].is_monotonic_increasing


# ============================================================
# Error handling tests
# ============================================================

class TestRunQ2ErrorHandling:
    """Test error handling in run_q2."""

    def test_invalid_quotes_missing_column(self):
        """Should raise error for missing required column."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5],
        })  # Missing cds_spread_bps
        
        with pytest.raises(ValueError):
            run_q2(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                save=False,
                verbose=False,
            )

    def test_invalid_quotes_empty_dataframe(self):
        """Should raise error for empty DataFrame."""
        quotes = pd.DataFrame({
            "maturity_years": [],
            "cds_spread_bps": [],
        })
        
        with pytest.raises(ValueError):
            run_q2(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                save=False,
                verbose=False,
            )

    def test_invalid_risk_free_rate_negative(self):
        """Should work with realistic negative rates (though unusual)."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5],
            "cds_spread_bps": [100, 110, 120],
        })
        
        # Negative rates should still work in the model
        result = run_q2(
            cds_quotes=quotes,
            risk_free_rate=-0.01,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        assert isinstance(result, Q2Outputs)

    def test_invalid_lgd_zero(self):
        """Should raise error for zero LGD."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5],
            "cds_spread_bps": [100, 110, 120],
        })
        
        with pytest.raises((ValueError, ZeroDivisionError)):
            run_q2(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=0.0,
                save=False,
                verbose=False,
            )

    def test_invalid_lgd_negative(self):
        """Should raise error for negative LGD."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5],
            "cds_spread_bps": [100, 110, 120],
        })
        
        with pytest.raises(ValueError):
            run_q2(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=-0.40,
                save=False,
                verbose=False,
            )


# ============================================================
# Output consistency tests
# ============================================================

class TestRunQ2OutputConsistency:
    """Test consistency of outputs from run_q2."""

    @pytest.fixture
    def standard_quotes(self):
        """Standard CDS quotes."""
        return pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })

    def test_repeated_runs_same_input_same_output(self, standard_quotes):
        """Running twice with same input should give same output."""
        result1 = run_q2(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        result2 = run_q2(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        # Check forward hazards are the same
        assert np.allclose(
            result1.strip_result.forward_hazards,
            result2.strip_result.forward_hazards,
        )
        
        # Check tables are the same
        pd.testing.assert_frame_equal(result1.table_full, result2.table_full)

    def test_table_full_data_types(self, standard_quotes):
        """Table columns should have appropriate data types."""
        result = run_q2(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        # Numerical columns should be numeric
        numeric_cols = ["maturity_years", "cds_spread_bps", "cds_spread",
                       "avg_hazard", "fwd_hazard", "fwd_default_prob",
                       "survival_T", "cum_default_prob"]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(result.table_full[col])

    def test_diagnostics_dataframe_structure(self, standard_quotes):
        """Diagnostics should be a valid DataFrame."""
        result = run_q2(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        diag = result.strip_result.diagnostics
        assert isinstance(diag, pd.DataFrame)
        assert not diag.empty
        assert "maturity_years" in diag.columns
        assert "npv" in diag.columns

    def test_hazard_curve_object(self, standard_quotes):
        """hazard_curve should be a valid PiecewiseConstantHazardCurve."""
        result = run_q2(
            cds_quotes=standard_quotes,
            risk_free_rate=0.03,
            lgd=0.40,
            save=False,
            verbose=False,
        )
        
        from services.instruments.curves import PiecewiseConstantHazardCurve
        assert isinstance(result.strip_result.hazard_curve, PiecewiseConstantHazardCurve)


# ============================================================
# Integration tests
# ============================================================

class TestRunQ2Integration:
    """Integration tests combining orchestration with file I/O."""

    def test_full_workflow_with_save(self):
        """Complete workflow: run_q2 with save, load, and verify."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run orchestration
            result = run_q2(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            # Verify files exist
            q2_dir = Path(tmpdir) / "q2"
            assert (q2_dir / "q2_results_full.csv").exists()
            assert (q2_dir / "q2_results_submission.csv").exists()
            assert (q2_dir / "q2_pricing_diagnostics.csv").exists()
            
            # Load and verify
            full_loaded = pd.read_csv(q2_dir / "q2_results_full.csv")
            pd.testing.assert_frame_equal(full_loaded, result.table_full)

    def test_workflow_without_save_produces_same_results(self):
        """Results should be identical whether save=True or save=False."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result_with_save = run_q2(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            result_without_save = run_q2(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=False,
                verbose=False,
            )
            
            # Results should be identical
            pd.testing.assert_frame_equal(
                result_with_save.table_full,
                result_without_save.table_full,
            )
            
            assert np.allclose(
                result_with_save.strip_result.forward_hazards,
                result_without_save.strip_result.forward_hazards,
            )

    def test_submission_table_format_matches_assignment(self):
        """Submission table should have correct assignment column names."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125],
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_q2(
                cds_quotes=quotes,
                risk_free_rate=0.03,
                lgd=0.40,
                output_root=tmpdir,
                save=True,
                verbose=False,
            )
            
            # Load submission table
            sub_path = Path(tmpdir) / "q2" / "q2_results_submission.csv"
            submission = pd.read_csv(sub_path)
            
            # Check required columns
            required_cols = [
                "Maturity",
                "CDS Rate (in bps)",
                "(Average) Hazard Rate",
                "Forward Hazard Rate (between T_{i-1} and T_i)",
                "Forward Default Probability (between T_{i-1} and T_i)",
            ]
            
            for col in required_cols:
                assert col in submission.columns

    def test_default_output_root_is_output_dir(self):
        """Default output_root should be 'output' directory."""
        quotes = pd.DataFrame({
            "maturity_years": [1, 3, 5],
            "cds_spread_bps": [100, 110, 120],
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                result = run_q2(
                    cds_quotes=quotes,
                    risk_free_rate=0.03,
                    lgd=0.40,
                    # output_root defaults to "output"
                    save=True,
                    verbose=False,
                )
                
                # Should create output/q2
                assert (Path(tmpdir) / "output" / "q2").exists()
            finally:
                os.chdir(original_cwd)
