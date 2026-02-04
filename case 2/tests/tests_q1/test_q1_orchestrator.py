import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

from services.q1_simple.orchestrator import (
    _ensure_dir,
    _save_outputs,
    run_q1,
)
from services.q1_simple.simple_model import Q1Outputs


# ============================================================
# Helper function tests
# ============================================================

class TestEnsureDir:
    """Test directory creation."""

    def test_create_single_directory(self):
        """Should create a single directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test_dir"
            assert not test_path.exists()
            _ensure_dir(test_path)
            assert test_path.exists()
            assert test_path.is_dir()

    def test_create_nested_directories(self):
        """Should create nested directory structure (mkdir -p behavior)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "level1" / "level2" / "level3"
            assert not test_path.exists()
            _ensure_dir(test_path)
            assert test_path.exists()
            assert test_path.is_dir()

    def test_idempotent_on_existing_directory(self):
        """Should not fail if directory already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "existing_dir"
            test_path.mkdir()
            assert test_path.exists()
            _ensure_dir(test_path)  # Should not raise
            assert test_path.exists()


# ============================================================
# Save outputs tests
# ============================================================

class TestSaveOutputs:
    """Test output saving functionality."""

    @pytest.fixture
    def sample_outputs(self):
        """Create sample Q1Outputs for testing."""
        table = pd.DataFrame({
            "maturity_years": [1, 3, 5],
            "cds_spread_bps": [100, 110, 120],
            "survival_T": [0.98, 0.94, 0.90]
        })
        answer_text = "# Q1 Results\n\nTest answer text."
        return Q1Outputs(table=table, answer_text=answer_text)

    def test_creates_output_directory(self, sample_outputs):
        """Should create output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output" / "q1"
            assert not out_dir.exists()
            _save_outputs(sample_outputs, out_dir, verbose=False)
            assert out_dir.exists()

    def test_saves_csv_file(self, sample_outputs):
        """Should save results as CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            _save_outputs(sample_outputs, out_dir, verbose=False)
            
            csv_path = out_dir / "q1_results.csv"
            assert csv_path.exists()
            
            # Verify CSV content
            df = pd.read_csv(csv_path)
            assert len(df) == 3
            assert "maturity_years" in df.columns

    def test_saves_xlsx_file(self, sample_outputs):
        """Should save results as XLSX."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            _save_outputs(sample_outputs, out_dir, verbose=False)
            
            xlsx_path = out_dir / "q1_results.xlsx"
            assert xlsx_path.exists()
            
            # Verify XLSX content
            df = pd.read_excel(xlsx_path)
            assert len(df) == 3
            assert "maturity_years" in df.columns

    def test_saves_markdown_file(self, sample_outputs):
        """Should save answer text as Markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            _save_outputs(sample_outputs, out_dir, verbose=False)
            
            md_path = out_dir / "q1_answer.md"
            assert md_path.exists()
            
            # Verify content
            content = md_path.read_text(encoding="utf-8")
            assert "Q1 Results" in content
            assert "Test answer text" in content

    def test_verbose_output(self, sample_outputs, capsys):
        """Verbose mode should print messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            _save_outputs(sample_outputs, out_dir, verbose=True)
            
            captured = capsys.readouterr()
            assert "[Q1]" in captured.out or "Saving" in captured.out

    def test_no_verbose_output(self, sample_outputs, capsys):
        """Non-verbose mode should not print messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            _save_outputs(sample_outputs, out_dir, verbose=False)
            
            captured = capsys.readouterr()
            # Should be minimal or no output
            assert "[Q1]" not in captured.out or captured.out == ""


# ============================================================
# Main orchestration tests
# ============================================================

class TestRunQ1:
    """Test Q1 orchestrator function."""

    @pytest.fixture
    def sample_cds_quotes(self):
        """Create sample CDS quotes."""
        return pd.DataFrame({
            "maturity_years": [1, 3, 5, 7, 10],
            "cds_spread_bps": [100, 110, 120, 120, 125]
        })

    @pytest.fixture
    def sample_lgd(self):
        """Standard LGD."""
        return 0.40

    def test_returns_q1_outputs(self, sample_cds_quotes, sample_lgd):
        """Should return a Q1Outputs object."""
        output = run_q1(
            cds_quotes=sample_cds_quotes,
            lgd=sample_lgd,
            save=False,
            verbose=False
        )
        assert isinstance(output, Q1Outputs)

    def test_output_contains_table(self, sample_cds_quotes, sample_lgd):
        """Output should contain a computed table."""
        output = run_q1(
            cds_quotes=sample_cds_quotes,
            lgd=sample_lgd,
            save=False,
            verbose=False
        )
        assert isinstance(output.table, pd.DataFrame)
        assert len(output.table) == len(sample_cds_quotes)

    def test_output_contains_answer_text(self, sample_cds_quotes, sample_lgd):
        """Output should contain answer text."""
        output = run_q1(
            cds_quotes=sample_cds_quotes,
            lgd=sample_lgd,
            save=False,
            verbose=False
        )
        assert isinstance(output.answer_text, str)
        assert len(output.answer_text) > 0

    def test_save_false_does_not_create_files(self, sample_cds_quotes, sample_lgd):
        """With save=False, no files should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = str(Path(tmpdir) / "output")
            output_dir = Path(output_root) / "q1"
            assert not output_dir.exists()
            
            run_q1(
                cds_quotes=sample_cds_quotes,
                lgd=sample_lgd,
                output_root=output_root,
                save=False,
                verbose=False
            )
            
            # Directory should not be created
            assert not output_dir.exists()

    def test_save_true_creates_files(self, sample_cds_quotes, sample_lgd):
        """With save=True, output files should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = str(Path(tmpdir) / "output")
            
            run_q1(
                cds_quotes=sample_cds_quotes,
                lgd=sample_lgd,
                output_root=output_root,
                save=True,
                verbose=False
            )
            
            out_dir = Path(output_root) / "q1"
            assert out_dir.exists()
            assert (out_dir / "q1_results.csv").exists()
            assert (out_dir / "q1_results.xlsx").exists()
            assert (out_dir / "q1_answer.md").exists()

    def test_custom_output_root(self, sample_cds_quotes, sample_lgd):
        """Should use custom output_root path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_root = str(Path(tmpdir) / "custom_output")
            
            run_q1(
                cds_quotes=sample_cds_quotes,
                lgd=sample_lgd,
                output_root=custom_root,
                save=True,
                verbose=False
            )
            
            out_dir = Path(custom_root) / "q1"
            assert out_dir.exists()
            assert (out_dir / "q1_results.csv").exists()

    def test_verbose_output_prints_progress(self, sample_cds_quotes, sample_lgd, capsys):
        """Verbose mode should print progress messages."""
        run_q1(
            cds_quotes=sample_cds_quotes,
            lgd=sample_lgd,
            save=False,
            verbose=True
        )
        
        captured = capsys.readouterr()
        assert "[Q1]" in captured.out
        assert "Starting" in captured.out

    def test_verbose_prints_inputs(self, sample_cds_quotes, sample_lgd, capsys):
        """Verbose output should include input information."""
        run_q1(
            cds_quotes=sample_cds_quotes,
            lgd=sample_lgd,
            save=False,
            verbose=True
        )
        
        captured = capsys.readouterr()
        assert "LGD" in captured.out
        assert "Quotes" in captured.out
        assert "5 rows" in captured.out

    def test_verbose_prints_completion(self, sample_cds_quotes, sample_lgd, capsys):
        """Verbose output should include completion message."""
        run_q1(
            cds_quotes=sample_cds_quotes,
            lgd=sample_lgd,
            save=False,
            verbose=True
        )
        
        captured = capsys.readouterr()
        assert "Done" in captured.out or "complete" in captured.out

    def test_no_verbose_output_when_false(self, sample_cds_quotes, sample_lgd, capsys):
        """Non-verbose mode should produce minimal output."""
        run_q1(
            cds_quotes=sample_cds_quotes,
            lgd=sample_lgd,
            save=False,
            verbose=False
        )
        
        captured = capsys.readouterr()
        assert "[Q1]" not in captured.out

    def test_computation_produces_valid_results(self, sample_cds_quotes, sample_lgd):
        """Computation should produce valid mathematical results."""
        output = run_q1(
            cds_quotes=sample_cds_quotes,
            lgd=sample_lgd,
            save=False,
            verbose=False
        )
        
        # Verify mathematical properties
        assert (output.table["survival_T"] >= 0).all()
        assert (output.table["survival_T"] <= 1).all()
        assert (output.table["cum_default_prob"] >= 0).all()
        assert (output.table["cum_default_prob"] <= 1).all()

    def test_with_single_maturity(self, sample_lgd):
        """Should work with single maturity."""
        cds_quotes = pd.DataFrame({
            "maturity_years": [5],
            "cds_spread_bps": [100]
        })
        
        output = run_q1(
            cds_quotes=cds_quotes,
            lgd=sample_lgd,
            save=False,
            verbose=False
        )
        
        assert len(output.table) == 1

    def test_with_many_maturities(self, sample_lgd):
        """Should work with many maturities."""
        cds_quotes = pd.DataFrame({
            "maturity_years": list(range(1, 31)),  # 1-30 years
            "cds_spread_bps": [100 + i*2 for i in range(30)]
        })
        
        output = run_q1(
            cds_quotes=cds_quotes,
            lgd=sample_lgd,
            save=False,
            verbose=False
        )
        
        assert len(output.table) == 30

    def test_default_parameters(self, sample_cds_quotes, sample_lgd):
        """Should work with default parameters."""
        # Call with minimal arguments
        output = run_q1(
            cds_quotes=sample_cds_quotes,
            lgd=sample_lgd
        )
        assert isinstance(output, Q1Outputs)

    def test_full_workflow_with_save(self, sample_cds_quotes, sample_lgd):
        """Full workflow including file saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = str(Path(tmpdir) / "results")
            
            output = run_q1(
                cds_quotes=sample_cds_quotes,
                lgd=sample_lgd,
                output_root=output_root,
                save=True,
                verbose=True
            )
            
            # Verify results
            assert isinstance(output, Q1Outputs)
            assert len(output.table) > 0
            assert len(output.answer_text) > 0
            
            # Verify files
            out_dir = Path(output_root) / "q1"
            assert (out_dir / "q1_results.csv").exists()
            assert (out_dir / "q1_results.xlsx").exists()
            assert (out_dir / "q1_answer.md").exists()
