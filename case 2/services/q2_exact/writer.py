# services/q2_exact/writer.py

from __future__ import annotations

from pathlib import Path
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# --- Submission column mapping (Q2 table required by the case outline) ---
_SUBMISSION_COLS = {
    "maturity_years": "Maturity",
    "cds_spread_bps": "CDS Rate (in bps)",
    "avg_hazard": "(Average) Hazard Rate",
    "fwd_hazard": "Forward Hazard Rate (between T_{i-1} and T_i)",
    "fwd_default_prob": "Forward Default Probability (between T_{i-1} and T_i)",
}


def _format_maturity_years_to_label(x: float) -> str:
    if abs(x - round(x)) < 1e-12:
        return f"{int(round(x))}Y"
    return f"{x}Y"


def build_q2_submission_table(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Build the exact submission table for Q2 from the full results table.

    Required columns:
      - Maturity
      - CDS Rate (in bps)
      - (Average) Hazard Rate
      - Forward Hazard Rate (between T_{i-1} and T_i)
      - Forward Default Probability (between T_{i-1} and T_i)
    """
    missing = [c for c in _SUBMISSION_COLS.keys() if c not in df_full.columns]
    if missing:
        raise ValueError(f"Cannot build Q2 submission table; missing columns: {missing}")

    out = df_full[list(_SUBMISSION_COLS.keys())].copy()
    out["maturity_years"] = out["maturity_years"].astype(float).map(_format_maturity_years_to_label)
    out = out.rename(columns=_SUBMISSION_COLS)

    return out


def save_q2_tables(
    df_full: pd.DataFrame,
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """
    Save both:
      1) full Q2 results (all columns)
      2) submission table matching the assignment outline
    """
    ensure_dir(output_dir)

    # 1) Full results
    full_csv = output_dir / "q2_results_full.csv"
    full_xlsx = output_dir / "q2_results_full.xlsx"

    if verbose:
        print(f"[Q2][writer] Writing full table -> {full_csv}")
    df_full.to_csv(full_csv, index=False)

    if verbose:
        print(f"[Q2][writer] Writing full table -> {full_xlsx}")
    df_full.to_excel(full_xlsx, index=False)

    # 2) Submission table
    df_sub = build_q2_submission_table(df_full)

    sub_csv = output_dir / "q2_results_submission.csv"
    sub_xlsx = output_dir / "q2_results_submission.xlsx"

    if verbose:
        print(f"[Q2][writer] Writing submission table -> {sub_csv}")
    df_sub.to_csv(sub_csv, index=False)

    if verbose:
        print(f"[Q2][writer] Writing submission table -> {sub_xlsx}")
    df_sub.to_excel(sub_xlsx, index=False)


def save_q2_pricing_diagnostics(
    diagnostics: pd.DataFrame,
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """
    Save the stripping diagnostics (per tenor PV premium/protection and NPV).
    This is extremely useful for validation and for Q3.

    File:
      - q2_pricing_diagnostics.csv
      - q2_pricing_diagnostics.xlsx
    """
    ensure_dir(output_dir)

    csv_path = output_dir / "q2_pricing_diagnostics.csv"
    xlsx_path = output_dir / "q2_pricing_diagnostics.xlsx"

    if verbose:
        print(f"[Q2][writer] Writing pricing diagnostics -> {csv_path}")
    diagnostics.to_csv(csv_path, index=False)

    if verbose:
        print(f"[Q2][writer] Writing pricing diagnostics -> {xlsx_path}")
    diagnostics.to_excel(xlsx_path, index=False)


def save_q2_outputs(
    *,
    df_full: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_root: str = "output",
    verbose: bool = True,
) -> Path:
    """
    Convenience wrapper to save all Q2 outputs under output/q2.

    Returns
    -------
    Path
        The output directory used.
    """
    out_dir = Path(output_root) / "q2"
    ensure_dir(out_dir)

    save_q2_tables(df_full=df_full, output_dir=out_dir, verbose=verbose)
    save_q2_pricing_diagnostics(diagnostics=diagnostics, output_dir=out_dir, verbose=verbose)

    return out_dir
