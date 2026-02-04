# services/q1_simple/writer.py

from __future__ import annotations

from pathlib import Path
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# --- column mapping for the required submission table ---
_SUBMISSION_COLS = {
    "maturity_years": "Maturity",
    "cds_spread_bps": "CDS Rate (in bps)",
    "avg_hazard": "(Average) Hazard Rate",
    "fwd_hazard": "Forward Hazard Rate (between T_{i-1} and T_i)",
    "fwd_default_prob": "Default Probability (between T_{i-1} and T_i)",
}


def _format_maturity_years_to_label(x: float) -> str:
    # 1 -> "1Y", 3 -> "3Y", etc.
    # also robust if it comes in as 1.0
    if abs(x - round(x)) < 1e-12:
        return f"{int(round(x))}Y"
    return f"{x}Y"


def build_q1_submission_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the exact table required by the case outline.

    Output columns:
      - Maturity
      - CDS Rate (in bps)
      - (Average) Hazard Rate
      - Forward Hazard Rate (between T_{i-1} and T_i)
      - Default Probability (between T_{i-1} and T_i)
    """
    missing = [c for c in _SUBMISSION_COLS.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot build submission table; missing columns: {missing}")

    out = df[list(_SUBMISSION_COLS.keys())].copy()

    # maturity formatting: 1 -> 1Y, 3 -> 3Y, etc.
    out["maturity_years"] = out["maturity_years"].astype(float).map(_format_maturity_years_to_label)

    # rename to match the assignment headings
    out = out.rename(columns=_SUBMISSION_COLS)

    return out


def save_q1_tables(
    df_full: pd.DataFrame,
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """
    Save both:
      1) full diagnostic outputs (all columns)
      2) submission table matching the assignment outline
    """
    ensure_dir(output_dir)

    # 1) Full outputs
    full_csv = output_dir / "q1_results_full.csv"
    full_xlsx = output_dir / "q1_results_full.xlsx"

    if verbose:
        print(f"[Q1][writer] Writing full table -> {full_csv}")
    df_full.to_csv(full_csv, index=False)

    if verbose:
        print(f"[Q1][writer] Writing full table -> {full_xlsx}")
    df_full.to_excel(full_xlsx, index=False)

    # 2) Submission outputs (exact required columns)
    df_sub = build_q1_submission_table(df_full)

    sub_csv = output_dir / "q1_results_submission.csv"
    sub_xlsx = output_dir / "q1_results_submission.xlsx"

    if verbose:
        print(f"[Q1][writer] Writing submission table -> {sub_csv}")
    df_sub.to_csv(sub_csv, index=False)

    if verbose:
        print(f"[Q1][writer] Writing submission table -> {sub_xlsx}")
    df_sub.to_excel(sub_xlsx, index=False)


def save_q1_answer(
    answer_text: str,
    output_dir: Path,
    filename: str = "q1_answer.md",
    verbose: bool = True,
) -> None:
    ensure_dir(output_dir)

    file_path = output_dir / filename
    if verbose:
        print(f"[Q1][writer] Writing answer text -> {file_path}")
    file_path.write_text(answer_text, encoding="utf-8")
