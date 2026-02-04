# services/q4_stress/writer.py

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from services.q4_stress.runner import ScenarioOutputs


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_df(df: pd.DataFrame, csv_path: Path, xlsx_path: Path, verbose: bool) -> None:
    if verbose:
        print(f"[Q4][writer] Writing -> {csv_path}")
    df.to_csv(csv_path, index=False)

    if verbose:
        print(f"[Q4][writer] Writing -> {xlsx_path}")
    df.to_excel(xlsx_path, index=False)


def save_scenario_outputs(
    *,
    scenario: ScenarioOutputs,
    output_root: str = "output",
    prefix: str,
    verbose: bool = True,
) -> Path:
    """
    Save tables for a single scenario under output/q4.

    Files saved:
      - {prefix}_q1_table.csv/xlsx
      - {prefix}_q2_table_full.csv/xlsx
      - {prefix}_compare_hazards.csv/xlsx
    """
    out_dir = Path(output_root) / "q4"
    _ensure_dir(out_dir)

    _save_df(
        scenario.q1_table,
        out_dir / f"{prefix}_q1_table.csv",
        out_dir / f"{prefix}_q1_table.xlsx",
        verbose,
    )

    _save_df(
        scenario.q2_table_full,
        out_dir / f"{prefix}_q2_table_full.csv",
        out_dir / f"{prefix}_q2_table_full.xlsx",
        verbose,
    )

    _save_df(
        scenario.compare_table,
        out_dir / f"{prefix}_compare_hazards.csv",
        out_dir / f"{prefix}_compare_hazards.xlsx",
        verbose,
    )

    return out_dir


def save_delta_tables(
    *,
    delta_tables: Dict[str, pd.DataFrame],
    output_root: str = "output",
    verbose: bool = True,
) -> Path:
    """
    Save delta tables under output/q4.

    Expected keys in delta_tables:
      - delta_q1
      - delta_q2
      - delta_gap
    """
    out_dir = Path(output_root) / "q4"
    _ensure_dir(out_dir)

    for name, df in delta_tables.items():
        _save_df(
            df,
            out_dir / f"{name}.csv",
            out_dir / f"{name}.xlsx",
            verbose,
        )

    return out_dir


def build_q4_notes_md(
    *,
    baseline_r: float,
    stressed_r: float,
    delta_q1: pd.DataFrame,
    delta_gap: pd.DataFrame,
) -> str:
    """
    Produce a short Markdown stub with key facts (no interpretation).
    """
    # Simple model change flag
    changed = bool(delta_q1["simple_changed_flag"].any())
    max_abs_delta_q1 = float(delta_q1["avg_hazard_delta"].abs().max())

    # Gap widening summary
    max_widen = float(delta_gap["gap_abs_widening"].max())
    mean_widen = float(delta_gap["gap_abs_widening"].mean())

    lines = []
    lines.append("# Q4 Scenario Outputs Notes\n")
    lines.append("This file summarizes key numeric facts from the Q4 scenario runs (no interpretation).\n")
    lines.append("## Scenarios\n")
    lines.append(f"- Baseline risk-free rate: **{baseline_r:.2%}**")
    lines.append(f"- Stressed risk-free rate: **{stressed_r:.2%}**\n")
    lines.append("## Simple model invariance check (Q1)\n")
    lines.append(f"- Any change detected in Q1 hazards: **{changed}**")
    lines.append(f"- Max |Δ avg hazard (Q1)| across maturities: **{max_abs_delta_q1:.3e}**\n")
    lines.append("## Gap widening summary (|Q2 − Q1|)\n")
    lines.append(f"- Mean widening in absolute hazard gap: **{mean_widen:.6g}**")
    lines.append(f"- Max widening in absolute hazard gap: **{max_widen:.6g}**\n")
    return "\n".join(lines)


def save_q4_notes(
    *,
    baseline_r: float,
    stressed_r: float,
    delta_tables: Dict[str, pd.DataFrame],
    output_root: str = "output",
    verbose: bool = True,
) -> Path:
    """
    Save a short markdown notes file under output/q4/q4_notes.md
    """
    out_dir = Path(output_root) / "q4"
    _ensure_dir(out_dir)

    if "delta_q1" not in delta_tables or "delta_gap" not in delta_tables:
        raise ValueError("delta_tables must include 'delta_q1' and 'delta_gap' for notes.")

    md = build_q4_notes_md(
        baseline_r=float(baseline_r),
        stressed_r=float(stressed_r),
        delta_q1=delta_tables["delta_q1"],
        delta_gap=delta_tables["delta_gap"],
    )

    md_path = out_dir / "q4_notes.md"
    if verbose:
        print(f"[Q4][writer] Writing -> {md_path}")
    md_path.write_text(md, encoding="utf-8")

    return out_dir
