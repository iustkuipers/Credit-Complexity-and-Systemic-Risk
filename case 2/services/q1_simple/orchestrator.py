# services/q1_simple/orchestrator.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from services.q1_simple.simple_model import run_q1_simple, Q1Outputs
from services.q1_simple.writer import save_q1_tables, save_q1_answer


def run_q1(
    cds_quotes: pd.DataFrame,
    lgd: float,
    output_root: str = "output",
    save: bool = True,
    verbose: bool = True,
) -> Q1Outputs:
    if verbose:
        print("[Q1] Starting Q1 orchestration (Simple Model).")
        print("[Q1] Inputs:")
        print(f"     - LGD: {lgd}")
        print(f"     - Quotes: {len(cds_quotes)} rows")

    if verbose:
        print("[Q1] Running computations (simple_model.run_q1_simple)...")
    outputs = run_q1_simple(cds_quotes=cds_quotes, lgd=lgd)

    if verbose:
        print("[Q1] Computation complete.")
        print("[Q1] Preview (first 5 rows):")
        print(outputs.table.head())

    if save:
        out_dir = Path(output_root) / "q1"
        if verbose:
            print(f"[Q1] Writing outputs to: {out_dir}")

        # Save BOTH: full + submission tables
        save_q1_tables(df_full=outputs.table, output_dir=out_dir, verbose=verbose)

        # Save answer text
        save_q1_answer(answer_text=outputs.answer_text, output_dir=out_dir, verbose=verbose)

    if verbose:
        print("[Q1] Done.")

    return outputs
