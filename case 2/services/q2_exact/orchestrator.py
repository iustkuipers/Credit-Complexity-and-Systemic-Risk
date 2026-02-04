# services/q2_exact/orchestrator.py

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from services.q2_exact.stripping import strip_forward_hazards_iterative, StripResult
from services.q2_exact.table_builder import build_q2_results_table
from services.q2_exact.writer import save_q2_outputs


@dataclass(frozen=True)
class Q2Outputs:
    """
    Container for Q2 outputs.
    """
    strip_result: StripResult
    table_full: pd.DataFrame


def run_q2(
    *,
    cds_quotes: pd.DataFrame,
    risk_free_rate: float,
    lgd: float,
    premium_frequency: int = 4,
    output_root: str = "output",
    save: bool = True,
    verbose: bool = True,
) -> Q2Outputs:
    """
    Q2 Orchestrator (Exact Model / Iterative Stripping).

    Responsibilities:
    - Print high-level progress (if verbose=True)
    - Strip forward hazard rates iteratively (root finding)
    - Build the Q2 results table (full diagnostics)
    - Save outputs to output/q2 (if save=True)

    Notes
    -----
    - This orchestrator deliberately contains no pricing logic.
    - Pricing is in services/instruments/cds.py
    - Curve math is in services/instruments/curves.py
    - Stripping is in services/q2_exact/stripping.py
    """
    if verbose:
        print("[Q2] Starting Q2 orchestration (Exact Model / Stripping).")
        print("[Q2] Inputs:")
        print(f"     - risk_free_rate (r): {risk_free_rate}")
        print(f"     - LGD: {lgd}")
        print(f"     - premium_frequency: {premium_frequency}")
        print(f"     - quotes rows: {len(cds_quotes)}")

    # 1) Strip forward hazards (λ1..λ5) such that each tenor CDS NPV ≈ 0
    if verbose:
        print("[Q2] Stripping forward hazard rates iteratively...")
    strip_result = strip_forward_hazards_iterative(
        cds_quotes=cds_quotes,
        r=float(risk_free_rate),
        lgd=float(lgd),
        payments_per_year=int(premium_frequency),
        verbose=verbose,
    )

    if verbose:
        print("[Q2] Stripping complete.")
        print("[Q2] Forward hazards:")
        for i, lam in enumerate(strip_result.forward_hazards, start=1):
            print(f"     - lambda_{i}: {lam:.8f}")

    # 2) Build the full results table required for Q2 deliverables
    if verbose:
        print("[Q2] Building Q2 results table...")
    table_full = build_q2_results_table(cds_quotes=cds_quotes, strip_result=strip_result)

    if verbose:
        print("[Q2] Table built. Preview (first 5 rows):")
        print(table_full.head())

    outputs = Q2Outputs(strip_result=strip_result, table_full=table_full)

    # 3) Save outputs
    if save:
        if verbose:
            print(f"[Q2] Saving outputs to {output_root}/q2 ...")
        save_q2_outputs(
            df_full=table_full,
            diagnostics=strip_result.diagnostics,
            output_root=output_root,
            verbose=verbose,
        )

    if verbose:
        print("[Q2] Done.")

    return outputs
