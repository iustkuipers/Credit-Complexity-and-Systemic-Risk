# services/q4_stress/orchestrator.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from services.q4_stress.runner import (
    ScenarioOutputs,
    run_single_scenario,
    build_delta_tables,
)
from services.q4_stress.writer import (
    save_scenario_outputs,
    save_delta_tables as save_delta_tables_files,
    save_q4_notes,
)


@dataclass(frozen=True)
class Q4Outputs:
    baseline: ScenarioOutputs
    stressed: ScenarioOutputs
    deltas: Dict[str, pd.DataFrame]
    output_dir: Optional[Path]


def run_q4(
    *,
    cds_quotes: pd.DataFrame,
    lgd: float,
    premium_frequency: int = 4,
    baseline_r: float = 0.03,
    stressed_r: float = 0.10,
    output_root: str = "output",
    save: bool = True,
    verbose: bool = True,
) -> Q4Outputs:
    """
    Q4 Orchestrator: Analysis & Stress Testing (numeric outputs only).

    Runs:
      - Baseline scenario (r = baseline_r)
      - High-rate scenario (r = stressed_r)

    Produces:
      - per-scenario tables for Q1, Q2, and comparison
      - delta tables showing changes and widening gaps
      - a short notes markdown with key numeric facts

    No interpretation is done here beyond computing deltas/flags.
    """
    if verbose:
        print("[Q4] Starting Q4 orchestration (baseline + stressed scenarios).")
        print("[Q4] Parameters:")
        print(f"     - LGD: {lgd}")
        print(f"     - premium_frequency: {premium_frequency}")
        print(f"     - baseline_r: {baseline_r}")
        print(f"     - stressed_r: {stressed_r}")

    # Run scenarios
    if verbose:
        print("[Q4] Running baseline scenario...")
    baseline = run_single_scenario(
        cds_quotes=cds_quotes,
        lgd=float(lgd),
        risk_free_rate=float(baseline_r),
        premium_frequency=int(premium_frequency),
        verbose=verbose,
    )

    if verbose:
        print("[Q4] Running stressed scenario...")
    stressed = run_single_scenario(
        cds_quotes=cds_quotes,
        lgd=float(lgd),
        risk_free_rate=float(stressed_r),
        premium_frequency=int(premium_frequency),
        verbose=verbose,
    )

    # Build deltas (for report support)
    if verbose:
        print("[Q4] Building delta tables (stressed - baseline)...")
    deltas = build_delta_tables(baseline=baseline, stressed=stressed)

    out_dir: Optional[Path] = None
    if save:
        if verbose:
            print("[Q4] Saving scenario outputs to output/q4 ...")

        out_dir = save_scenario_outputs(
            scenario=baseline,
            output_root=output_root,
            prefix=f"baseline_r{baseline_r:.2f}".replace(".", "p"),
            verbose=verbose,
        )
        save_scenario_outputs(
            scenario=stressed,
            output_root=output_root,
            prefix=f"stressed_r{stressed_r:.2f}".replace(".", "p"),
            verbose=verbose,
        )

        save_delta_tables_files(
            delta_tables=deltas,
            output_root=output_root,
            verbose=verbose,
        )

        save_q4_notes(
            baseline_r=float(baseline_r),
            stressed_r=float(stressed_r),
            delta_tables=deltas,
            output_root=output_root,
            verbose=verbose,
        )

    if verbose:
        print("[Q4] Done.")

    return Q4Outputs(
        baseline=baseline,
        stressed=stressed,
        deltas=deltas,
        output_dir=out_dir,
    )
