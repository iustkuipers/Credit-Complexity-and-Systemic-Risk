# services/q3_audit/orchestrator.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from services.q2_exact.stripping import StripResult
from services.q3_audit.audit_7y import audit_7y, Audit7YResult
from services.q3_audit.writer import save_q3_audit_7y


@dataclass(frozen=True)
class Q3Outputs:
    """
    Container for Q3 outputs.
    """
    result_7y: Audit7YResult
    output_dir: Optional[Path]


def run_q3(
    *,
    cds_quotes: pd.DataFrame,
    risk_free_rate: float,
    lgd: float,
    premium_frequency: int = 4,
    tolerance: float = 1e-6,
    strip_result: Optional[StripResult] = None,
    output_root: str = "output",
    save: bool = True,
    verbose: bool = True,
) -> Q3Outputs:
    """
    Q3 Orchestrator (Audit).

    Responsibilities:
    - Print high-level progress (if verbose=True)
    - Compute 7Y premium/protection PVs using Q2 hazard curve
    - Save artifacts to output/q3 (if save=True)

    Notes
    -----
    - If strip_result is not provided, Q3 will run Q2 stripping internally.
    """
    if verbose:
        print("[Q3] Starting Q3 orchestration (7Y Audit).")
        print("[Q3] Inputs:")
        print(f"     - r: {risk_free_rate}")
        print(f"     - LGD: {lgd}")
        print(f"     - premium_frequency: {premium_frequency}")
        print(f"     - tolerance: {tolerance}")

    result = audit_7y(
        cds_quotes=cds_quotes,
        risk_free_rate=float(risk_free_rate),
        lgd=float(lgd),
        premium_frequency=int(premium_frequency),
        tolerance=float(tolerance),
        strip_result=strip_result,
        verbose=verbose,
    )

    out_dir: Optional[Path] = None
    if save:
        if verbose:
            print("[Q3] Saving audit artifacts to output/q3 ...")
        out_dir = save_q3_audit_7y(
            result=result,
            output_root=output_root,
            verbose=verbose,
        )

    if verbose:
        print("[Q3] Done.")

    return Q3Outputs(result_7y=result, output_dir=out_dir)
