# validation/q4_validation.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    details: str


@dataclass(frozen=True)
class Q4ValidationOutputs:
    checks: List[CheckResult]
    summary_df: pd.DataFrame
    report_md: str


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Loaded empty file: {path}")
    return df


def run_q4_validation(
    *,
    output_root: str = "output",
    tol: float = 1e-12,
    save: bool = True,
    verbose: bool = True,
) -> Q4ValidationOutputs:
    """
    Q4 Validation:
    - Confirms Q1 invariance to interest rates
    - Confirms Q2 sensitivity to interest rates
    - Confirms widening gap between Q1 and Q2 at high rates
    """

    q4_dir = Path(output_root) / "q4"

    # Load delta tables produced by Q4
    delta_q1 = _load_csv(q4_dir / "delta_q1.csv")
    delta_q2 = _load_csv(q4_dir / "delta_q2.csv")
    delta_gap = _load_csv(q4_dir / "delta_gap.csv")

    checks: List[CheckResult] = []

    # --------------------------------------------------
    # Check 1: Simple model invariance
    # --------------------------------------------------
    max_q1_delta = float(delta_q1["avg_hazard_delta"].abs().max())

    checks.append(
        CheckResult(
            name="Q1 Simple model invariant to interest rate",
            passed=(max_q1_delta <= tol),
            details=f"Max |Δ avg hazard (Q1)| = {max_q1_delta:.3e}, tol = {tol:.1e}",
        )
    )

    # --------------------------------------------------
    # Check 2: Exact model sensitivity
    # --------------------------------------------------
    max_q2_delta = float(delta_q2["avg_hazard_delta"].abs().max())

    checks.append(
        CheckResult(
            name="Q2 Exact model sensitive to interest rate",
            passed=(max_q2_delta > tol),
            details=f"Max |Δ avg hazard (Q2)| = {max_q2_delta:.3e} (should be > tol)",
        )
    )

    # --------------------------------------------------
    # Check 3: Widening gap under stress
    # --------------------------------------------------
    widening = delta_gap["gap_abs_widening"].to_numpy(dtype=float)

    mean_widen = float(np.mean(widening))
    max_widen = float(np.max(widening))

    checks.append(
        CheckResult(
            name="Simple–Exact hazard gap widens at high rates",
            passed=(max_widen > tol),
            details=f"Mean widening = {mean_widen:.3e}, Max widening = {max_widen:.3e}",
        )
    )

    # --------------------------------------------------
    # Summary table
    # --------------------------------------------------
    summary_df = pd.DataFrame(
        {
            "check": [c.name for c in checks],
            "passed": [c.passed for c in checks],
            "details": [c.details for c in checks],
        }
    )

    # --------------------------------------------------
    # Markdown report
    # --------------------------------------------------
    total = len(checks)
    passed = sum(c.passed for c in checks)
    failed = total - passed

    lines = []
    lines.append("# Q4 Validation Report — Stress Testing\n")
    lines.append(f"- Total checks: **{total}**")
    lines.append(f"- Passed: **{passed}**")
    lines.append(f"- Failed: **{failed}**\n")

    if failed == 0:
        lines.append("All Q4 stress-test validation checks passed.\n")
    else:
        lines.append("## Failed checks\n")
        for c in checks:
            if not c.passed:
                lines.append(f"- **{c.name}** — {c.details}")
        lines.append("")

    lines.append("## Check details\n")
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        lines.append(f"- [{status}] **{c.name}** — {c.details}")

    lines.append("")
    report_md = "\n".join(lines)

    # --------------------------------------------------
    # Save artifacts
    # --------------------------------------------------
    if save:
        out_dir = Path(output_root) / "validation" / "q4"
        _ensure_dir(out_dir)

        summary_df.to_csv(out_dir / "q4_validation_metrics.csv", index=False)
        (out_dir / "q4_validation_report.md").write_text(report_md, encoding="utf-8")

        if verbose:
            print(f"[Q4][validation] Saved validation artifacts to {out_dir}")

    if verbose:
        print(f"[Q4][validation] Done. Failed checks: {failed}")

    return Q4ValidationOutputs(
        checks=checks,
        summary_df=summary_df,
        report_md=report_md,
    )


if __name__ == "__main__":
    run_q4_validation(
        output_root="output",
        tol=1e-12,
        save=True,
        verbose=True,
    )
