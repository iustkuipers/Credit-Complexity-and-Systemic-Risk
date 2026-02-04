# validation/q3_validation.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from services.q2_exact.stripping import strip_forward_hazards_iterative, StripResult
from services.q3_audit.audit_7y import audit_7y, Audit7YResult


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    details: str


@dataclass(frozen=True)
class Q3ValidationOutputs:
    audit_result: Audit7YResult
    checks: List[CheckResult]
    metrics_df: pd.DataFrame
    report_md: str


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_report_md(audit: Audit7YResult, checks: List[CheckResult]) -> str:
    total = len(checks)
    passed = sum(1 for c in checks if c.passed)
    failed = total - passed

    lines: List[str] = []
    lines.append("# Q3 Validation Report (7Y CDS Audit)\n")
    lines.append(f"- Total checks: **{total}**")
    lines.append(f"- Passed: **{passed}**")
    lines.append(f"- Failed: **{failed}**\n")

    if failed == 0:
        lines.append("All checks passed.\n")
    else:
        lines.append("## Failed checks\n")
        for c in checks:
            if not c.passed:
                lines.append(f"- **{c.name}** — {c.details}")
        lines.append("")

    lines.append("## Audit numbers (7Y)\n")
    lines.append(f"- PV Premium Leg: **{audit.premium_leg_pv:.12f}**")
    lines.append(f"- PV Protection Leg: **{audit.protection_leg_pv:.12f}**")
    lines.append(f"- NPV (Premium − Protection): **{audit.npv:.12e}**")
    lines.append(f"- |NPV|: **{audit.abs_npv:.12e}**")
    lines.append(f"- Tolerance: **{audit.tolerance:.1e}**")
    lines.append(f"- Passed: **{audit.passed}**\n")

    lines.append("## Check details\n")
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        lines.append(f"- [{status}] **{c.name}** — {c.details}")

    lines.append("")
    return "\n".join(lines)


def run_q3_validation(
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
) -> Q3ValidationOutputs:
    """
    Q3 Validation:
    - Reprice 7Y CDS premium and protection legs using Q2-stripped hazard curve
    - Confirm |NPV| <= 1e-6 (default)
    - Save validation report + metrics to output/validation/q3

    Notes
    -----
    - If strip_result is not provided, this validation runs Q2 stripping internally.
    - This is a *validation script*, not unit tests.
    """
    if strip_result is None:
        if verbose:
            print("[Q3][validation] No StripResult provided. Running Q2 stripping first...")
        strip_result = strip_forward_hazards_iterative(
            cds_quotes=cds_quotes,
            r=float(risk_free_rate),
            lgd=float(lgd),
            payments_per_year=int(premium_frequency),
            verbose=verbose,
        )

    audit = audit_7y(
        cds_quotes=cds_quotes,
        risk_free_rate=float(risk_free_rate),
        lgd=float(lgd),
        premium_frequency=int(premium_frequency),
        tolerance=float(tolerance),
        strip_result=strip_result,
        verbose=verbose,
    )

    checks: List[CheckResult] = []

    checks.append(
        CheckResult(
            name="Premium leg PV is positive",
            passed=(audit.premium_leg_pv > 0.0),
            details=f"premium_leg_pv={audit.premium_leg_pv:.12f} (should be > 0).",
        )
    )
    checks.append(
        CheckResult(
            name="Protection leg PV is positive",
            passed=(audit.protection_leg_pv > 0.0),
            details=f"protection_leg_pv={audit.protection_leg_pv:.12f} (should be > 0).",
        )
    )
    checks.append(
        CheckResult(
            name="NPV is within tolerance (|NPV| ≤ tol)",
            passed=(audit.abs_npv <= audit.tolerance),
            details=f"|NPV|={audit.abs_npv:.3e}, tol={audit.tolerance:.1e}.",
        )
    )
    # Extra: consistency check (definition)
    checks.append(
        CheckResult(
            name="NPV equals premium - protection (definition check)",
            passed=abs((audit.premium_leg_pv - audit.protection_leg_pv) - audit.npv) <= 1e-12,
            details=(
                f"(premium-protection) - npv = "
                f"{(audit.premium_leg_pv - audit.protection_leg_pv - audit.npv):.3e} (should be ~0)."
            ),
        )
    )

    metrics_df = pd.DataFrame(
        {
            "check": [c.name for c in checks],
            "passed": [c.passed for c in checks],
            "details": [c.details for c in checks],
        }
    )

    report_md = _build_report_md(audit=audit, checks=checks)

    if save:
        out_dir = Path(output_root) / "validation" / "q3"
        _ensure_dir(out_dir)

        (out_dir / "q3_validation_metrics.csv").write_text(metrics_df.to_csv(index=False), encoding="utf-8")
        (out_dir / "q3_validation_report.md").write_text(report_md, encoding="utf-8")

        # Save the audit result as a single-row CSV too (nice for quick inspection)
        df_audit = pd.DataFrame(
            [
                {
                    "maturity_years": audit.maturity_years,
                    "cds_spread_bps": audit.cds_spread_bps,
                    "cds_spread": audit.cds_spread,
                    "premium_leg_pv": audit.premium_leg_pv,
                    "protection_leg_pv": audit.protection_leg_pv,
                    "npv": audit.npv,
                    "abs_npv": audit.abs_npv,
                    "tolerance": audit.tolerance,
                    "passed": audit.passed,
                }
            ]
        )
        df_audit.to_csv(out_dir / "q3_audit_7y_recomputed.csv", index=False)

        if verbose:
            print(f"[Q3][validation] Saved validation artifacts to: {out_dir}")

    if verbose:
        failed = sum(1 for c in checks if not c.passed)
        print(f"[Q3][validation] Done. Failed checks: {failed}")

    return Q3ValidationOutputs(
        audit_result=audit,
        checks=checks,
        metrics_df=metrics_df,
        report_md=report_md,
    )


if __name__ == "__main__":
    from data import cds_quotes, RISK_FREE_RATE, LGD, PREMIUM_FREQUENCY

    run_q3_validation(
        cds_quotes=cds_quotes,
        risk_free_rate=RISK_FREE_RATE,
        lgd=LGD,
        premium_frequency=PREMIUM_FREQUENCY,
        tolerance=1e-6,
        save=True,
        verbose=True,
    )
