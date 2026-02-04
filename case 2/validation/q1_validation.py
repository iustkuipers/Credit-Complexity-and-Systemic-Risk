# validation/q1_validation.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    details: str


@dataclass(frozen=True)
class ValidationOutputs:
    checks: List[CheckResult]
    summary: pd.DataFrame
    report_text: str


# ============================================================
# Loading
# ============================================================

def load_q1_full_results(
    q1_output_dir: str = "output/q1",
    filename: str = "q1_results_full.csv",
) -> pd.DataFrame:
    """
    Load the full Q1 results produced by the Q1 writer.

    Expected default location:
      output/q1/q1_results_full.csv
    """
    path = Path(q1_output_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find Q1 full results at: {path}. "
            "Run Q1 first to generate outputs."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Loaded Q1 results but it is empty: {path}")
    return df


# ============================================================
# Core checks (sanity + accounting)
# ============================================================

def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Q1 results missing required columns: {missing}")


def check_monotone_decreasing(series: pd.Series, tol: float = 1e-12) -> bool:
    x = series.to_numpy(dtype=float)
    return np.all(x[:-1] + tol >= x[1:])


def check_monotone_increasing(series: pd.Series, tol: float = 1e-12) -> bool:
    x = series.to_numpy(dtype=float)
    return np.all(x[:-1] <= x[1:] + tol)


def check_bounds_0_1(series: pd.Series, tol: float = 1e-12) -> bool:
    x = series.to_numpy(dtype=float)
    return np.all((x >= -tol) & (x <= 1.0 + tol))


def check_non_negative(series: pd.Series, tol: float = 1e-12) -> bool:
    x = series.to_numpy(dtype=float)
    return np.all(x >= -tol)


def q1_validate_table_invariants(df: pd.DataFrame, lgd: float, tol: float = 1e-10) -> List[CheckResult]:
    """
    Validate key invariants that must hold for the Q1 simple model output.

    These checks are not "pricing" checks. They are:
    - unit consistency checks
    - monotonicity checks
    - probability bounds
    - simple accounting identities
    """
    required = [
        "maturity_years",
        "cds_spread_bps",
        "cds_spread",
        "avg_hazard",
        "fwd_hazard",
        "fwd_default_prob",
        "survival_T",
        "cum_default_prob",
    ]
    _require_cols(df, required)

    checks: List[CheckResult] = []

    # Sort by maturity in case file ordering got altered
    df = df.sort_values("maturity_years").reset_index(drop=True)

    # 1) Bounds: survival, cumulative default, forward default must be in [0,1]
    for col in ["survival_T", "cum_default_prob", "fwd_default_prob"]:
        ok = check_bounds_0_1(df[col], tol=tol)
        checks.append(CheckResult(
            name=f"Bounds [0,1] for {col}",
            passed=ok,
            details=f"All values of {col} should lie in [0,1] (within tol={tol})."
        ))

    # 2) Survival must be decreasing with maturity
    ok = check_monotone_decreasing(df["survival_T"], tol=tol)
    checks.append(CheckResult(
        name="Survival monotone decreasing",
        passed=ok,
        details="Survival probabilities should be non-increasing with maturity."
    ))

    # 3) Cumulative default must be increasing with maturity
    ok = check_monotone_increasing(df["cum_default_prob"], tol=tol)
    checks.append(CheckResult(
        name="Cumulative default monotone increasing",
        passed=ok,
        details="Cumulative default probabilities should be non-decreasing with maturity."
    ))

    # 4) Non-negativity: hazards and forward PDs should be non-negative
    for col in ["avg_hazard", "fwd_hazard", "fwd_default_prob"]:
        ok = check_non_negative(df[col], tol=tol)
        checks.append(CheckResult(
            name=f"Non-negativity for {col}",
            passed=ok,
            details=f"{col} should be non-negative (within tol={tol})."
        ))

    # 5) Accounting identity: sum of forward PDs equals final cumulative PD
    sum_fwd = float(df["fwd_default_prob"].sum())
    final_cum = float(df.loc[df.index[-1], "cum_default_prob"])
    ok = abs(sum_fwd - final_cum) <= 1e-8
    checks.append(CheckResult(
        name="Forward PDs sum to final cumulative PD",
        passed=ok,
        details=(
            f"sum(fwd_default_prob)={sum_fwd:.12g} vs final cum_default_prob={final_cum:.12g} "
            "(should match within tolerance ~1e-8)."
        ),
    ))

    # 6) Unit check: avg_hazard = (cds_spread / LGD)
    implied = df["cds_spread"].astype(float) / float(lgd)
    max_abs_err = float(np.max(np.abs(df["avg_hazard"].astype(float) - implied)))
    ok = max_abs_err <= 1e-12
    checks.append(CheckResult(
        name="Average hazard matches R/LGD",
        passed=ok,
        details=f"Max abs error between avg_hazard and cds_spread/LGD is {max_abs_err:.3e}."
    ))

    # 7) Survival consistency: survival_T = exp(-avg_hazard * T)
    surv_implied = np.exp(-df["avg_hazard"].astype(float) * df["maturity_years"].astype(float))
    max_abs_err = float(np.max(np.abs(df["survival_T"].astype(float) - surv_implied)))
    ok = max_abs_err <= 1e-12
    checks.append(CheckResult(
        name="Survival matches exp(-avg_hazard*T)",
        passed=ok,
        details=f"Max abs error between survival_T and exp(-avg_hazard*T) is {max_abs_err:.3e}."
    ))

    return checks


# ============================================================
# Metrics table + report
# ============================================================

def build_validation_summary(checks: List[CheckResult]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "check": [c.name for c in checks],
            "passed": [c.passed for c in checks],
            "details": [c.details for c in checks],
        }
    )


def build_report_text(checks: List[CheckResult]) -> str:
    total = len(checks)
    passed = sum(1 for c in checks if c.passed)
    failed = total - passed

    lines = []
    lines.append("# Q1 Validation Report\n")
    lines.append(f"- Total checks: **{total}**")
    lines.append(f"- Passed: **{passed}**")
    lines.append(f"- Failed: **{failed}**\n")

    if failed > 0:
        lines.append("## Failed checks\n")
        for c in checks:
            if not c.passed:
                lines.append(f"- **{c.name}**: {c.details}")
        lines.append("")
    else:
        lines.append("All checks passed.\n")

    lines.append("## Check details\n")
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        lines.append(f"- [{status}] **{c.name}** — {c.details}")

    lines.append("")
    return "\n".join(lines)


# ============================================================
# Saving
# ============================================================

def save_validation_outputs(
    outputs: ValidationOutputs,
    output_dir: str = "output/validation/q1",
    verbose: bool = True,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "q1_validation_metrics.csv"
    md_path = out_dir / "q1_validation_report.md"

    if verbose:
        print(f"[Q1][validation] Writing metrics -> {csv_path}")
    outputs.summary.to_csv(csv_path, index=False)

    if verbose:
        print(f"[Q1][validation] Writing report -> {md_path}")
    md_path.write_text(outputs.report_text, encoding="utf-8")


# ============================================================
# Public entry point
# ============================================================

def run_q1_validation(
    lgd: float,
    q1_output_dir: str = "output/q1",
    q1_filename: str = "q1_results_full.csv",
    save: bool = True,
    output_dir: str = "output/validation/q1",
    verbose: bool = True,
) -> ValidationOutputs:
    """
    Run Q1 validation by loading Q1 outputs from disk and checking invariants.

    This is intended for interpretation/audit:
    - loads Q1 full table
    - runs sanity + identity checks
    - saves a report and metrics for inclusion in your write-up
    """
    if verbose:
        print("[Q1][validation] Loading Q1 outputs...")
    df = load_q1_full_results(q1_output_dir=q1_output_dir, filename=q1_filename)

    if verbose:
        print("[Q1][validation] Running invariant checks...")
    checks = q1_validate_table_invariants(df=df, lgd=lgd)

    summary = build_validation_summary(checks)
    report_text = build_report_text(checks)
    outputs = ValidationOutputs(checks=checks, summary=summary, report_text=report_text)

    if save:
        if verbose:
            print(f"[Q1][validation] Saving validation artifacts to {output_dir} ...")
        save_validation_outputs(outputs, output_dir=output_dir, verbose=verbose)

    if verbose:
        failed = sum(1 for c in checks if not c.passed)
        print(f"[Q1][validation] Done. Failed checks: {failed}")

    return outputs


if __name__ == "__main__":
    # Allows running validation directly:
    #   python validation/q1_validation.py
    #
    # Ensure Q1 has been run first to create output/q1/q1_results_full.csv.
    from data import LGD  # local import to avoid coupling during module import

    run_q1_validation(lgd=LGD, save=True, verbose=True)
