# validation/q2_validation.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd

from services.instruments.curves import DiscountCurveFlat, PiecewiseConstantHazardCurve, build_payment_times
from services.instruments.cds import price_receiver_cds


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
    par_check: pd.DataFrame
    survival_grid: pd.DataFrame


# ----------------------------
# Loaders
# ----------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Loaded empty file: {path}")
    return df


def load_q2_outputs(output_root: str = "output") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (q2_results_full, q2_pricing_diagnostics)
    """
    root = Path(output_root) / "q2"
    df_full = _load_csv(root / "q2_results_full.csv")
    df_diag = _load_csv(root / "q2_pricing_diagnostics.csv")
    return df_full, df_diag


# ----------------------------
# Curve rebuild
# ----------------------------

def build_hazard_curve_from_q2_table(df_full: pd.DataFrame) -> PiecewiseConstantHazardCurve:
    required = ["maturity_years", "fwd_hazard"]
    missing = [c for c in required if c not in df_full.columns]
    if missing:
        raise ValueError(f"q2_results_full missing required columns: {missing}")

    df = df_full.sort_values("maturity_years").reset_index(drop=True)
    maturities = df["maturity_years"].astype(float).to_list()
    hazards = df["fwd_hazard"].astype(float).to_list()

    boundaries = [0.0] + maturities
    segments = []
    for i, lam in enumerate(hazards):
        a = float(boundaries[i])
        b = float(boundaries[i + 1])
        segments.append((a, b, float(lam)))

    return PiecewiseConstantHazardCurve(segments=tuple(segments))


# ----------------------------
# Checks
# ----------------------------

def _bounds_0_1(x: np.ndarray, tol: float) -> bool:
    return bool(np.all((x >= -tol) & (x <= 1.0 + tol)))


def _monotone_decreasing(x: np.ndarray, tol: float) -> bool:
    return bool(np.all(x[:-1] + tol >= x[1:]))


def _monotone_increasing(x: np.ndarray, tol: float) -> bool:
    return bool(np.all(x[:-1] <= x[1:] + tol))


def _non_negative(x: np.ndarray, tol: float) -> bool:
    return bool(np.all(x >= -tol))


def run_q2_validation(
    *,
    risk_free_rate: float,
    lgd: float,
    payments_per_year: int = 4,
    output_root: str = "output",
    save: bool = True,
    verbose: bool = True,
) -> ValidationOutputs:
    tol = 1e-10

    df_full, df_diag_saved = load_q2_outputs(output_root=output_root)
    df_full = df_full.sort_values("maturity_years").reset_index(drop=True)

    # Rebuild curve from stripped hazards
    hc = build_hazard_curve_from_q2_table(df_full)
    dc = DiscountCurveFlat(r=float(risk_free_rate))

    maturities = df_full["maturity_years"].astype(float).to_list()
    spreads = df_full["cds_spread"].astype(float).to_list()

    # Dense survival grid (quarterly)
    T_max = float(max(maturities))
    grid = np.array(build_payment_times(maturity=T_max, payments_per_year=payments_per_year), dtype=float)
    surv_grid = np.array([hc.survival(t) for t in grid], dtype=float)
    cumdef_grid = 1.0 - surv_grid

    # Par checks: recompute PVs/NPVs tenor-by-tenor from the rebuilt curve
    rows = []
    for T, R in zip(maturities, spreads):
        pay_times = build_payment_times(maturity=float(T), payments_per_year=payments_per_year)
        legs = price_receiver_cds(
            maturity=float(T),
            spread=float(R),
            lgd=float(lgd),
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=pay_times,
        )
        rows.append(
            {
                "maturity_years": float(T),
                "spread": float(R),
                "premium_leg_pv": float(legs.premium_leg),
                "protection_leg_pv": float(legs.protection_leg),
                "npv": float(legs.npv_receiver),
            }
        )
    df_par = pd.DataFrame(rows)

    checks: List[CheckResult] = []

    # 1) Survival sanity
    checks.append(CheckResult(
        "Bounds [0,1] for survival on quarterly grid",
        _bounds_0_1(surv_grid, tol),
        "All survival probabilities on quarterly grid must lie in [0,1].",
    ))
    checks.append(CheckResult(
        "Survival monotone decreasing on quarterly grid",
        _monotone_decreasing(surv_grid, tol),
        "Survival must be non-increasing over time.",
    ))
    checks.append(CheckResult(
        "Cumulative default monotone increasing on quarterly grid",
        _monotone_increasing(cumdef_grid, tol),
        "Cumulative default must be non-decreasing over time.",
    ))

    # 2) Hazard sanity
    hazards = df_full["fwd_hazard"].astype(float).to_numpy()
    checks.append(CheckResult(
        "Forward hazards non-negative",
        _non_negative(hazards, tol),
        "All stripped forward hazards must be >= 0.",
    ))
    # Soft plausibility bound (very generous)
    max_h = float(np.max(hazards))
    checks.append(CheckResult(
        "Forward hazards not absurdly large",
        bool(max_h < 5.0),
        f"Max forward hazard is {max_h:.6g}. Threshold set at 5.0 (very generous).",
    ))

    # 3) Par condition checks (recomputed)
    max_abs_npv = float(np.max(np.abs(df_par["npv"].to_numpy(dtype=float))))
    checks.append(CheckResult(
        "Par condition: |NPV| small at all tenors (recomputed)",
        bool(max_abs_npv <= 1e-8),
        f"Max |NPV| across tenors is {max_abs_npv:.3e} (threshold 1e-8).",
    ))
    checks.append(CheckResult(
        "Premium and protection PVs positive at all tenors",
        bool((df_par["premium_leg_pv"] > 0).all() and (df_par["protection_leg_pv"] > 0).all()),
        "Both premium_leg_pv and protection_leg_pv should be positive for all tenors.",
    ))

    # 4) Probability mass accounting at tenor boundaries
    surv_tenors = np.array([hc.survival(t) for t in maturities], dtype=float)
    fwd_pd = np.concatenate(([1.0], surv_tenors[:-1])) - surv_tenors
    sum_fwd_pd = float(np.sum(fwd_pd))
    final_cum_pd = float(1.0 - surv_tenors[-1])
    ok_mass = abs(sum_fwd_pd - final_cum_pd) <= 1e-10
    checks.append(CheckResult(
        "Forward PD mass sums to final cumulative PD (tenor boundaries)",
        ok_mass,
        f"sum(fwd_pd)={sum_fwd_pd:.12g} vs final cum PD={final_cum_pd:.12g}.",
    ))

    summary = pd.DataFrame(
        {"check": [c.name for c in checks], "passed": [c.passed for c in checks], "details": [c.details for c in checks]}
    )

    # Report text
    total = len(checks)
    passed = sum(1 for c in checks if c.passed)
    failed = total - passed

    lines = []
    lines.append("# Q2 Validation Report\n")
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

    lines.append("## Par re-pricing checks (recomputed)\n")
    lines.append(df_par.to_markdown(index=False))
    lines.append("\n")

    report_text = "\n".join(lines)

    # Survival grid artifact
    df_grid = pd.DataFrame({"t": grid, "survival": surv_grid, "cum_default": cumdef_grid})

    outputs = ValidationOutputs(
        checks=checks,
        summary=summary,
        report_text=report_text,
        par_check=df_par,
        survival_grid=df_grid,
    )

    if save:
        out_dir = Path(output_root) / "validation" / "q2"
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "q2_validation_metrics.csv").write_text(summary.to_csv(index=False), encoding="utf-8")
        (out_dir / "q2_validation_report.md").write_text(report_text, encoding="utf-8")
        df_par.to_csv(out_dir / "q2_par_check.csv", index=False)
        df_grid.to_csv(out_dir / "q2_survival_grid.csv", index=False)

        if verbose:
            print(f"[Q2][validation] Saved to {out_dir}")

    if verbose:
        print(f"[Q2][validation] Done. Failed checks: {failed}")

    return outputs


if __name__ == "__main__":
    from data import RISK_FREE_RATE, LGD

    run_q2_validation(
        risk_free_rate=RISK_FREE_RATE,
        lgd=LGD,
        payments_per_year=4,
        save=True,
        verbose=True,
    )
