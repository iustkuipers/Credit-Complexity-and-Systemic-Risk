# services/q2_exact/stripping.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, Optional

import math

import pandas as pd

from services.instruments.curves import (
    DiscountCurveFlat,
    PiecewiseConstantHazardCurve,
    build_payment_times,
)
from services.instruments.cds import (
    cds_npv_receiver,
    price_receiver_cds,
)


# ============================================================
# Optional SciPy root finding
# ============================================================

try:
    from scipy.optimize import brentq  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    brentq = None
    _HAVE_SCIPY = False


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class StrippingDiagnosticsRow:
    maturity_years: float
    cds_spread_bps: float
    cds_spread: float
    solved_forward_hazard: float
    premium_leg_pv: float
    protection_leg_pv: float
    npv: float


@dataclass(frozen=True)
class StripResult:
    """
    Output of Q2 stripping.
    """
    forward_hazards: List[float]                      # [λ1..λ5]
    hazard_curve: PiecewiseConstantHazardCurve        # full curve to last maturity
    diagnostics: pd.DataFrame                         # per tenor PVs and NPV


# ============================================================
# Helpers
# ============================================================

def _to_decimal_spread(bps: float) -> float:
    return float(bps) / 10_000.0


def _build_segments_for_tenor(
    tenor_boundaries: Sequence[float],
    forward_hazards: Sequence[float],
    horizon: float,
) -> Tuple[Tuple[float, float, float], ...]:
    """
    Build piecewise segments up to 'horizon' using provided forward hazards.
    tenor_boundaries should include [0, 1, 3, 5, 7, 10] (start at 0).
    forward_hazards is aligned with intervals:
      [0,1), [1,3), [3,5), [5,7), [7,10)
    """
    if tenor_boundaries[0] != 0:
        raise ValueError("tenor_boundaries must start at 0")

    segments = []
    # interval count is len(tenor_boundaries)-1
    for i in range(len(tenor_boundaries) - 1):
        a = float(tenor_boundaries[i])
        b = float(tenor_boundaries[i + 1])
        lam = float(forward_hazards[i])

        if horizon <= a + 1e-12:
            break

        end = min(b, horizon)
        segments.append((a, end, lam))

        if abs(end - horizon) <= 1e-12:
            break

    # Ensure we covered the horizon
    if not segments or abs(segments[-1][1] - horizon) > 1e-10:
        raise ValueError(
            f"Failed to build segments covering horizon={horizon}. "
            f"Last end={segments[-1][1] if segments else None}"
        )

    return tuple(segments)


def _solve_root(
    f: Callable[[float], float],
    low: float,
    high: float,
    *,
    expand_factor: float = 2.0,
    max_high: float = 50.0,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """
    Robust root solve for f(x)=0 on a bracket. Expands the high bound if needed.

    Uses SciPy brentq if available, otherwise bisection.
    """
    if low <= 0 or high <= 0:
        raise ValueError("Hazard rate bracket must be positive.")

    f_low = f(low)
    f_high = f(high)

    # Expand until sign change or max_high reached
    while f_low * f_high > 0 and high < max_high:
        high = min(max_high, high * expand_factor)
        f_high = f(high)

    if f_low * f_high > 0:
        raise ValueError(
            "Root not bracketed for hazard solve. "
            f"f({low})={f_low:.6g}, f({high})={f_high:.6g}. "
            "Try expanding max_high or check inputs."
        )

    if _HAVE_SCIPY and brentq is not None:
        return float(brentq(f, low, high, xtol=tol, rtol=tol, maxiter=max_iter))

    # Fallback: bisection
    a, b = low, high
    fa, fb = f_low, f_high
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) <= tol:
            return float(m)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return float(0.5 * (a + b))


# ============================================================
# Public API
# ============================================================

def strip_forward_hazards_iterative(
    cds_quotes: pd.DataFrame,
    *,
    r: float,
    lgd: float,
    payments_per_year: int = 4,
    verbose: bool = True,
) -> StripResult:
    """
    Iteratively strip piecewise-constant forward hazard rates λ1..λ5 from CDS spreads.

    Parameters
    ----------
    cds_quotes : pd.DataFrame
        Must contain:
          - maturity_years: [1, 3, 5, 7, 10]
          - cds_spread_bps: [100, 110, 120, 120, 125]
        Must be strictly increasing in maturity.
    r : float
        Flat continuously-compounded rate used in the case formula via exp(-r t).
    lgd : float
        Loss given default (0.40 in the case).
    payments_per_year : int
        Quarterly premiums -> 4.
    verbose : bool
        Print progress.

    Returns
    -------
    StripResult
        forward_hazards: [λ1..λ5]
        hazard_curve: piecewise curve to final maturity
        diagnostics: per-tenor PVs and NPV at solved lambdas
    """
    # Validate quotes
    req = ["maturity_years", "cds_spread_bps"]
    missing = [c for c in req if c not in cds_quotes.columns]
    if missing:
        raise ValueError(f"cds_quotes missing columns: {missing}")

    dfq = cds_quotes.sort_values("maturity_years").reset_index(drop=True)
    maturities = dfq["maturity_years"].astype(float).to_list()
    spreads_bps = dfq["cds_spread_bps"].astype(float).to_list()
    spreads = [_to_decimal_spread(x) for x in spreads_bps]

    if any(maturities[i] >= maturities[i + 1] for i in range(len(maturities) - 1)):
        raise ValueError("maturity_years must be strictly increasing")

    # Tenor boundaries must start at 0
    tenor_boundaries = [0.0] + maturities  # [0,1,3,5,7,10]
    n_intervals = len(tenor_boundaries) - 1

    discount_curve = DiscountCurveFlat(r=float(r))

    forward_hazards: List[float] = []
    diagnostics_rows: List[StrippingDiagnosticsRow] = []

    # Iteratively solve λ_i
    for i in range(1, len(tenor_boundaries)):
        T = float(tenor_boundaries[i])          # current maturity
        R = float(spreads[i - 1])               # current spread in decimal
        R_bps = float(spreads_bps[i - 1])

        if verbose:
            print(f"[Q2] Solving lambda_{i} for tenor T={T}y (spread={R_bps} bps) ...")

        # Candidate vector length = i (λ1..λi). Build objective in λi.
        # We will create a full vector for intervals up to i, with previous solved lambdas fixed.
        def objective(lam_i: float) -> float:
            lambdas_i = forward_hazards + [float(lam_i)]
            # Build hazard curve only up to horizon T using these lambdas
            segments = _build_segments_for_tenor(
                tenor_boundaries=tenor_boundaries[: i + 1],   # [0..T]
                forward_hazards=lambdas_i,
                horizon=T,
            )
            hc = PiecewiseConstantHazardCurve(segments=segments)

            pay_times = build_payment_times(maturity=T, payments_per_year=payments_per_year)

            return cds_npv_receiver(
                maturity=T,
                spread=R,
                lgd=float(lgd),
                discount_curve=discount_curve,
                hazard_curve=hc,
                payment_times=pay_times,
            )

        # Bracket for hazard: start small and expand if needed
        lam_star = _solve_root(objective, low=1e-8, high=1.0, max_high=50.0)

        forward_hazards.append(lam_star)

        # Compute diagnostics at solved lambda
        segments = _build_segments_for_tenor(
            tenor_boundaries=tenor_boundaries[: i + 1],
            forward_hazards=forward_hazards,
            horizon=T,
        )
        hc = PiecewiseConstantHazardCurve(segments=segments)
        pay_times = build_payment_times(maturity=T, payments_per_year=payments_per_year)

        legs = price_receiver_cds(
            maturity=T,
            spread=R,
            lgd=float(lgd),
            discount_curve=discount_curve,
            hazard_curve=hc,
            payment_times=pay_times,
        )

        npv = legs.npv_receiver

        if verbose:
            print(f"[Q2]   -> lambda_{i} = {lam_star:.8f}, NPV={npv:.3e}")

        diagnostics_rows.append(
            StrippingDiagnosticsRow(
                maturity_years=T,
                cds_spread_bps=R_bps,
                cds_spread=R,
                solved_forward_hazard=lam_star,
                premium_leg_pv=float(legs.premium_leg),
                protection_leg_pv=float(legs.protection_leg),
                npv=float(npv),
            )
        )

    # Build full hazard curve to final maturity (10Y)
    final_T = float(tenor_boundaries[-1])
    full_segments = _build_segments_for_tenor(
        tenor_boundaries=tenor_boundaries,
        forward_hazards=forward_hazards,
        horizon=final_T,
    )
    full_hc = PiecewiseConstantHazardCurve(segments=full_segments)

    diag_df = pd.DataFrame([r.__dict__ for r in diagnostics_rows])

    return StripResult(
        forward_hazards=forward_hazards,
        hazard_curve=full_hc,
        diagnostics=diag_df,
    )
