# services/q3_audit/audit_7y.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from services.instruments.curves import DiscountCurveFlat, build_payment_times
from services.instruments.cds import price_receiver_cds
from services.q2_exact.stripping import strip_forward_hazards_iterative, StripResult


@dataclass(frozen=True)
class Audit7YResult:
    """
    Output of the Q3 audit for the 7Y CDS.
    """
    maturity_years: float
    cds_spread_bps: float
    cds_spread: float

    premium_leg_pv: float
    protection_leg_pv: float
    npv: float
    abs_npv: float

    tolerance: float
    passed: bool


def audit_7y(
    *,
    cds_quotes: pd.DataFrame,
    risk_free_rate: float,
    lgd: float,
    premium_frequency: int = 4,
    tolerance: float = 1e-6,
    strip_result: Optional[StripResult] = None,
    verbose: bool = True,
) -> Audit7YResult:
    """
    Q3: Model Validation (Audit) for the 7Y CDS.

    Using hazard rates derived in Q2, explicitly compute:
      1) PV Premium Leg (quarterly premiums + accrued premium)
      2) PV Protection Leg (midpoint discounting)
      3) Confirm NPV = Premium - Protection is ~ 0 within tolerance

    Notes
    -----
    - No pricing logic should live here; we reuse services/instruments/cds.py.
    - If strip_result is not supplied, we run Q2 stripping internally.
    """
    # Identify the 7Y quote
    req_cols = ["maturity_years", "cds_spread_bps"]
    missing = [c for c in req_cols if c not in cds_quotes.columns]
    if missing:
        raise ValueError(f"cds_quotes missing columns: {missing}")

    dfq = cds_quotes.copy()
    dfq["maturity_years"] = dfq["maturity_years"].astype(float)
    dfq["cds_spread_bps"] = dfq["cds_spread_bps"].astype(float)

    row_7 = dfq.loc[np.isclose(dfq["maturity_years"], 7.0)]
    if row_7.empty:
        raise ValueError("Could not find 7Y row in cds_quotes (maturity_years == 7).")

    cds_spread_bps = float(row_7.iloc[0]["cds_spread_bps"])
    cds_spread = cds_spread_bps / 10_000.0

    # Get stripped curve (from provided result or by running Q2 stripping)
    if strip_result is None:
        if verbose:
            print("[Q3] No StripResult provided. Running Q2 stripping to obtain hazard curve...")
        strip_result = strip_forward_hazards_iterative(
            cds_quotes=cds_quotes,
            r=float(risk_free_rate),
            lgd=float(lgd),
            payments_per_year=int(premium_frequency),
            verbose=verbose,
        )
    else:
        if verbose:
            print("[Q3] Using provided StripResult (hazard curve from Q2).")

    hazard_curve = strip_result.hazard_curve
    discount_curve = DiscountCurveFlat(r=float(risk_free_rate))

    maturity = 7.0
    payment_times = build_payment_times(maturity=maturity, payments_per_year=int(premium_frequency))

    if verbose:
        print("[Q3] Pricing 7Y CDS legs using stripped hazard curve...")
        print(f"[Q3] Inputs: T={maturity}, spread={cds_spread_bps} bps, r={risk_free_rate}, LGD={lgd}")

    legs = price_receiver_cds(
        maturity=maturity,
        spread=float(cds_spread),
        lgd=float(lgd),
        discount_curve=discount_curve,
        hazard_curve=hazard_curve,
        payment_times=payment_times,
    )

    premium_pv = float(legs.premium_leg)
    protection_pv = float(legs.protection_leg)
    npv = float(legs.npv_receiver)
    abs_npv = abs(npv)
    passed = abs_npv <= float(tolerance)

    if verbose:
        print("[Q3] Results (7Y):")
        print(f"     - PV Premium Leg   : {premium_pv:.12f}")
        print(f"     - PV Protection Leg: {protection_pv:.12f}")
        print(f"     - NPV (Prem-Prot)  : {npv:.12e}")
        print(f"     - |NPV|            : {abs_npv:.12e}")
        print(f"     - Passed (tol={tolerance}): {passed}")

    return Audit7YResult(
        maturity_years=maturity,
        cds_spread_bps=cds_spread_bps,
        cds_spread=float(cds_spread),
        premium_leg_pv=premium_pv,
        protection_leg_pv=protection_pv,
        npv=npv,
        abs_npv=abs_npv,
        tolerance=float(tolerance),
        passed=bool(passed),
    )
