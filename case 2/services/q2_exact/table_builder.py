# services/q2_exact/table_builder.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from services.instruments.curves import PiecewiseConstantHazardCurve
from services.q2_exact.stripping import StripResult


_REQUIRED_QUOTE_COLS = ("maturity_years", "cds_spread_bps")


def _validate_quotes(cds_quotes: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(cds_quotes, pd.DataFrame):
        raise TypeError("cds_quotes must be a pandas DataFrame")

    missing = [c for c in _REQUIRED_QUOTE_COLS if c not in cds_quotes.columns]
    if missing:
        raise ValueError(f"cds_quotes missing required columns: {missing}")

    df = cds_quotes.copy().sort_values("maturity_years").reset_index(drop=True)
    if df.empty:
        raise ValueError("cds_quotes is empty")

    maturities = df["maturity_years"].astype(float).to_list()
    if any(maturities[i] >= maturities[i + 1] for i in range(len(maturities) - 1)):
        raise ValueError("maturity_years must be strictly increasing")

    return df


def _integrated_hazard_piecewise(
    tenor_boundaries: Sequence[float],
    forward_hazards: Sequence[float],
    T: float,
) -> float:
    """
    Compute ∫_0^T λ(u) du for a piecewise-constant hazard defined on tenor boundaries.

    tenor_boundaries: [0, 1, 3, 5, 7, 10]
    forward_hazards:  [λ1, λ2, λ3, λ4, λ5]
    """
    if tenor_boundaries[0] != 0.0:
        raise ValueError("tenor_boundaries must start at 0.0")
    if len(forward_hazards) != len(tenor_boundaries) - 1:
        raise ValueError("forward_hazards length must match intervals")

    integ = 0.0
    for i in range(len(forward_hazards)):
        a = float(tenor_boundaries[i])
        b = float(tenor_boundaries[i + 1])
        lam = float(forward_hazards[i])

        if T <= a + 1e-12:
            break

        end = min(T, b)
        integ += lam * max(0.0, end - a)

        if end >= T - 1e-12:
            break

    return float(integ)


def build_q2_results_table(
    cds_quotes: pd.DataFrame,
    strip_result: StripResult,
) -> pd.DataFrame:
    """
    Build the Q2 results table required by the assignment using stripped forward hazards.

    Output columns (full diagnostics):
      - maturity_years
      - cds_spread_bps
      - cds_spread (decimal)
      - avg_hazard (1/T ∫_0^T λ(u) du)
      - fwd_hazard (λ_i over (T_{i-1}, T_i))
      - fwd_default_prob (S(T_{i-1}) - S(T_i))
      - survival_T
      - cum_default_prob
    """
    dfq = _validate_quotes(cds_quotes)

    maturities = dfq["maturity_years"].astype(float).to_list()
    spreads_bps = dfq["cds_spread_bps"].astype(float).to_list()
    spreads = [x / 10_000.0 for x in spreads_bps]

    # Tenor boundaries implied by quotes
    tenor_boundaries = [0.0] + maturities  # [0,1,3,5,7,10]

    fwd_hazards = list(strip_result.forward_hazards)
    if len(fwd_hazards) != len(maturities):
        raise ValueError(
            f"Expected {len(maturities)} forward hazards, got {len(fwd_hazards)}."
        )

    hc: PiecewiseConstantHazardCurve = strip_result.hazard_curve

    # survival at tenor boundaries
    survivals = [hc.survival(t) for t in maturities]
    cum_defaults = [1.0 - s for s in survivals]

    # forward default probabilities between tenor boundaries
    fwd_pd = []
    s_prev = 1.0
    for s in survivals:
        fwd_pd.append(float(s_prev - s))
        s_prev = s

    # average hazard up to each maturity: (1/T) * integral_0^T lambda(u) du
    avg_hazards = []
    for T in maturities:
        integ = _integrated_hazard_piecewise(tenor_boundaries, fwd_hazards, T)
        avg_hazards.append(float(integ / T))

    out = pd.DataFrame(
        {
            "maturity_years": maturities,
            "cds_spread_bps": spreads_bps,
            "cds_spread": spreads,
            "avg_hazard": avg_hazards,
            "fwd_hazard": fwd_hazards,
            "fwd_default_prob": fwd_pd,
            "survival_T": survivals,
            "cum_default_prob": cum_defaults,
        }
    )

    return out


def build_q2_submission_table(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Build the exact table required by the assignment statement for Q2.

    Required columns:
      - Maturity
      - CDS Rate (in bps)
      - (Average) Hazard Rate (defined as (1/T)∫_0^T λ(u)du )
      - Forward Hazard Rate (between T_{i-1} and T_i)
      - Forward Default Probability (between T_{i-1} and T_i)
    """
    required_cols = [
        "maturity_years",
        "cds_spread_bps",
        "avg_hazard",
        "fwd_hazard",
        "fwd_default_prob",
    ]
    missing = [c for c in required_cols if c not in df_full.columns]
    if missing:
        raise ValueError(f"df_full missing columns needed for submission table: {missing}")

    def _mat_label(x: float) -> str:
        if abs(x - round(x)) < 1e-12:
            return f"{int(round(x))}Y"
        return f"{x}Y"

    sub = df_full[required_cols].copy()
    sub["maturity_years"] = sub["maturity_years"].astype(float).map(_mat_label)

    sub = sub.rename(
        columns={
            "maturity_years": "Maturity",
            "cds_spread_bps": "CDS Rate (in bps)",
            "avg_hazard": "(Average) Hazard Rate",
            "fwd_hazard": "Forward Hazard Rate (between T_{i-1} and T_i)",
            "fwd_default_prob": "Forward Default Probability (between T_{i-1} and T_i)",
        }
    )

    return sub
