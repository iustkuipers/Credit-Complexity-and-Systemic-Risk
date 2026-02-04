# services/q4_stress/runner.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from services.q1_simple.simple_model import run_q1_simple
from services.q2_exact.stripping import strip_forward_hazards_iterative
from services.q2_exact.table_builder import build_q2_results_table


@dataclass(frozen=True)
class ScenarioOutputs:
    """
    Outputs for a single scenario (given a risk-free rate r).
    """
    risk_free_rate: float
    q1_table: pd.DataFrame
    q2_table_full: pd.DataFrame
    compare_table: pd.DataFrame


def _validate_quotes(cds_quotes: pd.DataFrame) -> pd.DataFrame:
    req = ["maturity_years", "cds_spread_bps"]
    missing = [c for c in req if c not in cds_quotes.columns]
    if missing:
        raise ValueError(f"cds_quotes missing columns: {missing}")
    df = cds_quotes.copy().sort_values("maturity_years").reset_index(drop=True)
    if df.empty:
        raise ValueError("cds_quotes is empty")
    return df


def _merge_q1_q2(
    q1_table: pd.DataFrame,
    q2_table_full: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Q1 and Q2 outputs into a single comparison table.

    We keep both:
    - Q1 avg hazard (simple) = spread/LGD
    - Q2 avg hazard (exact)  = (1/T)∫_0^T λ(u) du
    - Q2 forward hazard      = λ_i
    - Q2 forward PD          = S(T_{i-1}) - S(T_i)
    """
    # Q1 expected columns (from your Q1 outputs):
    # maturity_years, cds_spread_bps, cds_spread, avg_hazard, fwd_hazard, fwd_default_prob, survival_T, cum_default_prob
    # (Q1 "fwd_hazard" isn't meaningful in the same way, but keep it if present.)

    # Q2 full expected columns:
    # maturity_years, cds_spread_bps, cds_spread, avg_hazard, fwd_hazard, fwd_default_prob, survival_T, cum_default_prob

    # To avoid name collisions, suffix columns.
    q1 = q1_table.copy()
    q2 = q2_table_full.copy()

    if "avg_hazard" not in q1.columns:
        raise ValueError("q1_table missing 'avg_hazard'")
    if "avg_hazard" not in q2.columns:
        raise ValueError("q2_table_full missing 'avg_hazard'")

    # Select a stable join key
    keys = ["maturity_years", "cds_spread_bps"]
    for k in keys:
        if k not in q1.columns or k not in q2.columns:
            raise ValueError(f"Missing join key '{k}' in q1 or q2 tables")

    # Drop cds_spread if present (it should be identical in both, derived from cds_spread_bps)
    q1 = q1.drop(columns=["cds_spread"], errors="ignore")
    q2 = q2.drop(columns=["cds_spread"], errors="ignore")

    # Rename key-identical but model-specific columns
    q1 = q1.rename(
        columns={
            "avg_hazard": "avg_hazard_q1_simple",
            "fwd_hazard": "fwd_hazard_q1_simple",
            "fwd_default_prob": "fwd_default_prob_q1_simple",
            "survival_T": "survival_T_q1_simple",
            "cum_default_prob": "cum_default_prob_q1_simple",
        }
    )

    q2 = q2.rename(
        columns={
            "avg_hazard": "avg_hazard_q2_exact",
            "fwd_hazard": "fwd_hazard_q2_exact",
            "fwd_default_prob": "fwd_default_prob_q2_exact",
            "survival_T": "survival_T_q2_exact",
            "cum_default_prob": "cum_default_prob_q2_exact",
        }
    )

    merged = pd.merge(q1, q2, on=keys, how="inner", suffixes=("", ""))

    # Useful derived columns for Q4 writeup
    merged["avg_hazard_gap_q2_minus_q1"] = merged["avg_hazard_q2_exact"] - merged["avg_hazard_q1_simple"]
    merged["avg_hazard_gap_abs"] = merged["avg_hazard_gap_q2_minus_q1"].abs()

    # Keep output tidy / ordered
    col_order = [
        "maturity_years",
        "cds_spread_bps",
        "avg_hazard_q1_simple",
        "avg_hazard_q2_exact",
        "avg_hazard_gap_q2_minus_q1",
        "avg_hazard_gap_abs",
        "fwd_hazard_q2_exact",
        "fwd_default_prob_q2_exact",
        "survival_T_q2_exact",
        "cum_default_prob_q2_exact",
    ]
    # Include any columns we missed (e.g., cds_spread etc.) without dropping
    remaining = [c for c in merged.columns if c not in col_order]
    merged = merged[col_order + remaining]

    return merged


def run_single_scenario(
    *,
    cds_quotes: pd.DataFrame,
    lgd: float,
    risk_free_rate: float,
    premium_frequency: int = 4,
    verbose: bool = True,
) -> ScenarioOutputs:
    """
    Run both models under a single scenario:
      - Q1 simple model (rate-independent)
      - Q2 exact stripping model (rate-dependent)

    Returns:
      - Q1 table
      - Q2 full table
      - merged comparison table
    """
    dfq = _validate_quotes(cds_quotes)

    if verbose:
        print(f"[Q4] Running scenario with r={risk_free_rate:.4f} ...")

    # Q1: simple model (does not use risk_free_rate)
    if verbose:
        print("[Q4]   - Running Q1 simple model...")
    q1_outputs = run_q1_simple(cds_quotes=dfq, lgd=float(lgd))
    q1_table = q1_outputs.table.copy()

    # Q2: exact model (uses risk_free_rate)
    if verbose:
        print("[Q4]   - Running Q2 exact stripping model...")
    strip_res = strip_forward_hazards_iterative(
        cds_quotes=dfq,
        r=float(risk_free_rate),
        lgd=float(lgd),
        payments_per_year=int(premium_frequency),
        verbose=verbose,
    )
    q2_table_full = build_q2_results_table(cds_quotes=dfq, strip_result=strip_res)

    # Comparison
    if verbose:
        print("[Q4]   - Building comparison table...")
    compare = _merge_q1_q2(q1_table=q1_table, q2_table_full=q2_table_full)

    if verbose:
        print("[Q4] Scenario complete.")

    return ScenarioOutputs(
        risk_free_rate=float(risk_free_rate),
        q1_table=q1_table,
        q2_table_full=q2_table_full,
        compare_table=compare,
    )


def build_delta_tables(
    *,
    baseline: ScenarioOutputs,
    stressed: ScenarioOutputs,
    tol: float = 1e-12,
) -> Dict[str, pd.DataFrame]:
    """
    Build delta tables for Q4:
      - delta_q1 = (Q1@r_high - Q1@r_low) should be ~0
      - delta_q2_avg_hazard = (Q2 avg hazards difference)
      - delta_compare = (compare_table differences for key columns)

    Returns a dict of named delta DataFrames.
    """
    # Align on maturity
    key = "maturity_years"

    b1 = baseline.q1_table.sort_values(key).reset_index(drop=True)
    s1 = stressed.q1_table.sort_values(key).reset_index(drop=True)

    b2 = baseline.q2_table_full.sort_values(key).reset_index(drop=True)
    s2 = stressed.q2_table_full.sort_values(key).reset_index(drop=True)

    # Delta Q1 (should be ~0)
    delta_q1 = pd.DataFrame(
        {
            key: b1[key].astype(float),
            "avg_hazard_delta": (s1["avg_hazard"] - b1["avg_hazard"]).astype(float),
        }
    )
    delta_q1["simple_changed_flag"] = delta_q1["avg_hazard_delta"].abs() > tol

    # Delta Q2 (key columns)
    delta_q2 = pd.DataFrame(
        {
            key: b2[key].astype(float),
            "avg_hazard_delta": (s2["avg_hazard"] - b2["avg_hazard"]).astype(float),
            "survival_T_delta": (s2["survival_T"] - b2["survival_T"]).astype(float),
            "cum_default_prob_delta": (s2["cum_default_prob"] - b2["cum_default_prob"]).astype(float),
        }
    )

    # Delta of gaps (widening)
    bc = baseline.compare_table.sort_values(key).reset_index(drop=True)
    sc = stressed.compare_table.sort_values(key).reset_index(drop=True)

    # Ensure required columns exist
    needed = ["avg_hazard_gap_abs", "avg_hazard_gap_q2_minus_q1"]
    for c in needed:
        if c not in bc.columns or c not in sc.columns:
            raise ValueError(f"compare_table missing required column: {c}")

    delta_gap = pd.DataFrame(
        {
            key: bc[key].astype(float),
            "gap_q2_minus_q1_baseline": bc["avg_hazard_gap_q2_minus_q1"].astype(float),
            "gap_q2_minus_q1_stressed": sc["avg_hazard_gap_q2_minus_q1"].astype(float),
            "gap_abs_baseline": bc["avg_hazard_gap_abs"].astype(float),
            "gap_abs_stressed": sc["avg_hazard_gap_abs"].astype(float),
        }
    )
    delta_gap["gap_abs_widening"] = delta_gap["gap_abs_stressed"] - delta_gap["gap_abs_baseline"]

    return {
        "delta_q1": delta_q1,
        "delta_q2": delta_q2,
        "delta_gap": delta_gap,
    }
