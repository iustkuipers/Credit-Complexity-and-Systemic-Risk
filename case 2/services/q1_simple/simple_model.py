# services/q1_simple/simple_model.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd


REQUIRED_COLUMNS = ("maturity_years", "cds_spread_bps")


@dataclass(frozen=True)
class Q1Outputs:
    """
    Container for Q1 outputs.
    - table: filled table for Q1 deliverable
    - answer_text: short Markdown-ready explanation (optional; can be created later)
    """
    table: pd.DataFrame
    answer_text: str


def _validate_cds_quotes_df(cds_quotes: pd.DataFrame) -> None:
    if not isinstance(cds_quotes, pd.DataFrame):
        raise TypeError("cds_quotes must be a pandas DataFrame")

    missing = [c for c in REQUIRED_COLUMNS if c not in cds_quotes.columns]
    if missing:
        raise ValueError(f"cds_quotes is missing required columns: {missing}")

    if cds_quotes.empty:
        raise ValueError("cds_quotes is empty")

    if (cds_quotes["maturity_years"] <= 0).any():
        raise ValueError("All maturities must be > 0")

    if (cds_quotes["cds_spread_bps"] < 0).any():
        raise ValueError("All CDS spreads must be >= 0")

    # ensure increasing maturities
    maturities = cds_quotes["maturity_years"].to_numpy()
    if any(maturities[i] >= maturities[i + 1] for i in range(len(maturities) - 1)):
        raise ValueError("maturity_years must be strictly increasing")


def _to_decimal_spread(bps: float) -> float:
    """Convert bps to decimal spread (e.g. 100 bps -> 0.01)."""
    return bps / 10_000.0


def _survival_from_avg_hazard(lam_avg: float, T: float) -> float:
    """S(T) = exp(-lam_avg * T)."""
    return math.exp(-lam_avg * T)


def compute_q1_table(cds_quotes: pd.DataFrame, lgd: float) -> pd.DataFrame:
    """
    Q1 "Simple" model: continuous premium approximation with the rule-of-thumb
      lambda_avg(T) = R(T) / LGD
    and implied survival:
      S(T) = exp(-lambda_avg(T)*T)

    Produces the table required in the assignment:
      - average hazard rate up to Ti
      - forward hazard rate between Ti-1 and Ti (implied from survival)
      - default probability between Ti-1 and Ti

    Notes:
    - This is intentionally a benchmark vs Q2 exact stripping.
    - No discounting is used here by construction.

    Parameters
    ----------
    cds_quotes : pd.DataFrame
        Must contain columns: maturity_years, cds_spread_bps
        Maturities must be strictly increasing.
    lgd : float
        Loss Given Default in decimal (e.g. 0.40)

    Returns
    -------
    pd.DataFrame
        A new DataFrame with added computed columns.
    """
    _validate_cds_quotes_df(cds_quotes)

    if not (0 < lgd <= 1):
        raise ValueError("lgd must be in (0, 1]")

    df = cds_quotes.copy()

    # Convert spreads to decimal
    df["cds_spread"] = df["cds_spread_bps"].astype(float).map(_to_decimal_spread)

    # Average hazard: lambda_avg(T) = R(T) / LGD
    df["avg_hazard"] = df["cds_spread"] / float(lgd)

    # Survival at maturities using average hazard up to each maturity
    df["survival_T"] = [
        _survival_from_avg_hazard(lam, T)
        for lam, T in zip(df["avg_hazard"], df["maturity_years"])
    ]

    # Cumulative default probability up to T
    df["cum_default_prob"] = 1.0 - df["survival_T"]

    # Forward quantities between buckets
    fwd_hazards = []
    fwd_default_probs = []

    maturities = df["maturity_years"].to_list()
    survivals = df["survival_T"].to_list()

    for i, (T_i, S_i) in enumerate(zip(maturities, survivals)):
        if i == 0:
            # interval (0, T1]
            T_prev = 0.0
            S_prev = 1.0
        else:
            T_prev = maturities[i - 1]
            S_prev = survivals[i - 1]

        dt = T_i - T_prev
        if dt <= 0:
            raise ValueError("Non-increasing maturities detected (should not happen after validation).")

        # Implied constant forward hazard over (T_prev, T_i]
        # lambda_fwd = -ln(S(T_i)/S(T_prev)) / (T_i - T_prev)
        # (robust handling when S_i == S_prev == 1.0)
        ratio = S_i / S_prev
        if ratio <= 0:
            raise ValueError("Survival ratio must be positive.")
        lam_fwd = -math.log(ratio) / dt

        # Default probability over (T_prev, T_i]
        pd_fwd = S_prev - S_i

        fwd_hazards.append(lam_fwd)
        fwd_default_probs.append(pd_fwd)

    df["fwd_hazard"] = fwd_hazards
    df["fwd_default_prob"] = fwd_default_probs

    # Order columns nicely for the table deliverable
    preferred_order = [
        "maturity_years",
        "cds_spread_bps",
        "cds_spread",
        "avg_hazard",
        "fwd_hazard",
        "fwd_default_prob",
        "survival_T",
        "cum_default_prob",
    ]
    df = df[[c for c in preferred_order if c in df.columns]]

    return df


def build_q1_answer_text() -> str:
    """
    Provide a short Markdown-ready explanation for Q1.
    Keep it compact; Q1 is primarily about producing the table.
    """
    return (
        "## Q1 – Simple CDS Model (Continuous Premium Approximation)\n\n"
        "Assumptions: continuous premium payments, constant (average) hazard rate up to maturity, "
        "and a fixed LGD.\n\n"
        "For maturity $T$ with par CDS spread $R(T)$ (in decimal), the rule-of-thumb implies:\n\n"
        "$$\\lambda_{\\text{avg}}(T) = \\frac{R(T)}{LGD}$$\n\n"
        "This yields the survival probability:\n\n"
        "$$S(T)=Q(\\tau>T)=\\exp\\{-\\lambda_{\\text{avg}}(T)\\,T\\}$$\n\n"
        "and cumulative default probability $PD(0,T)=1-S(T)$. Forward hazards over "
        "$(T_{i-1},T_i]$ are implied by\n\n"
        "$$\\lambda_{fwd,i}=-\\frac{\\ln(S(T_i)/S(T_{i-1}))}{T_i-T_{i-1}}$$\n\n"
        "and forward default probabilities by $PD(T_{i-1},T_i)=S(T_{i-1})-S(T_i)$. "
        "These outputs serve as a benchmark for the exact iterative stripping in Q2.\n"
    )

def run_q1_simple(cds_quotes: pd.DataFrame, lgd: float) -> Q1Outputs:
    """
    Convenience wrapper returning both the computed table and a ready-to-save answer text.
    (No saving, no prints: orchestration + I/O belongs in the Q1 orchestrator/writer.)
    """
    table = compute_q1_table(cds_quotes=cds_quotes, lgd=lgd)
    answer_text = build_q1_answer_text()
    return Q1Outputs(table=table, answer_text=answer_text)
