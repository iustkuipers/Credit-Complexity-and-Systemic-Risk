# validation.py
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from data import (
    TRANSITION,
    START_RATINGS,
    START_RATING_TO_ROW,
    make_portfolio,
)
from model import precompute_thresholds
from simulation import simulate_portfolio_values_t1
from risk_metrics import values_to_losses, var


# =============================================================================
# 1) Default-threshold validation (explicitly required by Case 1)
# =============================================================================
def bbb_default_threshold_analytic() -> float:
    """
    Analytically compute the Z-threshold for a BBB-rated issuer to migrate to Default.

    Definition:
        z_D = Phi^{-1}( P(BBB -> D) )

    Returns:
        z_D (float)
    """
    row = START_RATING_TO_ROW["BBB"]
    p_default = TRANSITION[row, -1]  # last column is 'D'
    return float(norm.ppf(p_default))


def assert_bbb_default_threshold_consistency(
    thresholds: np.ndarray,
    tol: float = 1e-10,
) -> None:
    """
    Check that the simulated threshold for BBB->D matches the analytic value.

    Raises AssertionError if inconsistent.
    """
    z_analytic = bbb_default_threshold_analytic()

    row = START_RATING_TO_ROW["BBB"]
    # In our construction, the threshold for D is the last-but-one cutoff
    # i.e. the cutoff where X <= thr[-2] implies default
    z_sim = thresholds[row, -2]

    if not np.isclose(z_sim, z_analytic, atol=tol):
        raise AssertionError(
            f"BBB default threshold mismatch:\n"
            f" analytic z = {z_analytic:.10f}\n"
            f" simulated z = {z_sim:.10f}\n"
            f" tolerance   = {tol}"
        )


# =============================================================================
# 2) Monte Carlo convergence check (explicitly required by Case 1)
# =============================================================================
def convergence_check_var995(
    *,
    total_notional: float,
    rho: float,
    n_issuers_per_rating: int,
    N: int,
    seeds: list[int],
    portfolio_weights: dict[str, float],
    thresholds: np.ndarray,
) -> dict:
    """
    Run the same experiment with multiple RNG seeds and measure the dispersion
    of the 99.5% VaR.

    This is used to justify the choice of N.

    Returns:
        dict with:
          - N
          - VaR_99_5_per_seed
          - min_VaR
          - max_VaR
          - range
    """
    portfolio = make_portfolio(portfolio_weights)

    v0 = None
    var_vals = []

    for s in seeds:
        v1 = simulate_portfolio_values_t1(
            portfolio=portfolio,
            total_notional=total_notional,
            n_issuers_per_rating=n_issuers_per_rating,
            rho=rho,
            N=N,
            seed=s,
            thresholds=thresholds,
        )

        if v0 is None:
            # compute deterministic V0 once (inside loop only for convenience)
            from data import portfolio_value_t0
            v0 = portfolio_value_t0(portfolio, total_notional)

        losses = values_to_losses(v0, v1)
        var_995 = var(losses, 0.995)
        var_vals.append(var_995)

    var_vals = np.array(var_vals)

    return {
        "N": N,
        "VaR_99_5_per_seed": var_vals,
        "min_VaR": float(var_vals.min()),
        "max_VaR": float(var_vals.max()),
        "range": float(var_vals.max() - var_vals.min()),
    }


def find_stable_N(
    *,
    total_notional: float,
    rho: float,
    n_issuers_per_rating: int,
    portfolio_weights: dict[str, float],
    seeds: list[int],
    N_grid: list[int],
    rel_tol: float = 0.01,
) -> dict:
    """
    Incrementally increase N until the relative dispersion of 99.5% VaR
    across seeds is acceptable.

    Stability criterion:
        (max VaR - min VaR) / mean VaR <= rel_tol

    Returns:
        dict with chosen N and convergence diagnostics
    """
    thresholds = precompute_thresholds()

    for N in N_grid:
        report = convergence_check_var995(
            total_notional=total_notional,
            rho=rho,
            n_issuers_per_rating=n_issuers_per_rating,
            N=N,
            seeds=seeds,
            portfolio_weights=portfolio_weights,
            thresholds=thresholds,
        )

        mean_var = report["VaR_99_5_per_seed"].mean()
        rel_range = report["range"] / mean_var

        report["relative_range"] = float(rel_range)

        if rel_range <= rel_tol:
            report["status"] = "ACCEPTED"
            return report

    report["status"] = "NOT_CONVERGED"
    return report


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    # 1) Threshold validation
    thr = precompute_thresholds()
    assert_bbb_default_threshold_consistency(thr)
    print("BBB default threshold check: OK")

    # 2) Convergence demo (small N just to show it runs)
    demo = find_stable_N(
        total_notional=1500.0,
        rho=0.33,
        n_issuers_per_rating=1,
        portfolio_weights={"BB": 0.6, "B": 0.35, "CCC": 0.05},
        seeds=[1, 2, 3],
        N_grid=[5_000, 10_000],
        rel_tol=0.10,
    )
    print("Convergence demo:", demo)
