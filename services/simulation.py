# simulation.py
from __future__ import annotations

import numpy as np

from data import BOND_VALUE_T1, RATINGS, START_RATINGS
from model import precompute_thresholds, migrate_many_to_indices


def _bond_values_t1_vector() -> np.ndarray:
    """
    Vector of bond forward values in RATINGS order (per 100 notional).
    This enables fast lookup by rating index.
    """
    return np.array([BOND_VALUE_T1[r] for r in RATINGS], dtype=float)


def simulate_one_factor_asset_returns(
    n_issuers: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    One-factor Merton model:
        X_i = sqrt(rho)*Y + sqrt(1-rho)*eps_i
    where Y ~ N(0,1) common factor, eps_i iid N(0,1).

    Returns:
        x: shape (n_issuers,)
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0,1]. Got {rho}")

    y = rng.standard_normal()
    eps = rng.standard_normal(n_issuers)
    return np.sqrt(rho) * y + np.sqrt(1.0 - rho) * eps


def simulate_portfolio_values_t1(
    portfolio: dict[str, float],
    total_notional: float,
    n_issuers_per_rating: int,
    rho: float,
    N: int,
    seed: int | None = None,
    thresholds: np.ndarray | None = None,
) -> np.ndarray:
    """
    Simulate portfolio forward values V1 (t=1) under migration/default.

    Args:
        portfolio: {rating: weight} at t=0, weights sum to 1.
                   Ratings must be subset of START_RATINGS (AAA..CCC), not include 'D'.
        total_notional: total invested amount (e.g., 1500.0 in EUR mln) - passed in, not hardcoded.
        n_issuers_per_rating: concentration control.
            - Q1 concentrated: 1
            - Q2 diversified: 100
        rho: asset correlation in [0,1]
        N: number of Monte Carlo scenarios
        seed: RNG seed for reproducibility
        thresholds: optionally pass precomputed thresholds from model.precompute_thresholds()

    Returns:
        v1: np.ndarray shape (N,) of simulated portfolio values at t=1 (same currency units as total_notional)
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if n_issuers_per_rating <= 0:
        raise ValueError("n_issuers_per_rating must be positive.")
    if total_notional <= 0:
        raise ValueError("total_notional must be positive.")
    if not (0.0 <= rho <= 1.0):
        raise ValueError("rho must be in [0,1].")
    for r in portfolio.keys():
        if r not in START_RATINGS:
            raise ValueError(
                f"Portfolio contains rating '{r}' not in start ratings {START_RATINGS}."
            )

    rng = np.random.default_rng(seed)

    if thresholds is None:
        thresholds = precompute_thresholds()

    bond_t1 = _bond_values_t1_vector()  # per 100 notional

    # Precompute allocations per rating bucket
    # allocation amount per rating bucket (currency units)
    alloc_per_rating = {r: w * total_notional for r, w in portfolio.items()}

    v1 = np.empty(N, dtype=float)

    # Main MC loop (kept explicit for clarity; optimize later if needed)
    for k in range(N):
        port_value = 0.0

        for r, alloc in alloc_per_rating.items():
            # Split bucket allocation equally across issuers within rating bucket
            alloc_per_issuer = alloc / n_issuers_per_rating

            # Simulate asset returns for issuers in this rating bucket (share same Y across *all* issuers overall?)
            # IMPORTANT: The case model says: Xi = sqrt(rho) Y + sqrt(1-rho) eps_i for issuer i.
            # That implies the SAME common factor Y applies across the entire portfolio (all issuers),
            # not per rating bucket. Therefore, we must draw ONE Y per scenario, not per bucket.
            #
            # Implementation approach:
            # - Draw Y once per scenario
            # - For each bucket: draw eps vector, build Xi using same Y
            #
            # We'll do that: draw y once outside rating loop.

        # We need the common factor y for whole scenario:
        y = rng.standard_normal()

        for r, alloc in alloc_per_rating.items():
            alloc_per_issuer = alloc / n_issuers_per_rating

            eps = rng.standard_normal(n_issuers_per_rating)
            x = np.sqrt(rho) * y + np.sqrt(1.0 - rho) * eps

            # migrate to rating indices
            idx = migrate_many_to_indices(r, x, thresholds)  # ints in [0..7]

            # lookup bond values and compute bucket value
            # bond values are per 100 notional => multiply by alloc_per_issuer/100
            bucket_values = bond_t1[idx] * (alloc_per_issuer / 100.0)
            port_value += float(bucket_values.sum())

        v1[k] = port_value

    return v1
