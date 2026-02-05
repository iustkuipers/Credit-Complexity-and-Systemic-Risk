# simulation.py
from __future__ import annotations

import numpy as np

from data import BOND_VALUE_T1, BOND_VALUE_T0, RATINGS, START_RATINGS
try:
    from .model import precompute_thresholds, migrate_many_to_indices
except ImportError:
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
    Simulate portfolio forward values V1 (t=1) under migration/default using
    the one-factor model and credit migration.

    Args:
        portfolio: {rating: weight} at t=0
        total_notional: total invested amount
        n_issuers_per_rating: number of issuers per rating (concentration control)
        rho: asset correlation [0,1]
        N: number of Monte Carlo scenarios
        seed: random seed
        thresholds: precomputed thresholds (optional)

    Returns:
        v1: np.ndarray of shape (N,) portfolio values at t=1
    """
    if thresholds is None:
        thresholds = precompute_thresholds()

    rng = np.random.default_rng(seed)

    # Precompute allocations per rating bucket (currency units)
    alloc_per_rating = {r: w * total_notional for r, w in portfolio.items()}

    v1 = np.empty(N, dtype=float)

    for k in range(N):
        port_value = 0.0

        # One common factor per scenario
        y = rng.standard_normal()

        for r, alloc in alloc_per_rating.items():
            alloc_per_issuer = alloc / n_issuers_per_rating

            # Simulate idiosyncratic asset returns
            eps = rng.standard_normal(n_issuers_per_rating)
            x = np.sqrt(rho) * y + np.sqrt(1 - rho) * eps

            # Migrate to rating indices
            idx = migrate_many_to_indices(r, x, thresholds)

            # Compute bond returns = BOND_VALUE_T1[new_rating] / BOND_VALUE_T0[old_rating]
            bond_returns = np.array([BOND_VALUE_T1[RATINGS[i]] / BOND_VALUE_T0[r] for i in idx])

            # Portfolio contribution = return * allocated value at t=0
            bucket_values = bond_returns * alloc_per_issuer
            port_value += bucket_values.sum()

            # === DEBUG ===
            
        v1[k] = port_value

    return v1

