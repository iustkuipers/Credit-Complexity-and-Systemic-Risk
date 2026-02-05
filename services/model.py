# model.py
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Support importing model in three contexts:
# 1. As part of the `services` package: from services.model import ...
# 2. As a standalone module in tests: sys.path includes 'services/', then from model import ...
# 3. Via main.py which uses package imports
try:
    from .data import (
        RATINGS,
        RATING_TO_IDX,
        START_RATINGS,
        START_RATING_TO_ROW,
        TRANSITION,
    )
except ImportError:
    try:
        from services.data import (
            RATINGS,
            RATING_TO_IDX,
            START_RATINGS,
            START_RATING_TO_ROW,
            TRANSITION,
        )
    except ImportError:
        from data import (
            RATINGS,
            RATING_TO_IDX,
            START_RATINGS,
            START_RATING_TO_ROW,
            TRANSITION,
        )

# Numerical safety to avoid norm.ppf(0)=-inf and norm.ppf(1)=+inf everywhere
_EPS = 1e-12

def precompute_thresholds(transition: np.ndarray = TRANSITION) -> np.ndarray:
    """
    Convert each transition row into Z-thresholds used to map asset return X -> next rating.

    For a given current rating row with probabilities p_k over next ratings k in RATINGS order:
        cdf_k = sum_{j<=k} p_j
        thr_k = Phi^{-1}(cdf_k)

    Interpretation (credit convention):
    - Bad outcomes live in the LEFT tail (low X).
    - We assign the first rating k such that X <= thr_k.

    Returns:
        thresholds: shape (len(START_RATINGS), len(RATINGS))
            thresholds[row_of_current_rating, k] is the cutoff to end up in RATINGS[k].
    """
    if transition.shape != (len(START_RATINGS), len(RATINGS)):
        raise ValueError(
            f"transition must have shape ({len(START_RATINGS)},{len(RATINGS)}); got {transition.shape}"
        )

    cdf = np.cumsum(transition, axis=1)

    # clip to avoid infinities except the last one which we *want* to be +inf
    cdf = np.clip(cdf, _EPS, 1.0 - _EPS)

    thresholds = norm.ppf(cdf)

    # ensure last cutoff is +inf so every X maps to some rating
    thresholds[:, -1] = np.inf
    return thresholds


def migrate_one(current_rating: str, x: float, thresholds: np.ndarray) -> str:
    """
    Map a single issuer asset return x to a migrated rating.

    Args:
        current_rating: one of START_RATINGS (AAA..CCC). 'D' is not a start state in this case.
        x: simulated asset return
        thresholds: precomputed thresholds from precompute_thresholds()

    Returns:
        migrated rating (string from RATINGS)
    """
    row = _row_index(current_rating)
    thr = thresholds[row]

    # first index k such that x <= thr[k]
    k = int(np.searchsorted(thr, x, side="right"))
    k = min(k, len(RATINGS) - 1)
    return RATINGS[k]


def migrate_many(current_rating: str, x_vec: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Vectorized migration: map many issuer asset returns to migrated ratings.

    Returns:
        np.ndarray of dtype '<U...' with shape x_vec.shape containing rating strings.
    """
    row = _row_index(current_rating)
    thr = thresholds[row]

    k = np.searchsorted(thr, x_vec, side="right")
    k = np.minimum(k, len(RATINGS) - 1)

    # map indices to rating labels
    ratings_arr = np.array(RATINGS, dtype="U4")
    return ratings_arr[k]


def migrate_many_to_indices(current_rating: str, x_vec: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Same as migrate_many, but returns integer rating indices (0..len(RATINGS)-1).
    Useful for fast value lookups.
    """
    row = _row_index(current_rating)
    thr = thresholds[row]
    k = np.searchsorted(thr, x_vec, side="right")
    return np.minimum(k, len(RATINGS) - 1).astype(int)

def _row_index(current_rating: str) -> int:
    if current_rating not in START_RATINGS:
        raise ValueError(
            f"current_rating must be one of {START_RATINGS}. "
            f"Got '{current_rating}'. Note: 'D' is not a start state in the transition matrix."
        )
    return START_RATING_TO_ROW[current_rating]


# Optional: quick sanity smoke-test (run: python model.py)
if __name__ == "__main__":
    thr = precompute_thresholds()

    # sanity: very negative X should tend to worse ratings; very positive X -> upgrades
    test_x = np.array([-4.0, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0])
    migrated = migrate_many("BBB", test_x, thr)

    print("X:", test_x)
    print("BBB ->", migrated.tolist())
