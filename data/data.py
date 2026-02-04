# data.py
from __future__ import annotations
import numpy as np

# =============================================================================
# Canonical rating order (use everywhere)
# Best -> Worst
# =============================================================================
RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
RATING_TO_IDX = {r: i for i, r in enumerate(RATINGS)}

# Ratings with transition rows (AAA..CCC)
START_RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
START_RATING_TO_ROW = {r: i for i, r in enumerate(START_RATINGS)}

# =============================================================================
# One-period Migration & Default Transition Matrix (Case 1, Table 1)
# Rows: current rating in START_RATINGS order
# Cols: next rating in RATINGS order
# =============================================================================
TRANSITION = np.array(
    [
        # AAA
        [0.91115, 0.08179, 0.00607, 0.00072, 0.00024, 0.00003, 0.00000, 0.00000],
        # AA
        [0.00844, 0.89626, 0.08954, 0.00437, 0.00064, 0.00036, 0.00018, 0.00021],
        # A
        [0.00055, 0.02595, 0.91138, 0.05509, 0.00499, 0.00107, 0.00045, 0.00052],
        # BBB
        [0.00031, 0.00147, 0.04289, 0.90584, 0.03898, 0.00708, 0.00175, 0.00168],
        # BB
        [0.00007, 0.00044, 0.00446, 0.06741, 0.83274, 0.07667, 0.00895, 0.00926],
        # B
        [0.00008, 0.00031, 0.00150, 0.00490, 0.05373, 0.82531, 0.07894, 0.03523],
        # CCC
        [0.00000, 0.00015, 0.00023, 0.00091, 0.00388, 0.07630, 0.83035, 0.08818],
    ],
    dtype=float,
)

# =============================================================================
# Bond values (per 100 notional) — Case 1, Table 2
# =============================================================================
BOND_VALUE_T0 = {
    "AAA": 99.40,
    "AA": 98.39,
    "A": 97.22,
    "BBB": 92.79,
    "BB": 90.11,
    "B": 86.60,
    "CCC": 77.16,
    # no "D" at t=0 in the case
}

BOND_VALUE_T1 = {
    "AAA": 99.50,
    "AA": 98.51,
    "A": 97.53,
    "BBB": 92.77,
    "BB": 90.48,
    "B": 88.25,
    "CCC": 77.88,
    "D": 60.00,
}

# =============================================================================
# Portfolio helpers: you "push" any portfolio into the engine
# =============================================================================
def make_portfolio(weights: dict[str, float], *, normalize: bool = False) -> dict[str, float]:
    """
    Validate/construct a portfolio {rating: weight}.
    - Must not include 'D' at t=0
    - Non-negative weights
    - Sum to 1 unless normalize=True
    """
    if not weights:
        raise ValueError("Portfolio weights cannot be empty.")

    cleaned: dict[str, float] = {}
    for r, w in weights.items():
        if r not in RATINGS:
            raise ValueError(f"Unknown rating '{r}'. Allowed: {RATINGS}")
        if r == "D":
            raise ValueError("Portfolio must not include 'D' at t=0.")
        w = float(w)
        if w < 0:
            raise ValueError(f"Negative weight for rating '{r}': {w}")
        if w > 0:
            cleaned[r] = w

    s = sum(cleaned.values())
    if s <= 0:
        raise ValueError("Portfolio has zero total weight after cleaning.")

    if normalize:
        return {r: w / s for r, w in cleaned.items()}

    if not np.isclose(s, 1.0, atol=1e-12):
        raise ValueError(f"Portfolio weights must sum to 1.0 (got {s}). "
                         f"Use normalize=True to rescale automatically.")
    return cleaned


def portfolio_value_t0(portfolio: dict[str, float], total_notional: float) -> float:
    """
    Deterministic current value (t=0) in the same currency units as total_notional.
    total_notional is an INPUT, not stored globally.
    """
    v0 = 0.0
    for r, w in portfolio.items():
        if r not in BOND_VALUE_T0:
            raise ValueError(f"Missing t=0 bond value for rating '{r}'.")
        v0 += w * total_notional * (BOND_VALUE_T0[r] / 100.0)
    return v0


def validate_case_primitives() -> None:
    """Sanity checks for transition matrix + bond values."""
    if TRANSITION.shape != (7, 8):
        raise ValueError(f"TRANSITION must be shape (7,8); got {TRANSITION.shape}")

    row_sums = TRANSITION.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(f"Each transition row must sum to 1. Row sums: {row_sums}")

    for r in START_RATINGS:
        if r not in BOND_VALUE_T0:
            raise ValueError(f"Missing t=0 bond value for rating {r}")
    for r in RATINGS:
        if r not in BOND_VALUE_T1:
            raise ValueError(f"Missing t=1 bond value for rating {r}")


if __name__ == "__main__":
    validate_case_primitives()
    p = make_portfolio({"AAA": 0.6, "AA": 0.3, "BBB": 0.1})
    print("Portfolio OK:", p)
    print("Example V0 (notional=1500):", portfolio_value_t0(p, total_notional=1500.0))
