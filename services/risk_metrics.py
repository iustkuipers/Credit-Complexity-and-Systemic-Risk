# risk_metrics.py
from __future__ import annotations

import numpy as np


def values_to_losses(v0: float, v1: np.ndarray) -> np.ndarray:
    """
    Convert simulated portfolio values at t=1 into losses.

    Convention (lock this down once):
        Loss = V0 - V1
    So positive losses are bad outcomes.

    Args:
        v0: deterministic current portfolio value at t=0 (scalar)
        v1: simulated portfolio values at t=1 (array)

    Returns:
        losses: np.ndarray same shape as v1
    """
    v1 = np.asarray(v1, dtype=float)
    return float(v0) - v1


def expected_value(values: np.ndarray) -> float:
    """Mean of portfolio values (e.g., E[V1])."""
    values = np.asarray(values, dtype=float)
    return float(values.mean())


def var(losses: np.ndarray, alpha: float) -> float:
    """
    Value-at-Risk at confidence level alpha, computed on the LOSS distribution.

    Example:
        alpha = 0.995 -> 99.5% VaR

    Returns:
        VaR_alpha (a loss threshold)
    """
    _validate_alpha(alpha)
    losses = np.asarray(losses, dtype=float)
    return float(np.quantile(losses, alpha))


def es(losses: np.ndarray, alpha: float) -> float:
    """
    Expected Shortfall (Conditional VaR) at confidence level alpha on LOSSES.

    Definition:
        ES_alpha = E[ Loss | Loss >= VaR_alpha ]

    Returns:
        ES_alpha (mean tail loss)
    """
    _validate_alpha(alpha)
    losses = np.asarray(losses, dtype=float)
    v = np.quantile(losses, alpha)
    tail = losses[losses >= v]
    # In extreme small samples tail can be empty due to numerical issues; guard it.
    if tail.size == 0:
        return float(v)
    return float(tail.mean())


def summarize_losses(losses: np.ndarray, alphas: tuple[float, ...] = (0.90, 0.995)) -> dict:
    """
    Convenience summary for required metrics given a loss sample.

    Returns dict with:
        - mean_loss
        - for each alpha: var_{alpha}, es_{alpha}
    """
    losses = np.asarray(losses, dtype=float)
    out: dict[str, float] = {"mean_loss": float(losses.mean())}

    for a in alphas:
        out[f"var_{a}"] = var(losses, a)
        out[f"es_{a}"] = es(losses, a)

    return out


def summarize_case_metrics(v0: float, v1: np.ndarray, alphas: tuple[float, ...] = (0.90, 0.995)) -> dict:
    """
    Case-friendly summary combining:
      - expected portfolio value at t=1
      - VaR/ES on losses

    Returns dict with:
        expected_value, and for each alpha: var_{alpha}, es_{alpha}
    """
    v1 = np.asarray(v1, dtype=float)
    losses = values_to_losses(v0, v1)

    out: dict[str, float] = {"expected_value": float(v1.mean())}
    for a in alphas:
        out[f"var_{a}"] = var(losses, a)
        out[f"es_{a}"] = es(losses, a)
    return out


def es_over_var_ratio(losses: np.ndarray, alpha: float = 0.995) -> float:
    """
    Compute ES(alpha) / VaR(alpha) on losses.
    Used in Question 3 tail-fatness discussion.

    Returns:
        ratio (float). If VaR is ~0, returns np.nan.
    """
    v = var(losses, alpha)
    if np.isclose(v, 0.0):
        return float("nan")
    return es(losses, alpha) / v


def _validate_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1). Got {alpha}")


# Quick sanity check if run directly
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fake_losses = rng.normal(loc=0.0, scale=1.0, size=100000)
    print("VaR 99.5%:", var(fake_losses, 0.995))
    print("ES 99.5%:", es(fake_losses, 0.995))
    print("ES/VaR:", es_over_var_ratio(fake_losses, 0.995))
