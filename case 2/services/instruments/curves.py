# services/instruments/curves.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import math


# ============================================================
# Discounting
# ============================================================

@dataclass(frozen=True)
class DiscountCurveFlat:
    """
    Flat interest-rate curve.

    Note: the case states 'annually compounded' rates, but the CDS pricing
    formula provided uses exp(-r * t). We will follow the case formula
    consistently throughout pricing/stripping.
    """
    r: float  # e.g. 0.03

    def df(self, t: float) -> float:
        """Discount factor P(0,t)."""
        if t < 0:
            raise ValueError("t must be non-negative")
        return math.exp(-self.r * t)


# ============================================================
# Hazard / survival
# ============================================================

@dataclass(frozen=True)
class PiecewiseConstantHazardCurve:
    """
    Piecewise-constant hazard rate curve.

    segments: list of (t_start, t_end, lambda)
      - t_start inclusive, t_end exclusive by convention
      - last segment should cover required horizon
    """
    segments: Tuple[Tuple[float, float, float], ...]

    def _validate(self) -> None:
        if not self.segments:
            raise ValueError("Hazard curve must contain at least one segment.")
        prev_end = None
        for (a, b, lam) in self.segments:
            if a < 0 or b <= a:
                raise ValueError(f"Invalid segment [{a},{b}).")
            if lam < 0:
                raise ValueError("Hazard rates must be non-negative.")
            if prev_end is not None and abs(a - prev_end) > 1e-12:
                raise ValueError("Segments must be contiguous and ordered.")
            prev_end = b

    def integrated_hazard(self, t: float) -> float:
        """
        Compute ∫_0^t λ(u) du under piecewise-constant segments.
        """
        if t < 0:
            raise ValueError("t must be non-negative")
        self._validate()

        ih = 0.0
        remaining = t

        for (a, b, lam) in self.segments:
            if remaining <= 0:
                break
            # overlap of [a,b) with [0,t]
            seg_len = max(0.0, min(b, t) - a)
            if seg_len > 0:
                ih += lam * seg_len

        # If t exceeds last segment end, that's a modeling error
        last_end = self.segments[-1][1]
        if t - last_end > 1e-12:
            raise ValueError(
                f"Requested t={t} exceeds hazard curve horizon {last_end}. "
                "Extend segments to cover required maturity."
            )
        return ih

    def survival(self, t: float) -> float:
        """Survival probability S(t) = Q(τ > t)."""
        ih = self.integrated_hazard(t)
        return math.exp(-ih)

    def default_prob_between(self, t0: float, t1: float) -> float:
        """
        Forward default probability between (t0, t1]:
        Q(τ > t0) - Q(τ > t1)
        """
        if t1 < t0:
            raise ValueError("t1 must be >= t0")
        return self.survival(t0) - self.survival(t1)

    def forward_hazard(self, t0: float, t1: float) -> float:
        """
        Implied constant forward hazard over (t0, t1] from survival curve:
          λ_fwd = - ln(S(t1)/S(t0)) / (t1 - t0)
        """
        if t1 <= t0:
            raise ValueError("t1 must be > t0")
        s0 = self.survival(t0)
        s1 = self.survival(t1)
        if s1 <= 0 or s0 <= 0:
            raise ValueError("Survival probabilities must be positive.")
        return -math.log(s1 / s0) / (t1 - t0)

    def average_hazard(self, t: float) -> float:
        """
        Average hazard over [0,t]:
          λ_avg(t) = (1/t) ∫_0^t λ(u) du
        """
        if t <= 0:
            raise ValueError("t must be > 0")
        return self.integrated_hazard(t) / t


# ============================================================
# Time grids
# ============================================================

def build_payment_times(maturity: float, payments_per_year: int) -> List[float]:
    """
    Build payment times  (e.g., quarterly) up to maturity.
    Example: maturity=1, freq=4 -> [0.25, 0.5, 0.75, 1.0]
    """
    if maturity <= 0:
        raise ValueError("maturity must be > 0")
    if payments_per_year <= 0:
        raise ValueError("payments_per_year must be > 0")

    dt = 1.0 / payments_per_year
    n = int(round(maturity / dt))
    # ensure exact maturity is included
    times = [round((i + 1) * dt, 12) for i in range(n)]
    if abs(times[-1] - maturity) > 1e-10:
        # if maturity isn't an integer multiple of dt, append it
        times.append(maturity)
    return times
