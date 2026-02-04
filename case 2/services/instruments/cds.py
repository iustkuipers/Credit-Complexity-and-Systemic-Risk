# services/instruments/cds.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import math

from services.instruments.curves import DiscountCurveFlat, PiecewiseConstantHazardCurve


@dataclass(frozen=True)
class CDSLegPV:
    """
    Present values of CDS legs at inception.
    """
    premium_leg: float
    protection_leg: float

    @property
    def npv_receiver(self) -> float:
        """
        Receiver (fixed) CDS NPV = PV(premium leg) - PV(protection leg)
        This matches the case's formula structure:
          CDS(0,T, R, LGD; λ) = PremiumLeg - ProtectionLeg
        """
        return self.premium_leg - self.protection_leg


def _validate_schedule(payment_times: Sequence[float], maturity: float) -> None:
    if maturity <= 0:
        raise ValueError("maturity must be > 0")
    if not payment_times:
        raise ValueError("payment_times must be non-empty")
    if any(t <= 0 for t in payment_times):
        raise ValueError("payment_times must be strictly positive")
    if any(payment_times[i] >= payment_times[i + 1] for i in range(len(payment_times) - 1)):
        raise ValueError("payment_times must be strictly increasing")
    if abs(payment_times[-1] - maturity) > 1e-10:
        raise ValueError("payment_times must end at maturity (last element equals maturity).")


def pv_premium_leg_quarterly_with_accrual_midpoint(
    *,
    maturity: float,
    spread: float,
    discount_curve: DiscountCurveFlat,
    hazard_curve: PiecewiseConstantHazardCurve,
    payment_times: Sequence[float],
) -> float:
    """
    Premium leg PV under the case discretization:

    Premium PV = R * [ Sum_i exp(-r*T_i) * Δ_i * Q(τ > T_i)
                     + Sum_i exp(-r*T_mid_i) * (Q(τ > T_{i-1}) - Q(τ > T_i)) * (Δ_i/2) ]

    where:
      T_mid_i = (T_i + T_{i-1})/2
      Δ_i = T_i - T_{i-1}
      T_0 = 0

    Parameters
    ----------
    maturity : float
    spread : float
        CDS spread in decimal (e.g. 100 bps -> 0.01)
    discount_curve : DiscountCurveFlat
    hazard_curve : PiecewiseConstantHazardCurve
    payment_times : Sequence[float]
        Quarterly payment times ending exactly at maturity

    Returns
    -------
    float
        PV of premium leg
    """
    _validate_schedule(payment_times, maturity)

    pv_regular = 0.0
    pv_accrual = 0.0

    t_prev = 0.0
    s_prev = 1.0  # Q(τ > 0) = 1

    for t_i in payment_times:
        dt = t_i - t_prev
        if dt <= 0:
            raise ValueError("Non-increasing payment schedule detected.")

        s_i = hazard_curve.survival(t_i)

        # Regular premium at T_i, conditional on survival to T_i
        pv_regular += discount_curve.df(t_i) * dt * s_i

        # Accrued premium in default interval (T_{i-1}, T_i] via midpoint approximation
        t_mid = 0.5 * (t_prev + t_i)
        pv_accrual += discount_curve.df(t_mid) * (s_prev - s_i) * (dt / 2.0)

        # roll
        t_prev = t_i
        s_prev = s_i

    return float(spread) * (pv_regular + pv_accrual)


def pv_protection_leg_midpoint(
    *,
    maturity: float,
    lgd: float,
    discount_curve: DiscountCurveFlat,
    hazard_curve: PiecewiseConstantHazardCurve,
    payment_times: Sequence[float],
) -> float:
    """
    Protection leg PV under the case discretization (midpoint discounting):

    Protection PV = LGD * Sum_i exp(-r*T_mid_i) * (Q(τ > T_{i-1}) - Q(τ > T_i))

    Parameters
    ----------
    maturity : float
    lgd : float
        Loss given default (e.g. 0.40)
    discount_curve : DiscountCurveFlat
    hazard_curve : PiecewiseConstantHazardCurve
    payment_times : Sequence[float]
        Quarterly grid ending exactly at maturity

    Returns
    -------
    float
        PV of protection leg
    """
    if not (0 < lgd <= 1):
        raise ValueError("lgd must be in (0,1].")

    _validate_schedule(payment_times, maturity)

    pv = 0.0

    t_prev = 0.0
    s_prev = 1.0

    for t_i in payment_times:
        s_i = hazard_curve.survival(t_i)
        t_mid = 0.5 * (t_prev + t_i)

        pv += discount_curve.df(t_mid) * (s_prev - s_i)

        t_prev = t_i
        s_prev = s_i

    return float(lgd) * pv


def price_receiver_cds(
    *,
    maturity: float,
    spread: float,
    lgd: float,
    discount_curve: DiscountCurveFlat,
    hazard_curve: PiecewiseConstantHazardCurve,
    payment_times: Sequence[float],
) -> CDSLegPV:
    """
    Compute the receiver CDS legs PV and NPV (premium - protection)
    using the discretization given in the case.

    Returns
    -------
    CDSLegPV
        Contains PV(premium leg) and PV(protection leg).
    """
    prem = pv_premium_leg_quarterly_with_accrual_midpoint(
        maturity=maturity,
        spread=spread,
        discount_curve=discount_curve,
        hazard_curve=hazard_curve,
        payment_times=payment_times,
    )
    prot = pv_protection_leg_midpoint(
        maturity=maturity,
        lgd=lgd,
        discount_curve=discount_curve,
        hazard_curve=hazard_curve,
        payment_times=payment_times,
    )
    return CDSLegPV(premium_leg=prem, protection_leg=prot)


def cds_npv_receiver(
    *,
    maturity: float,
    spread: float,
    lgd: float,
    discount_curve: DiscountCurveFlat,
    hazard_curve: PiecewiseConstantHazardCurve,
    payment_times: Sequence[float],
) -> float:
    """
    Convenience wrapper: receiver CDS NPV = PV(premium) - PV(protection).
    """
    legs = price_receiver_cds(
        maturity=maturity,
        spread=spread,
        lgd=lgd,
        discount_curve=discount_curve,
        hazard_curve=hazard_curve,
        payment_times=payment_times,
    )
    return legs.npv_receiver
