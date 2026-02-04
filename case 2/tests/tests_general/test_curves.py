# tests/test_curves.py

import math
import pytest

from services.instruments.curves import (
    DiscountCurveFlat,
    PiecewiseConstantHazardCurve,
    build_payment_times,
)


# ============================================================
# DiscountCurveFlat
# ============================================================

def test_discount_curve_flat_df_basic():
    dc = DiscountCurveFlat(r=0.03)
    assert dc.df(0.0) == pytest.approx(1.0)
    assert dc.df(1.0) == pytest.approx(math.exp(-0.03 * 1.0))


def test_discount_curve_flat_df_negative_time_raises():
    dc = DiscountCurveFlat(r=0.03)
    with pytest.raises(ValueError):
        dc.df(-0.1)


# ============================================================
# PiecewiseConstantHazardCurve validation
# ============================================================

def test_hazard_curve_validation_empty_segments_raises():
    hc = PiecewiseConstantHazardCurve(segments=tuple())
    with pytest.raises(ValueError):
        hc.integrated_hazard(0.5)


def test_hazard_curve_validation_non_contiguous_raises():
    # gap between 1.0 and 2.0
    hc = PiecewiseConstantHazardCurve(
        segments=((0.0, 1.0, 0.01), (2.0, 3.0, 0.02))
    )
    with pytest.raises(ValueError):
        hc.integrated_hazard(2.5)


def test_hazard_curve_validation_negative_lambda_raises():
    hc = PiecewiseConstantHazardCurve(
        segments=((0.0, 1.0, -0.01),)
    )
    with pytest.raises(ValueError):
        hc.integrated_hazard(0.5)


def test_hazard_curve_validation_invalid_segment_bounds_raises():
    # end <= start
    hc = PiecewiseConstantHazardCurve(
        segments=((1.0, 1.0, 0.01),)
    )
    with pytest.raises(ValueError):
        hc.integrated_hazard(0.5)


# ============================================================
# PiecewiseConstantHazardCurve computations
# ============================================================

def test_integrated_hazard_single_segment():
    hc = PiecewiseConstantHazardCurve(segments=((0.0, 10.0, 0.02),))
    assert hc.integrated_hazard(0.0) == pytest.approx(0.0)
    assert hc.integrated_hazard(3.0) == pytest.approx(0.02 * 3.0)


def test_integrated_hazard_piecewise():
    # lambda = 0.01 on [0,1), 0.03 on [1,3), 0.02 on [3,5)
    hc = PiecewiseConstantHazardCurve(
        segments=((0.0, 1.0, 0.01), (1.0, 3.0, 0.03), (3.0, 5.0, 0.02))
    )
    # ∫_0^4 = 0.01*1 + 0.03*2 + 0.02*1
    expected = 0.01 * 1.0 + 0.03 * 2.0 + 0.02 * 1.0
    assert hc.integrated_hazard(4.0) == pytest.approx(expected)


def test_integrated_hazard_beyond_horizon_raises():
    hc = PiecewiseConstantHazardCurve(segments=((0.0, 1.0, 0.02),))
    with pytest.raises(ValueError):
        hc.integrated_hazard(1.5)


def test_survival_probability_matches_definition():
    hc = PiecewiseConstantHazardCurve(segments=((0.0, 10.0, 0.02),))
    t = 4.0
    expected = math.exp(-0.02 * t)
    assert hc.survival(t) == pytest.approx(expected)


def test_default_prob_between_is_survival_difference():
    hc = PiecewiseConstantHazardCurve(segments=((0.0, 10.0, 0.02),))
    t0, t1 = 2.0, 5.0
    expected = hc.survival(t0) - hc.survival(t1)
    assert hc.default_prob_between(t0, t1) == pytest.approx(expected)


def test_default_prob_between_invalid_order_raises():
    hc = PiecewiseConstantHazardCurve(segments=((0.0, 10.0, 0.02),))
    with pytest.raises(ValueError):
        hc.default_prob_between(3.0, 2.0)


def test_forward_hazard_constant_segment_equals_lambda():
    lam = 0.025
    hc = PiecewiseConstantHazardCurve(segments=((0.0, 10.0, lam),))
    # For constant hazard, implied forward hazard over any interval equals lam
    assert hc.forward_hazard(1.0, 4.0) == pytest.approx(lam, rel=1e-12, abs=1e-12)


def test_forward_hazard_invalid_interval_raises():
    hc = PiecewiseConstantHazardCurve(segments=((0.0, 10.0, 0.02),))
    with pytest.raises(ValueError):
        hc.forward_hazard(2.0, 2.0)
    with pytest.raises(ValueError):
        hc.forward_hazard(3.0, 2.0)


def test_average_hazard_constant_segment_equals_lambda():
    lam = 0.017
    hc = PiecewiseConstantHazardCurve(segments=((0.0, 10.0, lam),))
    assert hc.average_hazard(5.0) == pytest.approx(lam)


def test_average_hazard_nonpositive_time_raises():
    hc = PiecewiseConstantHazardCurve(segments=((0.0, 10.0, 0.02),))
    with pytest.raises(ValueError):
        hc.average_hazard(0.0)
    with pytest.raises(ValueError):
        hc.average_hazard(-1.0)


# ============================================================
# build_payment_times
# ============================================================

def test_build_payment_times_quarterly_one_year():
    times = build_payment_times(maturity=1.0, payments_per_year=4)
    assert times == [0.25, 0.5, 0.75, 1.0]


def test_build_payment_times_invalid_inputs_raise():
    with pytest.raises(ValueError):
        build_payment_times(maturity=0.0, payments_per_year=4)
    with pytest.raises(ValueError):
        build_payment_times(maturity=1.0, payments_per_year=0)

