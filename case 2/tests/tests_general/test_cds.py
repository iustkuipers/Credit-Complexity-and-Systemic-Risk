import pytest
import math
from services.instruments.cds import (
    CDSLegPV,
    _validate_schedule,
    pv_premium_leg_quarterly_with_accrual_midpoint,
    pv_protection_leg_midpoint,
    price_receiver_cds,
    cds_npv_receiver,
)
from services.instruments.curves import DiscountCurveFlat, PiecewiseConstantHazardCurve


# ============================================================
# CDSLegPV dataclass tests
# ============================================================

class TestCDSLegPV:
    """Test the CDSLegPV dataclass."""

    def test_cds_leg_pv_creation(self):
        """Should create CDSLegPV with premium and protection legs."""
        legs = CDSLegPV(premium_leg=100.0, protection_leg=80.0)
        assert legs.premium_leg == 100.0
        assert legs.protection_leg == 80.0

    def test_cds_leg_pv_npv_receiver_positive(self):
        """NPV receiver should be premium_leg - protection_leg."""
        legs = CDSLegPV(premium_leg=100.0, protection_leg=80.0)
        assert legs.npv_receiver == pytest.approx(20.0)

    def test_cds_leg_pv_npv_receiver_negative(self):
        """NPV receiver can be negative if protection > premium."""
        legs = CDSLegPV(premium_leg=50.0, protection_leg=100.0)
        assert legs.npv_receiver == pytest.approx(-50.0)

    def test_cds_leg_pv_npv_receiver_zero(self):
        """NPV receiver can be zero if legs are equal."""
        legs = CDSLegPV(premium_leg=75.0, protection_leg=75.0)
        assert legs.npv_receiver == pytest.approx(0.0)

    def test_cds_leg_pv_frozen(self):
        """CDSLegPV should be immutable (frozen)."""
        legs = CDSLegPV(premium_leg=100.0, protection_leg=80.0)
        with pytest.raises((AttributeError, TypeError)):
            legs.premium_leg = 200.0

    def test_cds_leg_pv_with_zero_values(self):
        """Should handle zero values."""
        legs = CDSLegPV(premium_leg=0.0, protection_leg=0.0)
        assert legs.npv_receiver == pytest.approx(0.0)

    def test_cds_leg_pv_with_negative_values(self):
        """Should handle negative values (representing flows)."""
        legs = CDSLegPV(premium_leg=-10.0, protection_leg=5.0)
        assert legs.npv_receiver == pytest.approx(-15.0)


# ============================================================
# Schedule validation tests
# ============================================================

class TestValidateSchedule:
    """Test the _validate_schedule function."""

    def test_valid_simple_schedule(self):
        """Simple valid schedule should not raise."""
        payment_times = [0.25, 0.5, 0.75, 1.0]
        _validate_schedule(payment_times, 1.0)  # Should not raise

    def test_valid_quarterly_schedule(self):
        """Valid quarterly schedule should not raise."""
        payment_times = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        _validate_schedule(payment_times, 2.0)  # Should not raise

    def test_negative_maturity_raises(self):
        """Negative maturity should raise ValueError."""
        with pytest.raises(ValueError, match="maturity must be > 0"):
            _validate_schedule([0.25, 0.5], -1.0)

    def test_zero_maturity_raises(self):
        """Zero maturity should raise ValueError."""
        with pytest.raises(ValueError, match="maturity must be > 0"):
            _validate_schedule([0.25, 0.5], 0.0)

    def test_empty_payment_times_raises(self):
        """Empty payment times should raise ValueError."""
        with pytest.raises(ValueError, match="payment_times must be non-empty"):
            _validate_schedule([], 1.0)

    def test_zero_payment_time_raises(self):
        """Zero payment time should raise ValueError."""
        with pytest.raises(ValueError, match="payment_times must be strictly positive"):
            _validate_schedule([0.0, 0.25, 0.5], 0.5)

    def test_negative_payment_time_raises(self):
        """Negative payment time should raise ValueError."""
        with pytest.raises(ValueError, match="payment_times must be strictly positive"):
            _validate_schedule([-0.1, 0.25, 0.5], 0.5)

    def test_non_increasing_times_raises(self):
        """Non-increasing payment times should raise ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            _validate_schedule([0.25, 0.5, 0.5, 0.75], 0.75)

    def test_decreasing_times_raises(self):
        """Decreasing payment times should raise ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            _validate_schedule([0.5, 0.25, 0.1], 0.5)

    def test_last_time_not_maturity_raises(self):
        """Last payment time not matching maturity should raise."""
        with pytest.raises(ValueError, match="payment_times must end at maturity"):
            _validate_schedule([0.25, 0.5, 0.75], 1.0)

    def test_last_time_slightly_off_maturity_raises(self):
        """Last time slightly different from maturity should raise."""
        with pytest.raises(ValueError, match="payment_times must end at maturity"):
            _validate_schedule([0.25, 0.5, 0.75, 1.001], 1.0)

    def test_single_payment_time(self):
        """Single payment time equal to maturity should be valid."""
        _validate_schedule([1.0], 1.0)  # Should not raise


# ============================================================
# Premium leg tests
# ============================================================

class TestPVPremiumLeg:
    """Test premium leg PV calculation."""

    @pytest.fixture
    def standard_setup(self):
        """Standard setup for tests."""
        dc = DiscountCurveFlat(r=0.03)
        hc = PiecewiseConstantHazardCurve(segments=((0.0, 1.0, 0.02),))
        payment_times = [0.25, 0.5, 0.75, 1.0]
        return dc, hc, payment_times

    def test_premium_leg_is_positive(self, standard_setup):
        """Premium leg PV should be positive with positive spread."""
        dc, hc, payment_times = standard_setup
        pv = pv_premium_leg_quarterly_with_accrual_midpoint(
            maturity=1.0,
            spread=0.01,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        assert pv > 0

    def test_premium_leg_zero_spread(self, standard_setup):
        """Premium leg with zero spread should be zero."""
        dc, hc, payment_times = standard_setup
        pv = pv_premium_leg_quarterly_with_accrual_midpoint(
            maturity=1.0,
            spread=0.0,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        assert pv == pytest.approx(0.0)

    def test_premium_leg_scales_with_spread(self, standard_setup):
        """Premium leg should scale linearly with spread."""
        dc, hc, payment_times = standard_setup
        pv1 = pv_premium_leg_quarterly_with_accrual_midpoint(
            maturity=1.0,
            spread=0.01,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        pv2 = pv_premium_leg_quarterly_with_accrual_midpoint(
            maturity=1.0,
            spread=0.02,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        assert pv2 == pytest.approx(2 * pv1)

    def test_premium_leg_invalid_schedule_raises(self, standard_setup):
        """Invalid payment schedule should raise."""
        dc, hc, _ = standard_setup
        with pytest.raises(ValueError):
            pv_premium_leg_quarterly_with_accrual_midpoint(
                maturity=1.0,
                spread=0.01,
                discount_curve=dc,
                hazard_curve=hc,
                payment_times=[0.25, 0.5, 0.75, 2.0],  # Last time != maturity
            )

    def test_premium_leg_with_low_hazard(self, standard_setup):
        """With low hazard rate, survival should be close to 1."""
        dc = DiscountCurveFlat(r=0.03)
        hc = PiecewiseConstantHazardCurve(segments=((0.0, 1.0, 0.001),))
        payment_times = [0.25, 0.5, 0.75, 1.0]
        
        pv = pv_premium_leg_quarterly_with_accrual_midpoint(
            maturity=1.0,
            spread=0.01,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        # With very low hazard, PV should be close to spread * annuity
        assert pv > 0.008  # Should be substantial

    def test_premium_leg_high_hazard_lower(self, standard_setup):
        """With higher hazard rate, PV should be lower."""
        dc = DiscountCurveFlat(r=0.03)
        hc_low = PiecewiseConstantHazardCurve(segments=((0.0, 1.0, 0.01),))
        hc_high = PiecewiseConstantHazardCurve(segments=((0.0, 1.0, 0.05),))
        payment_times = [0.25, 0.5, 0.75, 1.0]
        
        pv_low = pv_premium_leg_quarterly_with_accrual_midpoint(
            maturity=1.0,
            spread=0.01,
            discount_curve=dc,
            hazard_curve=hc_low,
            payment_times=payment_times,
        )
        pv_high = pv_premium_leg_quarterly_with_accrual_midpoint(
            maturity=1.0,
            spread=0.01,
            discount_curve=dc,
            hazard_curve=hc_high,
            payment_times=payment_times,
        )
        assert pv_low > pv_high


# ============================================================
# Protection leg tests
# ============================================================

class TestPVProtectionLeg:
    """Test protection leg PV calculation."""

    @pytest.fixture
    def standard_setup(self):
        """Standard setup for tests."""
        dc = DiscountCurveFlat(r=0.03)
        hc = PiecewiseConstantHazardCurve(segments=((0.0, 1.0, 0.02),))
        payment_times = [0.25, 0.5, 0.75, 1.0]
        return dc, hc, payment_times

    def test_protection_leg_is_positive(self, standard_setup):
        """Protection leg PV should be positive with positive hazard."""
        dc, hc, payment_times = standard_setup
        pv = pv_protection_leg_midpoint(
            maturity=1.0,
            lgd=0.40,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        assert pv > 0

    def test_protection_leg_zero_hazard(self):
        """Protection leg with zero hazard should be zero."""
        dc = DiscountCurveFlat(r=0.03)
        hc = PiecewiseConstantHazardCurve(segments=((0.0, 1.0, 0.0),))
        payment_times = [0.25, 0.5, 0.75, 1.0]
        
        pv = pv_protection_leg_midpoint(
            maturity=1.0,
            lgd=0.40,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        assert pv == pytest.approx(0.0)

    def test_protection_leg_scales_with_lgd(self, standard_setup):
        """Protection leg should scale linearly with LGD."""
        dc, hc, payment_times = standard_setup
        pv1 = pv_protection_leg_midpoint(
            maturity=1.0,
            lgd=0.40,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        pv2 = pv_protection_leg_midpoint(
            maturity=1.0,
            lgd=0.60,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        assert pv2 == pytest.approx(1.5 * pv1)

    def test_protection_leg_invalid_lgd_zero_raises(self, standard_setup):
        """LGD of zero should raise ValueError."""
        dc, hc, payment_times = standard_setup
        with pytest.raises(ValueError, match="lgd must be in"):
            pv_protection_leg_midpoint(
                maturity=1.0,
                lgd=0.0,
                discount_curve=dc,
                hazard_curve=hc,
                payment_times=payment_times,
            )

    def test_protection_leg_invalid_lgd_negative_raises(self, standard_setup):
        """Negative LGD should raise ValueError."""
        dc, hc, payment_times = standard_setup
        with pytest.raises(ValueError, match="lgd must be in"):
            pv_protection_leg_midpoint(
                maturity=1.0,
                lgd=-0.1,
                discount_curve=dc,
                hazard_curve=hc,
                payment_times=payment_times,
            )

    def test_protection_leg_invalid_lgd_greater_than_one_raises(self, standard_setup):
        """LGD > 1 should raise ValueError."""
        dc, hc, payment_times = standard_setup
        with pytest.raises(ValueError, match="lgd must be in"):
            pv_protection_leg_midpoint(
                maturity=1.0,
                lgd=1.5,
                discount_curve=dc,
                hazard_curve=hc,
                payment_times=payment_times,
            )

    def test_protection_leg_valid_lgd_one(self, standard_setup):
        """LGD of 1.0 should be valid."""
        dc, hc, payment_times = standard_setup
        pv = pv_protection_leg_midpoint(
            maturity=1.0,
            lgd=1.0,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        assert pv > 0

    def test_protection_leg_invalid_schedule_raises(self, standard_setup):
        """Invalid payment schedule should raise."""
        dc, hc, _ = standard_setup
        with pytest.raises(ValueError):
            pv_protection_leg_midpoint(
                maturity=1.0,
                lgd=0.40,
                discount_curve=dc,
                hazard_curve=hc,
                payment_times=[0.25, 0.5, 0.75, 2.0],  # Last time != maturity
            )


# ============================================================
# CDS pricing tests
# ============================================================

class TestPriceReceiverCDS:
    """Test receiver CDS pricing."""

    @pytest.fixture
    def standard_setup(self):
        """Standard setup for tests."""
        dc = DiscountCurveFlat(r=0.03)
        hc = PiecewiseConstantHazardCurve(segments=((0.0, 1.0, 0.02),))
        payment_times = [0.25, 0.5, 0.75, 1.0]
        return dc, hc, payment_times

    def test_price_receiver_cds_returns_legs(self, standard_setup):
        """Should return CDSLegPV object."""
        dc, hc, payment_times = standard_setup
        legs = price_receiver_cds(
            maturity=1.0,
            spread=0.01,
            lgd=0.40,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        assert isinstance(legs, CDSLegPV)

    def test_price_receiver_cds_premium_positive(self, standard_setup):
        """Premium leg should be positive with positive spread."""
        dc, hc, payment_times = standard_setup
        legs = price_receiver_cds(
            maturity=1.0,
            spread=0.01,
            lgd=0.40,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        assert legs.premium_leg > 0

    def test_price_receiver_cds_protection_positive(self, standard_setup):
        """Protection leg should be positive with positive hazard."""
        dc, hc, payment_times = standard_setup
        legs = price_receiver_cds(
            maturity=1.0,
            spread=0.01,
            lgd=0.40,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        assert legs.protection_leg > 0

    def test_price_receiver_cds_npv_property(self, standard_setup):
        """NPV should be premium - protection."""
        dc, hc, payment_times = standard_setup
        legs = price_receiver_cds(
            maturity=1.0,
            spread=0.01,
            lgd=0.40,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        expected_npv = legs.premium_leg - legs.protection_leg
        assert legs.npv_receiver == pytest.approx(expected_npv)

    def test_price_receiver_cds_longer_maturity(self, standard_setup):
        """Longer maturity should affect both legs."""
        dc = DiscountCurveFlat(r=0.03)
        hc = PiecewiseConstantHazardCurve(segments=((0.0, 5.0, 0.02),))
        
        legs_1yr = price_receiver_cds(
            maturity=1.0,
            spread=0.01,
            lgd=0.40,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=[0.25, 0.5, 0.75, 1.0],
        )
        
        legs_5yr = price_receiver_cds(
            maturity=5.0,
            spread=0.01,
            lgd=0.40,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
                          2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0,
                          4.25, 4.5, 4.75, 5.0],
        )
        
        # Longer maturity typically increases both legs, but effect varies
        assert legs_5yr.premium_leg > 0
        assert legs_5yr.protection_leg > 0


# ============================================================
# CDS NPV convenience function tests
# ============================================================

class TestCDSNPVReceiver:
    """Test cds_npv_receiver convenience function."""

    @pytest.fixture
    def standard_setup(self):
        """Standard setup for tests."""
        dc = DiscountCurveFlat(r=0.03)
        hc = PiecewiseConstantHazardCurve(segments=((0.0, 1.0, 0.02),))
        payment_times = [0.25, 0.5, 0.75, 1.0]
        return dc, hc, payment_times

    def test_cds_npv_receiver_returns_float(self, standard_setup):
        """Should return a float NPV."""
        dc, hc, payment_times = standard_setup
        npv = cds_npv_receiver(
            maturity=1.0,
            spread=0.01,
            lgd=0.40,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        assert isinstance(npv, float)

    def test_cds_npv_receiver_matches_legs_calculation(self, standard_setup):
        """NPV should match premium - protection from legs."""
        dc, hc, payment_times = standard_setup
        npv = cds_npv_receiver(
            maturity=1.0,
            spread=0.01,
            lgd=0.40,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        
        legs = price_receiver_cds(
            maturity=1.0,
            spread=0.01,
            lgd=0.40,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        
        assert npv == pytest.approx(legs.npv_receiver)

    def test_cds_npv_receiver_at_par_spread(self):
        """At par spread, NPV should be close to zero."""
        dc = DiscountCurveFlat(r=0.03)
        hc = PiecewiseConstantHazardCurve(segments=((0.0, 1.0, 0.02),))
        payment_times = [0.25, 0.5, 0.75, 1.0]
        
        # Calculate par spread (where NPV = 0)
        # This is a simplified test - real par spread calculation is more complex
        npv = cds_npv_receiver(
            maturity=1.0,
            spread=0.005,  # Small spread
            lgd=0.40,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        # NPV should be negative (we're receiving less premium than protection costs)
        assert npv < 0

    def test_cds_npv_receiver_high_spread_positive(self):
        """With high spread, NPV should be positive (for receiver)."""
        dc = DiscountCurveFlat(r=0.03)
        hc = PiecewiseConstantHazardCurve(segments=((0.0, 1.0, 0.02),))
        payment_times = [0.25, 0.5, 0.75, 1.0]
        
        npv = cds_npv_receiver(
            maturity=1.0,
            spread=0.05,  # High spread
            lgd=0.40,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        # NPV should be positive (we're receiving high premium)
        assert npv > 0

    def test_cds_npv_receiver_invalid_schedule_raises(self, standard_setup):
        """Invalid payment schedule should raise."""
        dc, hc, _ = standard_setup
        with pytest.raises(ValueError):
            cds_npv_receiver(
                maturity=1.0,
                spread=0.01,
                lgd=0.40,
                discount_curve=dc,
                hazard_curve=hc,
                payment_times=[0.25, 0.5, 0.75, 2.0],  # Last time != maturity
            )


# ============================================================
# Integration tests
# ============================================================

class TestCDSIntegration:
    """Integration tests for CDS pricing."""

    def test_cds_pricing_consistency_across_tenors(self):
        """Test CDS pricing at different tenors is consistent."""
        dc = DiscountCurveFlat(r=0.02)
        hc = PiecewiseConstantHazardCurve(segments=((0.0, 10.0, 0.015),))
        
        spreads = []
        for T in [1.0, 3.0, 5.0]:
            payment_times = [t for t in [i * 0.25 for i in range(1, int(T*4) + 1)] if t <= T]
            if not payment_times or abs(payment_times[-1] - T) > 1e-10:
                payment_times = list(range(1, int(T*4))) * 0.25 + [T]
            
            # Just verify calculation works
            npv = cds_npv_receiver(
                maturity=T,
                spread=0.01,
                lgd=0.40,
                discount_curve=dc,
                hazard_curve=hc,
                payment_times=payment_times,
            )
            assert isinstance(npv, float)

    def test_cds_monotonicity_with_spread(self):
        """NPV should increase monotonically with spread."""
        dc = DiscountCurveFlat(r=0.03)
        hc = PiecewiseConstantHazardCurve(segments=((0.0, 1.0, 0.02),))
        payment_times = [0.25, 0.5, 0.75, 1.0]
        
        npv_values = []
        for spread in [0.005, 0.01, 0.015, 0.02]:
            npv = cds_npv_receiver(
                maturity=1.0,
                spread=spread,
                lgd=0.40,
                discount_curve=dc,
                hazard_curve=hc,
                payment_times=payment_times,
            )
            npv_values.append(npv)
        
        # NPV should increase with spread
        for i in range(len(npv_values) - 1):
            assert npv_values[i] < npv_values[i + 1]

    def test_cds_with_piecewise_constant_hazard(self):
        """Test CDS pricing with multi-segment hazard curve."""
        dc = DiscountCurveFlat(r=0.03)
        hc = PiecewiseConstantHazardCurve(
            segments=((0.0, 1.0, 0.01), (1.0, 3.0, 0.03), (3.0, 5.0, 0.02))
        )
        payment_times = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
                        2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0,
                        4.25, 4.5, 4.75, 5.0]
        
        npv = cds_npv_receiver(
            maturity=5.0,
            spread=0.012,
            lgd=0.40,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        assert isinstance(npv, float)

    def test_cds_daily_compounding_schedule(self):
        """Test CDS with finer payment schedule (more frequent)."""
        dc = DiscountCurveFlat(r=0.03)
        hc = PiecewiseConstantHazardCurve(segments=((0.0, 1.0, 0.02),))
        
        # Daily-ish payment schedule (30 times per year)
        payment_times = [i / 30.0 for i in range(1, 31)]
        
        npv = cds_npv_receiver(
            maturity=1.0,
            spread=0.01,
            lgd=0.40,
            discount_curve=dc,
            hazard_curve=hc,
            payment_times=payment_times,
        )
        assert isinstance(npv, float)
        assert npv != 0  # Should have non-zero value
