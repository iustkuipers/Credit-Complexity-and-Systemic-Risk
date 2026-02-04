# Q3 Validation Report (7Y CDS Audit)

- Total checks: **4**
- Passed: **4**
- Failed: **0**

All checks passed.

## Audit numbers (7Y)

- PV Premium Leg: **0.068550054470**
- PV Protection Leg: **0.068550054470**
- NPV (Premium − Protection): **-2.056688153118e-14**
- |NPV|: **2.056688153118e-14**
- Tolerance: **1.0e-06**
- Passed: **True**

## Check details

- [PASS] **Premium leg PV is positive** — premium_leg_pv=0.068550054470 (should be > 0).
- [PASS] **Protection leg PV is positive** — protection_leg_pv=0.068550054470 (should be > 0).
- [PASS] **NPV is within tolerance (|NPV| ≤ tol)** — |NPV|=2.057e-14, tol=1.0e-06.
- [PASS] **NPV equals premium - protection (definition check)** — (premium-protection) - npv = 0.000e+00 (should be ~0).
