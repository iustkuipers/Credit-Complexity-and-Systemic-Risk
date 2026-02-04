# Q3 Audit: 7Y CDS Leg PV Check

This audit validates the Q2 stripped hazard curve by explicitly re-pricing the 7Y CDS.

## Inputs

- Maturity: **7Y**
- Spread: **120 bps** (decimal: 0.012000)
- Tolerance: **1.0e-06**

## Present Values

- PV Premium Leg: **0.068550054470**
- PV Protection Leg: **0.068550054470**
- NPV (Premium − Protection): **-2.056688153118e-14**
- |NPV|: **2.056688153118e-14**

## Conclusion

- Passed: **True** (criterion: |NPV| ≤ 1.0e-06)
