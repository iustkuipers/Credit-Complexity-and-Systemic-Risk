# Q1 Validation Report

- Total checks: **11**
- Passed: **11**
- Failed: **0**

All checks passed.

## Check details

- [PASS] **Bounds [0,1] for survival_T** — All values of survival_T should lie in [0,1] (within tol=1e-10).
- [PASS] **Bounds [0,1] for cum_default_prob** — All values of cum_default_prob should lie in [0,1] (within tol=1e-10).
- [PASS] **Bounds [0,1] for fwd_default_prob** — All values of fwd_default_prob should lie in [0,1] (within tol=1e-10).
- [PASS] **Survival monotone decreasing** — Survival probabilities should be non-increasing with maturity.
- [PASS] **Cumulative default monotone increasing** — Cumulative default probabilities should be non-decreasing with maturity.
- [PASS] **Non-negativity for avg_hazard** — avg_hazard should be non-negative (within tol=1e-10).
- [PASS] **Non-negativity for fwd_hazard** — fwd_hazard should be non-negative (within tol=1e-10).
- [PASS] **Non-negativity for fwd_default_prob** — fwd_default_prob should be non-negative (within tol=1e-10).
- [PASS] **Forward PDs sum to final cumulative PD** — sum(fwd_default_prob)=0.268384371053 vs final cum_default_prob=0.268384371053 (should match within tolerance ~1e-8).
- [PASS] **Average hazard matches R/LGD** — Max abs error between avg_hazard and cds_spread/LGD is 9.714e-17.
- [PASS] **Survival matches exp(-avg_hazard*T)** — Max abs error between survival_T and exp(-avg_hazard*T) is 4.441e-16.
