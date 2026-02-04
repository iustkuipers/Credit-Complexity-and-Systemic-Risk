# data.py

import pandas as pd

# =========================
# Case metadata
# =========================
CASE_NAME = "Case 2 (2026): CDS Stripping"
COUNTERPARTY = "C"

# =========================
# Market assumptions
# =========================
RISK_FREE_RATE = 0.03        # annually compounded, flat
LGD = 0.40                  # loss given default
PREMIUM_FREQUENCY = 4       # quarterly payments

# =========================
# CDS market data
# =========================
cds_quotes = pd.DataFrame(
    {
        "maturity_years": [1, 3, 5, 7, 10],
        "cds_spread_bps": [100, 110, 120, 120, 125],
        "formula": ["R(1)", "R(3)", "R(5)", "R(7)", "R(10)"],
    }
)

# Optional: also store spreads in decimal form
cds_quotes["cds_spread"] = cds_quotes["cds_spread_bps"] / 10_000
