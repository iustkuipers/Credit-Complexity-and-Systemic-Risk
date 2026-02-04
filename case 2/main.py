# main.py

"""
Case 2 (2026): CDS Stripping
Main orchestration file.

This file intentionally contains:
- no pricing logic
- no numerical methods
- no print or logging statements

It only wires data to services.
"""

from data.data import (
    CASE_NAME,
    COUNTERPARTY,
    cds_quotes,
    RISK_FREE_RATE,
    LGD,
    PREMIUM_FREQUENCY,
)
#q1
from services.q1_simple.orchestrator import run_q1
from validation.q1_validation import run_q1_validation
#q2
from services.q2_exact.orchestrator import run_q2
from validation.q2_validation import run_q2_validation
#q3
from services.q3_audit.orchestrator import run_q3
from validation.q3_validation import run_q3_validation
#q4
from services.q4_stress.orchestrator import run_q4
from validation.q4_validation import run_q4_validation


def main():
    """
    Entry point for the Case 2 CDS stripping workflow.
    """

    # Package inputs in a single dict if desired
    inputs = {
        "case_name": CASE_NAME,
        "counterparty": COUNTERPARTY,
        "cds_quotes": cds_quotes,
        "risk_free_rate": RISK_FREE_RATE,
        "lgd": LGD,
        "premium_frequency": PREMIUM_FREQUENCY,
    }

    # Q1: Simple model
    run_q1(
        cds_quotes=inputs["cds_quotes"],
        lgd=inputs["lgd"],
        output_root="output",
        save=True,
        verbose=True,  # prints progress inside the Q1 orchestrator
    )

    # Q1: Validation
    run_q1_validation(
        lgd=inputs["lgd"],
        q1_output_dir="output/q1",
        save=True,
        verbose=True,
    )

    # Q2: Exact CDS stripping
    q2_outputs = run_q2(
        cds_quotes=inputs["cds_quotes"],
        risk_free_rate=inputs["risk_free_rate"],
        lgd=inputs["lgd"],
        premium_frequency=inputs["premium_frequency"],  # quarterly = 4
        output_root="output",
        save=True,
        verbose=True,
    )

    # Q2: Validation
    run_q2_validation(
        risk_free_rate=inputs["risk_free_rate"],
        lgd=inputs["lgd"],
        payments_per_year=inputs["premium_frequency"],
        output_root="output",
        save=True,
        verbose=True,
    )

    # Q3: Audit / validation
    run_q3(
        cds_quotes=inputs["cds_quotes"],
        risk_free_rate=inputs["risk_free_rate"],
        lgd=inputs["lgd"],
        premium_frequency=inputs["premium_frequency"],
        tolerance=1e-6,
        strip_result=q2_outputs.strip_result,
        output_root="output",
        save=True,
        verbose=True,
    )

    # Q3: Validation
    run_q3_validation(
        cds_quotes=inputs["cds_quotes"],
        risk_free_rate=inputs["risk_free_rate"],
        lgd=inputs["lgd"],
        premium_frequency=inputs["premium_frequency"],
        tolerance=1e-6,
        strip_result=q2_outputs.strip_result,
        output_root="output",
        save=True,
        verbose=True,
    )

    # Q4: Stress testing
    run_q4(
        cds_quotes=inputs["cds_quotes"],
        lgd=inputs["lgd"],
        premium_frequency=inputs["premium_frequency"],
        baseline_r=inputs["risk_free_rate"],
        stressed_r=inputs["risk_free_rate"] * 3.33,  # 3x multiplier (0.03 -> 0.10)
        output_root="output",
        save=True,
        verbose=True,
    )

    # Q4: Validation
    run_q4_validation(
        output_root="output",
        save=True,
        verbose=True,
    )

    # =========================
    # Workflow complete
    # =========================

    return


if __name__ == "__main__":
    main()
