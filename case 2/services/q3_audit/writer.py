# services/q3_audit/writer.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from services.q3_audit.audit_7y import Audit7YResult


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def audit7y_to_dataframe(result: Audit7YResult) -> pd.DataFrame:
    """
    Convert Audit7YResult into a 1-row DataFrame suitable for CSV/XLSX export.
    """
    return pd.DataFrame(
        [
            {
                "maturity_years": result.maturity_years,
                "cds_spread_bps": result.cds_spread_bps,
                "cds_spread": result.cds_spread,
                "premium_leg_pv": result.premium_leg_pv,
                "protection_leg_pv": result.protection_leg_pv,
                "npv": result.npv,
                "abs_npv": result.abs_npv,
                "tolerance": result.tolerance,
                "passed": result.passed,
            }
        ]
    )


def build_audit7y_markdown(result: Audit7YResult) -> str:
    """
    Build a short Markdown report for the Q3 audit.
    """
    lines = []
    lines.append("# Q3 Audit: 7Y CDS Leg PV Check\n")
    lines.append("This audit validates the Q2 stripped hazard curve by explicitly re-pricing the 7Y CDS.\n")
    lines.append("## Inputs\n")
    lines.append(f"- Maturity: **{result.maturity_years:.0f}Y**")
    lines.append(f"- Spread: **{result.cds_spread_bps:.0f} bps** (decimal: {result.cds_spread:.6f})")
    lines.append(f"- Tolerance: **{result.tolerance:.1e}**\n")
    lines.append("## Present Values\n")
    lines.append(f"- PV Premium Leg: **{result.premium_leg_pv:.12f}**")
    lines.append(f"- PV Protection Leg: **{result.protection_leg_pv:.12f}**")
    lines.append(f"- NPV (Premium − Protection): **{result.npv:.12e}**")
    lines.append(f"- |NPV|: **{result.abs_npv:.12e}**\n")
    lines.append("## Conclusion\n")
    lines.append(
        f"- Passed: **{result.passed}** "
        f"(criterion: |NPV| ≤ {result.tolerance:.1e})"
    )
    lines.append("")
    return "\n".join(lines)


def save_q3_audit_7y(
    *,
    result: Audit7YResult,
    output_root: str = "output",
    verbose: bool = True,
) -> Path:
    """
    Save Q3 audit artifacts to output/q3:
      - q3_audit_7y.csv
      - q3_audit_7y.xlsx
      - q3_audit_7y.md

    Returns
    -------
    Path
        Output directory used (output/q3)
    """
    out_dir = Path(output_root) / "q3"
    _ensure_dir(out_dir)

    df = audit7y_to_dataframe(result)

    csv_path = out_dir / "q3_audit_7y.csv"
    xlsx_path = out_dir / "q3_audit_7y.xlsx"
    md_path = out_dir / "q3_audit_7y.md"

    if verbose:
        print(f"[Q3][writer] Writing audit CSV  -> {csv_path}")
    df.to_csv(csv_path, index=False)

    if verbose:
        print(f"[Q3][writer] Writing audit XLSX -> {xlsx_path}")
    df.to_excel(xlsx_path, index=False)

    md_text = build_audit7y_markdown(result)
    if verbose:
        print(f"[Q3][writer] Writing audit MD   -> {md_path}")
    md_path.write_text(md_text, encoding="utf-8")

    return out_dir
