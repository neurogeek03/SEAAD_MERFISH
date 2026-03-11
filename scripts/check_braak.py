"""
check_braak.py

For each donor: BRAAK score and total number of sections (after depth filter).
Saves to data/donor_braak_sections.csv.

Usage (from scripts/ or repo root):
    uv run python scripts/check_braak.py
"""

import pandas as pd
from pathlib import Path

project_path = Path.cwd().parents[0]
OBS_CSV      = Path("/scratch/mfafouti/SEA-AD/data/SEAAD_obs.csv")
OUT_PATH     = project_path / "data" / "donor_braak_sections.csv"

DONOR_COL   = "Donor ID"
BRAAK_COL   = "Braak"
SECTION_COL = "Section"
DEPTH_COL   = "Depth from pia"

BRAAK_MAP = {
    "Braak 0":   0,
    "Braak I":   1,
    "Braak II":  2,
    "Braak III": 3,
    "Braak IV":  4,
    "Braak V":   5,
    "Braak VI":  6,
}

obs = pd.read_csv(OBS_CSV, usecols=[DONOR_COL, BRAAK_COL, SECTION_COL, DEPTH_COL])
obs = obs[obs[DEPTH_COL].notna()]
obs[BRAAK_COL] = obs[BRAAK_COL].map(BRAAK_MAP).astype(int)

summary = (
    obs.drop_duplicates([DONOR_COL, SECTION_COL])
    .groupby(DONOR_COL)
    .agg(
        braak_score=(BRAAK_COL, "first"),
        n_sections=(SECTION_COL, "nunique"),
    )
    .reset_index()
    .rename(columns={DONOR_COL: "donor_id"})
    .sort_values(["braak_score", "donor_id"])
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT_PATH, index=False)

print(f"Saved {len(summary)} donors to {OUT_PATH}")

grouped = summary.groupby('braak_score').agg(n_donors=('donor_id','count'), total_sections=('n_sections','sum'))
totals  = grouped.sum().rename('total')
print(f"\n{grouped.to_string()}")
print(f"\n{'total':>11}  {totals['n_donors']:>8}  {totals['total_sections']:>14}")