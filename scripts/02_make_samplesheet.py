"""
02_make_samplesheet.py

Modes:
  --compare          Save distribution CSVs for both schemes (B and C) to notes/
  --scheme B|C       Write samplesheet + save distribution CSV for that scheme

Filters (matching 01_preprocessing.py):
  - Only rows with a non-null "Depth from pia" value
  - BRAAK 0 donors excluded

Split cycle per BRAAK group: [V,E,E,T,T,T,T,T,T,T]
  Val/test filled first → all groups represented in every split even if small.
  ~70% train / 10% val / 20% test overall.

Usage (from repo root):
    uv run python scripts/02_make_samplesheet.py --compare
    uv run python scripts/02_make_samplesheet.py --scheme B
    uv run python scripts/02_make_samplesheet.py --scheme C
"""

import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict

# --- Paths ---
project_path = Path.cwd()
OBS_CSV      = Path("/scratch/mfafouti/SEA-AD/data/SEAAD_obs.csv")
NOTES_DIR    = project_path / "notes"
OUT_PATH     = project_path / "data" / "samplesheet.csv"

# --- Column names ---
DONOR_COL   = "Donor ID"
BRAAK_COL   = "Braak"
SECTION_COL = "Section"
DEPTH_COL   = "Depth from pia"

BRAAK_MAP = {
    "Braak 0": 0, "Braak I": 1, "Braak II": 2, "Braak III": 3,
    "Braak IV": 4, "Braak V": 5, "Braak VI": 6,
}

GROUPING_SCHEMES = {
    "B": {2: "2-3", 3: "2-3", 4: "4",   5: "5-6", 6: "5-6"},
    "C": {2: "2-4", 3: "2-4", 4: "2-4", 5: "5-6", 6: "5-6"},
}

# Val/test first → guarantees representation even for small groups
CYCLE = ["val", "test", "test",
         "training", "training", "training", "training",
         "training", "training", "training"]

SPLITS = ["training", "val", "test"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_obs(obs_csv: Path) -> pd.DataFrame:
    """Load, filter (depth + BRAAK 0), deduplicate to one row per (donor, section)."""
    obs = pd.read_csv(obs_csv, usecols=[DONOR_COL, BRAAK_COL, SECTION_COL, DEPTH_COL])

    braak1_before = obs[obs[BRAAK_COL] == "Braak I"][DONOR_COL].nunique()
    obs = obs[obs[DEPTH_COL].notna()]
    braak1_after  = obs[obs[BRAAK_COL] == "Braak I"][DONOR_COL].nunique()
    print(f"BRAAK 1 donors — before depth filter: {braak1_before}, after: {braak1_after}")

    obs[BRAAK_COL] = obs[BRAAK_COL].map(BRAAK_MAP)
    if obs[BRAAK_COL].isna().any():
        raise ValueError("Unrecognised Braak values found — update BRAAK_MAP")
    obs[BRAAK_COL] = obs[BRAAK_COL].astype(int)
    obs = obs[obs[BRAAK_COL] != 0]
    obs = obs.drop_duplicates([DONOR_COL, SECTION_COL])

    print(f"{obs[DONOR_COL].nunique()} donors | {obs[SECTION_COL].nunique()} sections after filtering")
    return obs


def build_donor_map(obs: pd.DataFrame) -> dict[str, dict]:
    donor_braak    = obs.drop_duplicates(DONOR_COL).set_index(DONOR_COL)[BRAAK_COL].to_dict()
    donor_sections = obs.groupby(DONOR_COL)[SECTION_COL].apply(sorted).to_dict()
    return {
        d: {"braak": donor_braak[d], "sections": list(donor_sections[d])}
        for d in donor_braak
    }


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def round_robin_split(donor_map: dict[str, dict],
                      strat_group: dict[int, str]) -> dict[str, str]:
    """Round-robin within each BRAAK group. Returns {donor_id: split}."""
    by_group: dict[str, list[str]] = defaultdict(list)
    for donor, info in donor_map.items():
        if info["braak"] in strat_group:
            by_group[strat_group[info["braak"]]].append(donor)
    for g in by_group:
        by_group[g].sort()

    donor_to_split: dict[str, str] = {}
    for group in sorted(by_group):
        donors = by_group[group]
        if len(donors) < 3:
            print(f"Warning: group '{group}' has {len(donors)} donor(s) — not all splits guaranteed")
        for i, donor in enumerate(donors):
            donor_to_split[donor] = CYCLE[i % len(CYCLE)]
    return donor_to_split


def build_distribution_df(donor_map: dict[str, dict],
                          donor_to_split: dict[str, str],
                          strat_group: dict[int, str]) -> pd.DataFrame:
    """One row per BRAAK score: group label, donor counts and section counts per split."""
    rows = []
    for braak in sorted(set(info["braak"] for info in donor_map.values())):
        group = strat_group.get(braak, str(braak))
        row = {"braak_score": braak, "strat_group": group}
        for s in SPLITS:
            donors_in = [d for d, info in donor_map.items()
                         if info["braak"] == braak and donor_to_split.get(d) == s]
            row[f"{s}_donors"]   = len(donors_in)
            row[f"{s}_sections"] = sum(len(donor_map[d]["sections"]) for d in donors_in)
        total_d = sum(row[f"{s}_donors"]   for s in SPLITS)
        total_s = sum(row[f"{s}_sections"] for s in SPLITS)
        row["total_donors"]   = total_d
        row["total_sections"] = total_s
        for s in SPLITS:
            row[f"pct_{s}_donors"]   = round(100 * row[f"{s}_donors"]   / total_d, 1) if total_d else 0
            row[f"pct_{s}_sections"] = round(100 * row[f"{s}_sections"] / total_s, 1) if total_s else 0
        rows.append(row)

    # Totals row
    df = pd.DataFrame(rows)
    total_row = {"braak_score": "total", "strat_group": ""}
    for s in SPLITS:
        total_row[f"{s}_donors"]   = df[f"{s}_donors"].sum()
        total_row[f"{s}_sections"] = df[f"{s}_sections"].sum()
    total_row["total_donors"]   = df["total_donors"].sum()
    total_row["total_sections"] = df["total_sections"].sum()
    grand_d = total_row["total_donors"]
    grand_s = total_row["total_sections"]
    for s in SPLITS:
        total_row[f"pct_{s}_donors"]   = round(100 * total_row[f"{s}_donors"]   / grand_d, 1) if grand_d else 0
        total_row[f"pct_{s}_sections"] = round(100 * total_row[f"{s}_sections"] / grand_s, 1) if grand_s else 0

    return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)


def main():
    obs       = load_obs(OBS_CSV)
    donor_map = build_donor_map(obs)

    if "--compare" in sys.argv:
        for name, scheme in GROUPING_SCHEMES.items():
            dm             = {d: info for d, info in donor_map.items() if info["braak"] in scheme}
            donor_to_split = round_robin_split(dm, scheme)
            df             = build_distribution_df(dm, donor_to_split, scheme)
            out            = project_path / 'notes' / f"compare_scheme_{name}.csv"
            df.to_csv(out, index=False)
            print(f"Saved scheme {name} → {out}")
        return

    scheme_arg = next((a for a in sys.argv if a in GROUPING_SCHEMES), None)
    if scheme_arg is None:
        print(f"Usage: python 02_make_samplesheet.py --compare | --scheme B | --scheme C")
        sys.exit(1)

    strat_group    = GROUPING_SCHEMES[scheme_arg]
    dm             = {d: info for d, info in donor_map.items() if info["braak"] in strat_group}
    donor_to_split = round_robin_split(dm, strat_group)

    # Write samplesheet
    rows = [
        {"donor_id": donor, "section_id": section, "split": donor_to_split[donor]}
        for donor, info in sorted(dm.items())
        for section in sorted(info["sections"])
    ]
    samplesheet = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    samplesheet.to_csv(OUT_PATH, index=False)
    print(f"Samplesheet written → {OUT_PATH}  ({len(samplesheet)} rows)")

    # Save distribution report
    df  = build_distribution_df(dm, donor_to_split, strat_group)
    out = NOTES_DIR / f"distribution_scheme_{scheme_arg}.csv"
    df.to_csv(out, index=False)
    print(f"Distribution saved  → {out}")


if __name__ == "__main__":
    main()