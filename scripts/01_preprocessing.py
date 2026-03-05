import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse
from pathlib import Path

PATH = "/scratch/mfafouti/SEA-AD/SEAAD_MTG_MERFISH.2024-12-11.h5ad"
OUT_DIR = Path("expression_by_celltype_donor")
OUT_DIR_NORM = Path("xy_expression_by_celltype_donor_lognorm_zscore")

DONOR_COL = "Donor ID"
SUBCLASS_COL = "Subclass"
SECTION_COL = "Section"
DEPTH_COL = "Depth from pia"
braak_col = "Braak"

print(f"Loading {PATH} ...")
adata = ad.read_h5ad(PATH)
print(f"AnnData shape: {adata.shape}  ({adata.n_obs} cells, {adata.n_vars} genes)\n")

# --- Extract spatial coordinates ---
print("--- adata.obsm keys ---")
print(list(adata.obsm.keys()))
SPATIAL_KEY = "spatial"  # update if obsm key differs
spatial_coords = adata.obsm[SPATIAL_KEY]  # shape: (n_cells, 2) — columns: [x, y]
print(f"Spatial coords shape: {spatial_coords.shape}\n")

# --- Filter: keep only cells with a value for "Depth from pia" ---
depth_mask = adata.obs[DEPTH_COL].notna().values
n_before = adata.n_obs
adata = adata[depth_mask].copy()
spatial_coords = spatial_coords[depth_mask]
print(f"Filtered by '{DEPTH_COL}': {n_before} → {adata.n_obs} cells ({n_before - adata.n_obs} removed)\n")

# --- Check all subclasses present in all sections ---
obs = adata.obs[[DONOR_COL, SECTION_COL, SUBCLASS_COL, braak_col]].copy()

all_subclasses = sorted(obs[SUBCLASS_COL].unique())
all_sections = sorted(obs[SECTION_COL].unique())

print(f"Subclasses ({len(all_subclasses)}): {all_subclasses}")
print(f"Sections ({len(all_sections)}): {all_sections}\n")

presence = obs.groupby(SECTION_COL)[SUBCLASS_COL].apply(set)

missing = {}
for section in all_sections:
    absent = set(all_subclasses) - presence[section]
    if absent:
        missing[section] = sorted(absent)

if not missing:
    print("All subclasses are present in every section.\n")
else:
    print("Sections missing one or more subclasses:")
    for section, absent in missing.items():
        print(f"  {section}: missing {absent}")
    print()

# --- Normalization: global log1p → per-cell-type z-score per gene ---
print("Applying global log1p ...")
X = adata.X
if scipy.sparse.issparse(X):
    X = X.toarray()
X = np.log1p(X.astype(np.float32))
print(f"  log1p done. min={X.min():.3f}, max={X.max():.3f}")

print("\nApplying per-subclass z-score per gene ...")
X_zscore = np.zeros_like(X)
for subclass in all_subclasses:
    mask = (adata.obs[SUBCLASS_COL] == subclass).values
    X_sub = X[mask]                         
    mean = X_sub.mean(axis=0)                
    std  = X_sub.std(axis=0)
    std[std == 0] = 1.0                     
    X_zscore[mask] = (X_sub - mean) / std
    print(f"  {subclass:<35} n={mask.sum():>7}  gene-std range: [{std.min():.3f}, {std.max():.3f}]")

adata.layers["lognorm"] = X
adata.layers["lognorm_zscore"] = X_zscore
print()

# --- Save normalized data: {Subclass}/{Section}.csv ---
print(f"Saving normalized CSVs to {OUT_DIR_NORM} ...")
gene_cols = list(adata.var_names)

section_counts = []
for subclass in all_subclasses:
    subclass_mask = (adata.obs[SUBCLASS_COL] == subclass).values
    subclass_dir = OUT_DIR_NORM / subclass
    subclass_dir.mkdir(parents=True, exist_ok=True)

    sections_in_subclass = sorted(adata.obs.loc[subclass_mask, SECTION_COL].unique())
    for section in sections_in_subclass:
        section_mask = subclass_mask & (adata.obs[SECTION_COL] == section).values
        braak_val = adata.obs.loc[section_mask, braak_col].iloc[0]
        donor_val = adata.obs.loc[section_mask, DONOR_COL].iloc[0]

        df = pd.DataFrame(X_zscore[section_mask], columns=gene_cols)
        df.insert(0, "BRAAK_score", braak_val)
        df.insert(0, "Donor_ID", donor_val)
        df.insert(0, "Section", section)
        df.insert(0, "Subclass", subclass)
        df["x"] = spatial_coords[section_mask, 0]
        df["y"] = spatial_coords[section_mask, 1]

        out_path = subclass_dir / f"{section}.csv"
        df.to_csv(out_path, index=False)

    section_counts.append({"Subclass": subclass, "n_sections": len(sections_in_subclass)})
    print(f"  {subclass:<35} {len(sections_in_subclass)} sections")

counts_path = Path("data/section_counts.csv")
counts_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(section_counts).to_csv(counts_path, index=False)
print(f"\nSaved section counts to {counts_path}")