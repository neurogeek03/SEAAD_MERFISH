import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse
from pathlib import Path

PATH = "/scratch/mfafouti/SEA-AD/SEAAD_MTG_MERFISH.2024-12-11.h5ad"
OUT_DIR = Path("expression_by_celltype_donor")
OUT_DIR_NORM = Path("expression_by_celltype_donor_lognorm_zscore")

DONOR_COL = "Donor ID"
SUBCLASS_COL = "Subclass"
braak_col = "Braak"

print(f"Loading {PATH} ...")
adata = ad.read_h5ad(PATH)
print(f"AnnData shape: {adata.shape}  ({adata.n_obs} cells, {adata.n_vars} genes)\n")

# --- Inspect var (gene metadata) ---
print("--- adata.var columns ---")
print(list(adata.var.columns))
print("\n--- First 5 var rows ---")
print(adata.var.head())
print("\n--- Blank entries in var_names ---")
blank_vars = [v for v in adata.var_names if "Blank" in v or "blank" in v]
print(f"  {len(blank_vars)} blank entries: {blank_vars[:5]} ...")
print()

# --- Check all subclasses present in all donors ---
obs = adata.obs[[DONOR_COL, SUBCLASS_COL, braak_col]].copy()

all_subclasses = sorted(obs[SUBCLASS_COL].unique())
all_donors = sorted(obs[DONOR_COL].unique())

print(f"Subclasses ({len(all_subclasses)}): {all_subclasses}")
print(f"Donors ({len(all_donors)}): {all_donors}\n")

presence = obs.groupby(DONOR_COL)[SUBCLASS_COL].apply(set)

missing = {}
for donor in all_donors:
    absent = set(all_subclasses) - presence[donor]
    if absent:
        missing[donor] = sorted(absent)

if not missing:
    print("All subclasses are present in every donor.\n")
else:
    print("Donors missing one or more subclasses:")
    for donor, absent in missing.items():
        print(f"  {donor}: missing {absent}")
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
    X_sub = X[mask]                          # (n_cells_in_subclass, n_genes)
    mean = X_sub.mean(axis=0)                # (n_genes,)
    std  = X_sub.std(axis=0)
    std[std == 0] = 1.0                      # avoid divide-by-zero for const genes
    X_zscore[mask] = (X_sub - mean) / std
    print(f"  {subclass:<35} n={mask.sum():>7}  gene-std range: [{std.min():.3f}, {std.max():.3f}]")

adata.layers["lognorm"] = X
adata.layers["lognorm_zscore"] = X_zscore
print("\nStored in adata.layers['lognorm'] and adata.layers['lognorm_zscore']")
print()

# --- Save normalized data: expression_by_celltype_donor_lognorm_zscore/{Subclass}/{Donor_ID}.csv ---
print(f"Saving normalized CSVs to {OUT_DIR_NORM} ...")
gene_cols = list(adata.var_names)

for subclass in all_subclasses:
    subclass_mask = (adata.obs[SUBCLASS_COL] == subclass).values
    subclass_dir = OUT_DIR_NORM / subclass
    subclass_dir.mkdir(parents=True, exist_ok=True)

    donors_in_subclass = sorted(adata.obs.loc[subclass_mask, DONOR_COL].unique())
    for donor in donors_in_subclass:
        donor_mask = subclass_mask & (adata.obs[DONOR_COL] == donor).values
        braak_val = adata.obs.loc[donor_mask, braak_col].iloc[0]

        df = pd.DataFrame(X_zscore[donor_mask], columns=gene_cols)
        df.insert(0, "BRAAK_score", braak_val)
        df.insert(0, "Donor_ID", donor)
        df.insert(0, "Subclass", subclass)

        out_path = subclass_dir / f"{donor}.csv"
        df.to_csv(out_path, index=False)

    print(f"  {subclass:<35} {len(donors_in_subclass)} donors")

print("\nDone.")