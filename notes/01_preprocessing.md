# Preprocessing — SEAAD MERFISH

## Data source

`SEAAD_MTG_MERFISH.2024-12-11.h5ad`

- ~1.88M cells, 180 genes (targeted MERFISH panel)
- `adata.X` contains volume-normalized transcript counts 

## Normalization pipeline

### 1. Global log1p

```
X_lognorm = log1p(adata.X)
```

Applied to the full dataset at once. Log1p compresses the wide dynamic range of
transcript counts (0–885) into a smaller, more symmetric range (~0–7), which
stabilizes variance and improves ML model behaviour.

Log1p is applied globally (not per cell type) because it is a monotonic,
element-wise transform — the result is identical regardless of grouping.

### 2. Per-subclass z-score per gene

```
for each subclass:
    mean_g = mean of each gene across all cells in that subclass
    std_g  = std  of each gene across all cells in that subclass
    X_zscore[subclass cells] = (X_lognorm - mean_g) / std_g
```

Z-scoring is done **within each cell type** rather than globally. Rationale:

- Cell types have vastly different baseline expression profiles. 
- Per-subclass z-scoring removes the cell-type-level baseline, leaving only
  within-type variation — the signal most relevant to disease
- Genes with zero standard deviation within a subclass (constant expression) are
  assigned std=1 to avoid NaNs; their z-score remains 0.

## Folder structure

Raw (original) per-cell-type per-donor CSVs:

```
expression_by_celltype_donor/
    {Subclass}/
        {Donor_ID}.csv
```

Normalized (lognorm + z-score) per-cell-type per-donor CSVs:

```
expression_by_celltype_donor_lognorm_zscore/
    {Subclass}/
        {Donor_ID}.csv
```

### CSV format

Each file contains all cells from one subclass × donor combination. Columns:

```
Subclass | Donor_ID | BRAAK_score | <gene_1> | <gene_2> | ... | <gene_180>
```

One row = one cell.

Example path: `expression_by_celltype_donor_lognorm_zscore/Sncg/H21.33.040.csv`