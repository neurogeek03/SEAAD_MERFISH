# SEAAD_MERFISH

Predict **BRAAK score** (Alzheimer's staging, 0–6) from spatial gene expression patterns of individual cell types using graph-based methods applied to MERFISH data from the middle temporal gyrus (MTG).

Raw data (`h5ad`) available at: https://sea-ad-spatial-transcriptomics.s3.amazonaws.com/index.html#middle-temporal-gyrus/all_donors-h5ad/

---

## Data

- **Source**: `SEAAD_MTG_MERFISH.2024-12-11.h5ad` — ~1.88M cells, 140 targeted genes, MTG region
- **Processed data**: `data/braak_xy_expression_by_celltype_donor_lognorm_zscore/` (versioned with DVC)
  - Structure: `{Subclass}/{Section}.csv` — one file per (cell type × tissue section)
  - Columns: `Subclass`, `Section`, `Donor_ID`, `BRAAK_score`, `[genes]`, `x`, `y`
  - Normalization: global log1p → per-subclass z-score per gene

### Key `.obs` columns (raw h5ad)

- `Depth from pia`: non-null only for cells in a rectangular cortical strip used in the original paper's downstream analyses
- `X_spatial_raw_0/1`: coordinates for plotting a single section
- `X_spatial_tiled_0/1`: coordinates for plotting all sections together

---

## Method

**Graph Kernel SVM** (`train_kernel.py`)

- One graph per donor: cell-type node labels, spatial k-NN edges
- Weisfeiler-Lehman graph kernels combined via Multiple Kernel Learning (MKL/GRAM)
- BRAAK scores binned into 2 groups: {II–IV} vs {V–VI}
- Donor-level cross-validation using pre-defined folds (`folds.json`)

---

## Code Structure

```
SEAAD_MERFISH/
├── train_kernel.py           # Graph kernel SVM training
├── scripts/
│   ├── 00_export_data.py     # Export from h5ad
│   ├── 00_plot_sections.py   # Visualize tissue sections
│   ├── 01_preprocessing.py   # h5ad → normalized CSVs
│   ├── check_braak.py        # BRAAK label distribution checks
│   ├── check_class_balance.py
│   ├── check_graphs.py       # Graph construction validation
│   ├── evaluate.py           # Model evaluation
│   └── explore_cps_braak.py
├── data/
│   ├── braak_xy_expression_by_celltype_donor_lognorm_zscore/  # processed CSVs (DVC)
│   ├── baseline_info.csv     # demographic data
│   └── section_counts.csv    # sections per subclass
├── folds.json                # donor-level train/test splits
├── figures/                  # output plots
└── pyproject.toml            # uv project (Python 3.13)
```

---

## Setup

```bash
uv sync
```

### Running

```bash
uv run python train_kernel.py
```

---
