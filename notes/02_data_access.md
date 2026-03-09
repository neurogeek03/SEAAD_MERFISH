# Data Access via DVC

Processed data is not stored in git. It is tracked with [DVC](https://dvc.org) and stored on a remote. Follow the steps below to obtain it.

## Prerequisites

1. Clone the repo and set up the Python environment:
   ```bash
   git clone git@github.com:neurogeek03/SEAAD_MERFISH.git
   cd SEAAD_MERFISH
   uv sync
   ```

2. Extract data

## Extracting the Data

```bash
tar -xJvf data/braak_xy_expression_by_celltype_donor_lognorm_zscore.tar.xz
```

This will download the processed data into `data/` on your machine, matching the exact version tied to your current git commit.

## Data contents

After pulling, `data/` will contain:

```
data/
├── section_counts.csv                              # number of sections per subclass
└── xy_expression_by_celltype_donor_lognorm_zscore/
    └── {Subclass}/
        └── {Section}.csv                           # one file per tissue section
```

Each `{Section}.csv` has the following columns:

| Column | Description |
|---|---|
| `Subclass` | Cell type subclass |
| `Section` | Tissue section ID |
| `Donor_ID` | Donor the section belongs to |
| `BRAAK_score` | Braak stage for that donor |
| `[genes...]` | Log-normalised and per-subclass z-scored expression |
| `x` | Spatial x coordinate (from `adata.obsm["spatial"]`) |
| `y` | Spatial y coordinate (from `adata.obsm["spatial"]`) |
