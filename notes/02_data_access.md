## Prerequisites

1. Clone the repo and set up the Python environment:
   ```bash
   git clone git@github.com:neurogeek03/SEAAD_MERFISH.git
   cd SEAAD_MERFISH
   uv sync
   ```

2. Extract data

## Generating the Data (from raw h5ad)

Run preprocessing to produce log1p-normalised CSVs (no z-score — that is done per split at load time):

```bash
uv run python scripts/01_preprocessing.py
```

Output: `data/braak_xy_expression_by_celltype_donor_lognorm/`

Then create the tarball for distribution or archiving:

```bash
cd data
tar -cJvf braak_xy_expression_by_celltype_donor_lognorm.tar.xz \
    braak_xy_expression_by_celltype_donor_lognorm/
```

`-c` = create, `-J` = xz compression, `-v` = verbose, `-f` = output file.
xz is slow to compress but produces the smallest archive — use `-z` (gzip) for faster compression if size is not a concern.

## Extracting the Data

```bash
tar -xJvf data/braak_xy_expression_by_celltype_donor_lognorm.tar.xz -C data/
```

## Data contents

After extracting, `data/` will contain:

```
data/
├── section_counts.csv                              # number of sections per subclass
├── samplesheet.csv                                 # donor → section → split mapping
└── braak_xy_expression_by_celltype_donor_lognorm/
    └── {Subclass}/
        └── {Section}.csv                           # one file per tissue section
```

Each `{Section}.csv` has the following columns:

| Column | Description |
|---|---|
| `Subclass` | Cell type subclass |
| `Section` | Tissue section ID |
| `Donor_ID` | Donor the section belongs to |
| `BRAAK_score` | Braak stage for that donor (integer 0–6) |
| `[genes...]` | Log1p-normalised expression (z-score applied per split at load time) |
| `x` | Spatial x coordinate (from `adata.obsm["spatial"]`) |
| `y` | Spatial y coordinate (from `adata.obsm["spatial"]`) |
