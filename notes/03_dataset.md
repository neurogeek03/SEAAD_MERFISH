# Dataset — Graph Construction (`utils/dataset.py`)

## Overview

Each graph represents one tissue section for one cell type subclass. The dataset
converts the preprocessed CSVs into PyTorch Geometric `Data` objects ready for
GCN training.

---

## Graph design

| Element | Definition |
|---|---|
| One graph | One `{Subclass}/{Section}.csv` file |
| Nodes | Individual cells of that subclass in that section |
| Node features (`x`) | 140 z-scored gene expression values |
| Edges (`edge_index`) | Spatial k-NN graph built from (x, y) coordinates |
| Graph label (`y`) | BRAAK score integer (0–6) |

---

## Nodes

Each **node = one cell** from the tissue section CSV. Here is how nodes are constructed inside `process()` (`utils/dataset.py:43`):

### Column filtering

```python
blank_cols = [c for c in df.columns if c.startswith('Blank-')]
drop_cols  = set(blank_cols) | META_COLS
gene_cols  = [c for c in df.columns if c not in drop_cols]
```

Three column groups are separated:

| Group | Action | Why |
|---|---|---|
| `Blank-*` probes | Dropped | Negative-control probes — no biological target; would add noise |
| `META_COLS` (`Subclass`, `Section`, `Donor_ID`, `BRAAK_score`, `x`, `y`) | Dropped from features | Not gene expression; some (x, y) are used separately for edges |
| Everything else | Kept as `gene_cols` | Real targeted genes — the actual node features |

### Node feature matrix `x`

```python
x = torch.tensor(df[gene_cols].values, dtype=torch.float)  # shape: [N_cells, N_genes]
```

- Rows = cells (nodes), columns = gene expression values
- Values are already **log1p-normalised** and **per-subclass z-scored** from `scripts/01_preprocessing.py`
- `dtype=torch.float` → 32-bit floats, as expected by PyG and most GCN layers

### Spatial coordinates `pos`

```python
pos = torch.tensor(df[['x', 'y']].values, dtype=torch.float)  # shape: [N_cells, 2]
```

- Stored on the `Data` object as `pos` (not as node features in `x`)
- Used exclusively to build `edge_index` via `knn_graph`; the GCN itself never sees `pos`

### Graph label `y`

```python
y = torch.tensor([int(df['BRAAK_score'].iloc[0])], dtype=torch.long)  # shape: [1]
```

- All cells in one CSV share the same BRAAK score (it is a donor-level label, constant within a section)
- Reading only the first row (`.iloc[0]`) is sufficient
- `dtype=torch.long` is required by `CrossEntropyLoss`

---

## Step 1 — Collect file paths (`collect_files`)

```python
MERFISHDataset.collect_files(data_root, subclasses=None)
```

- Scans `data_root/{subclass}/*.csv` for all (or specified) subclasses
- Returns a sorted flat list of absolute CSV paths
- Each path = one future graph

---

## Step 2 — Donor-level split (`donor_split`)

```python
MERFISHDataset.donor_split(file_list, val_frac=0.15, test_frac=0.15, seed=42)
```

- Reads the first row of each CSV to get `Donor_ID`
- Groups files by donor
- Randomly assigns **whole donors** to train / val / test
- A donor never spans two splits — prevents data leakage
- Returns `(train_files, val_files, test_files)`

---

## Step 3 — Build one graph (`__getitem__`)

Called automatically by the DataLoader for each sample. For a given CSV:

### 3a — Load and filter columns

```python
df = pd.read_csv(path)
```

Drop:
- `Blank-*` columns (40 probes with no biological target)
- Metadata: `Subclass`, `Section`, `Donor_ID`, `BRAAK_score`, `x`, `y`

Remaining columns = 140 gene features.

### 3b — Build node feature matrix

```python
x = torch.tensor(df[gene_cols].values, dtype=torch.float)  # shape: [N_cells, 140]
```

### 3c — Build spatial edge index

```python
pos = torch.tensor(df[['x', 'y']].values, dtype=torch.float)
edge_index = knn_graph(pos, k=10, loop=False)  # shape: [2, N_cells * k]
```

Each cell is connected to its `k` nearest spatial neighbours. `loop=False`
excludes self-loops. `k` is configurable (`cfg.data.k`).

### 3d — Extract graph label

```python
y = torch.tensor([int(df['BRAAK_score'].iloc[0])], dtype=torch.long)  # shape: [1]
```

All cells in a section share the same BRAAK score (it is a donor-level label).

### 3e — Return

```python
Data(x=x, edge_index=edge_index, y=y, pos=pos)
```

---

## Usage

```python
from utils.dataset import MERFISHDataset
from torch_geometric.loader import DataLoader

all_files = MERFISHDataset.collect_files(cfg.data.root)
train_files, val_files, test_files = MERFISHDataset.donor_split(all_files)

train_loader = DataLoader(MERFISHDataset(train_files, k=cfg.data.k),
                          batch_size=cfg.batch_size, shuffle=True)
val_loader   = DataLoader(MERFISHDataset(val_files,   k=cfg.data.k),
                          batch_size=cfg.batch_size, shuffle=False)
```

---

## Column count reference

| Group | Count |
|---|---|
| Metadata | 4 (`Subclass`, `Section`, `Donor_ID`, `BRAAK_score`) |
| Spatial | 2 (`x`, `y`) |
| Blank probes | 40 (`Blank-*`) |
| **Gene features** | **140** |
| Total | 186 |