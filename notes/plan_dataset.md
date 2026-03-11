# Data Pipeline Plan

## Goal
Build `utils/dataset.py` and wire it into `train.py`.

## Graph Design (confirmed)
- **One graph per (subclass, section)** — each CSV file becomes one graph
- **Nodes**: cells, features = 140 gene expression values (drop `Blank-*`)
- **Edges**: k-NN on (x, y) spatial coordinates, default k=10
- **Label**: `BRAAK_score` integer (0–6), graph-level
- **Split**: by donor — all sections of a donor go to the same split

---

## 1. `utils/dataset.py`

### Class: `MERFISHDataset(torch.utils.data.Dataset)`
- Takes a pre-built `file_list` (list of CSV paths) and `k` (knn neighbours)
- `__len__`: returns number of files
- `__getitem__`: reads CSV, drops `Blank-*` and metadata cols, builds knn_graph, returns `torch_geometric.data.Data(x, edge_index, y)`

### Static: `MERFISHDataset.collect_files(data_root, subclasses=None)`
- Scans `data_root/{subclass}/*.csv` for all (or given) subclasses
- Returns list of absolute CSV paths

### Static: `MERFISHDataset.donor_split(file_list, val_frac=0.15, test_frac=0.15, seed=42)`
- Peeks at Donor_ID from each file (first row only)
- Groups files by donor
- Randomly assigns donors to train/val/test
- Returns `(train_files, val_files, test_files)`

---

## 2. `utils/__init__.py`
- Add `from .dataset import MERFISHDataset`

---

## 3. `train.py`
- Replace `TUDataset` / `MUTAG` block with `MERFISHDataset`
- Add data_root and k to config (or hardcode for now)

---

## 4. `configs/config.yaml`
- Update `model.kwargs.num_features: 140`
- Update `model.kwargs.num_classes: 7`
- Add `data.root` and `data.k` fields

---

## Known columns
- Total: 186
- Meta (drop from features): Subclass, Section, Donor_ID, BRAAK_score, x, y
- Blank-* (40 cols): drop
- Gene features: 140