# Plan: Stratified Donor Split, Samplesheet, and Per-Split Z-scoring

## Context

- Data: one CSV per (subclass × section), columns: `Subclass, Section, Donor_ID, BRAAK_score, [genes], x, y`
- Each donor has a single BRAAK score (0–6); donors can have multiple sections
- `dataset.py` already supports loading splits from a samplesheet (`donor_split` method)
- Samplesheet format (already expected): `donor_id | section_id | split`
- Current preprocessing z-scores globally (data leakage) — must be redone per split

---

## Task 1 — Stratified Donor Split → `data/samplesheet.csv`

### Script: `scripts/02_make_samplesheet.py`

**Steps:**

1. Pick any one subclass folder (e.g., `Astrocyte`) from the extracted CSVs
2. For each `{Section}.csv`, read first row only → extract `Donor_ID` and `BRAAK_score`
3. Build a `donor → (BRAAK_score, [section_ids])` mapping
4. **Stratified split logic:**
   - Group donors by BRAAK score (7 groups: 0–6)
   - Within each BRAAK group, sort donors by ID (deterministic)
   - Assign donors using a repeating 10-slot cycle: `[T,T,T,T,T,T,T,V,E,E]`
     - T = training (7/10 = 70%), V = val (1/10 = 10%), E = test (2/10 = 20%)
   - For groups with < 10 donors the cycle truncates naturally, e.g.:
     - 3 donors → T, V, E (one each)
     - 5 donors → T, T, T, V, E
   - Log which BRAAK groups had < 3 donors (can't guarantee all splits represented)
5. Flatten: for each donor, emit one row per section
6. Write `data/samplesheet.csv` with columns: `donor_id, section_id, split`
   - `split` values: `training`, `val`, `test`

**Why round-robin within BRAAK groups?**
Avoids randomness — same result every run without a seed, and maximises balance even for small groups.

### Step 5 — BRAAK distribution check (inline, after writing samplesheet)

After writing the samplesheet, print a verification table:

```
BRAAK | n_donors_train | n_donors_val | n_donors_test | %train | %val | %test
  0   |       X        |      X       |       X       |  XX%   |  XX% |  XX%
  1   |       ...
  ...
total |       X        |      X       |       X       |  XX%   |  XX% |  XX%
```

Also print a section-level count table (since donors vary in number of sections).
Goal: confirm val and test BRAAK distributions are proportionally similar to train.

---

## Task 2 — Per-Split Z-scoring → Updated Preprocessing

### Problem with current `01_preprocessing.py`

- Z-score mean/std computed across ALL donors (all splits mixed)
- Val/test donor stats influence normalization → data leakage

### New approach: `scripts/03_normalize_by_split.py`

**Inputs:**
- Existing per-section CSVs (log1p already applied in `01_preprocessing.py`)
- `data/samplesheet.csv`

**Steps:**

1. Load samplesheet → build `section → split` mapping
2. For each subclass:
   a. Separate sections into train / val / test using samplesheet
   b. For each split independently: stack all sections in that split → compute per-gene mean and std → z-score
      - train: `z = (x - mean_train) / std_train`
      - val:   `z = (x - mean_val)   / std_val`
      - test:  `z = (x - mean_test)  / std_test`
   c. Write z-scored CSVs to output dir (same structure)
3. Save per-subclass per-split stats to `data/zscore_stats/{subclass}_{split}_stats.csv`
   (columns: `gene, mean, std`) — needed to z-score new data consistently in future

**Output directory:** `data/xy_expression_by_celltype_donor_lognorm_zscore_splitwise/`
(keep old dir intact until validated)

---

## Task 3 — Wire Samplesheet into Training Pipeline

- `dataset.py` already has `donor_split(file_list, samplesheet)` — no changes needed
- Update `configs/config.yaml`: add `data.samplesheet` and `data.root` fields
- Update `train.py` to pass samplesheet path from config

---

## Order of Execution

```
[prerequisite] Extract tarball:
    tar -xJvf data/braak_xy_expression_by_celltype_donor_lognorm_zscore.tar.xz

1. scripts/02_make_samplesheet.py     → data/samplesheet.csv  +  prints BRAAK distribution table
2. scripts/03_normalize_by_split.py   → data/xy_expression_by_celltype_donor_lognorm_zscore_splitwise/
3. Update configs/config.yaml         → add data.samplesheet, update data.root
4. Verify train.py loads correctly
```

---

## Files to Create / Modify

| File | Action |
|---|---|
| `scripts/02_make_samplesheet.py` | Create |
| `scripts/03_normalize_by_split.py` | Create |
| `data/samplesheet.csv` | Generated output |
| `configs/config.yaml` | Add `data.samplesheet` field |
| `utils/data_split.py` | Replace `random_donor_split` with `stratified_donor_split` |

---

## Open Questions

- How many donors per BRAAK level? → answered when `02_make_samplesheet.py` runs and prints distribution
- Is the raw h5ad available on this cluster? (only needed if we ever want to re-run log1p from scratch)
