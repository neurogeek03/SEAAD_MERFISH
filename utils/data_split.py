import os
import random
import pandas as pd


def random_donor_split(
    file_list: list[str],
    output_path: str,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split file_list by donor into train/val/test using random assignment,
    then write a samplesheet CSV recording the result.

    A donor is never split across sets — all their sections land in the same split.

    Args:
        file_list:   All CSV paths (output of MERFISHDataset.collect_files).
        output_path: Where to write the samplesheet CSV.
        val_frac:    Fraction of donors assigned to validation set.
        test_frac:   Fraction of donors assigned to test set.
        seed:        Random seed for reproducible shuffling.

    Returns:
        (train_files, val_files, test_files) — each a sorted list of CSV paths.

    Samplesheet columns:
        donor_id  |  section_id  |  split
    where split is one of: training, val, test
    """
    # --- Build donor → files mapping ---
    donor_to_files: dict[str, list[str]] = {}
    file_to_donor: dict[str, str] = {}
    for fpath in file_list:
        df = pd.read_csv(fpath, nrows=1)
        donor = str(df['Donor_ID'].iloc[0])
        donor_to_files.setdefault(donor, []).append(fpath)
        file_to_donor[fpath] = donor

    # --- Randomly assign donors to splits ---
    donors = sorted(donor_to_files.keys())
    rng = random.Random(seed)
    rng.shuffle(donors)

    n = len(donors)
    n_val = max(1, round(n * val_frac))
    n_test = max(1, round(n * test_frac))

    val_donors = set(donors[:n_val])
    test_donors = set(donors[n_val: n_val + n_test])
    train_donors = set(donors[n_val + n_test:])

    donor_to_split = {}
    for d in train_donors:
        donor_to_split[d] = 'training'
    for d in val_donors:
        donor_to_split[d] = 'val'
    for d in test_donors:
        donor_to_split[d] = 'test'

    # --- Collect files per split ---
    train_files, val_files, test_files = [], [], []
    for donor, files in donor_to_files.items():
        if donor in train_donors:
            train_files.extend(files)
        elif donor in val_donors:
            val_files.extend(files)
        elif donor in test_donors:
            test_files.extend(files)

    # --- Write samplesheet ---
    rows = []
    for fpath in file_list:
        donor = file_to_donor[fpath]
        section = os.path.splitext(os.path.basename(fpath))[0]
        rows.append({
            'donor_id': donor,
            'section_id': section,
            'split': donor_to_split[donor],
        })

    pd.DataFrame(rows).sort_values(['split', 'donor_id', 'section_id']).to_csv(
        output_path, index=False
    )
    print(f"Samplesheet written to {output_path}")

    return sorted(train_files), sorted(val_files), sorted(test_files)