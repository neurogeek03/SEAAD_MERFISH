import os
import random
import pandas as pd

# PRELIMINARY
def random_donor_split(
    file_list: list[str],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split file_list by donor into train/val/test using random assignment.
    A donor is never split across sets — all their sections land in the same split.

    Args:
        file_list: All CSV paths (output of MERFISHDataset.collect_files).
        val_frac:  Fraction of donors assigned to validation set.
        test_frac: Fraction of donors assigned to test set.
        seed:      Random seed for reproducible shuffling.

    Returns:
        (train_files, val_files, test_files) — each a sorted list of CSV paths.
    """
    donor_to_files: dict[str, list[str]] = {}
    for fpath in file_list:
        df = pd.read_csv(fpath, nrows=1)
        donor = str(df['Donor_ID'].iloc[0])
        donor_to_files.setdefault(donor, []).append(fpath)

    donors = sorted(donor_to_files.keys())
    rng = random.Random(seed)
    rng.shuffle(donors)

    n = len(donors)
    n_val = max(1, round(n * val_frac))
    n_test = max(1, round(n * test_frac))

    val_donors = set(donors[:n_val])
    test_donors = set(donors[n_val: n_val + n_test])
    train_donors = set(donors[n_val + n_test:])

    train_files, val_files, test_files = [], [], []
    for donor, files in donor_to_files.items():
        if donor in train_donors:
            train_files.extend(files)
        elif donor in val_donors:
            val_files.extend(files)
        elif donor in test_donors:
            test_files.extend(files)

    return sorted(train_files), sorted(val_files), sorted(test_files)
