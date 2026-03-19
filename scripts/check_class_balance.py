import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
from collections import Counter
from utils.dataset import MERFISHDataset, BRAAK_TO_GROUP

files = MERFISHDataset.collect_files('data/braak_xy_expression_by_celltype_donor_lognorm')
train, val, test = MERFISHDataset.donor_split(files, 'data/samplesheet.csv')

group_names = {0: '2-3', 1: '4', 2: '5-6'}

for name, split in [('train', train), ('val', val), ('test', test)]:
    labels = [BRAAK_TO_GROUP[int(pd.read_csv(f, usecols=['BRAAK_score'])['BRAAK_score'].iloc[0])] for f in split]
    counts = Counter(labels)
    total = sum(counts.values())
    print(f"\n{name} ({total} graphs):")
    for group_idx in sorted(counts):
        pct = 100 * counts[group_idx] / total
        print(f"  class {group_idx} ({group_names[group_idx]}): {counts[group_idx]} ({pct:.1f}%)")
