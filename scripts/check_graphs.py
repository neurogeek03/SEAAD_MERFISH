 
from utils.dataset import MERFISHDataset

files = MERFISHDataset.collect_files('data/braak_xy_expression_by_celltype_donor_lognorm') 
train, val, test = MERFISHDataset.donor_split(files, 'data/samplesheet.csv')
print('train graphs:', len(train))
