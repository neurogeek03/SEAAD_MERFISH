import os
import torch
import pandas as pd
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import knn_graph
from torch_geometric.transforms import Distance
import torch.nn.functional as F

# params
META_COLS = {'Subclass', 'Section', 'Donor_ID', 'BRAAK_score', 'x', 'y'}


class MERFISHDataset(Dataset):

    def __init__(self, root: str, file_list: list[str], device: str=None, k: int = 10, 
                 transform=None, pre_transform=None, ):
        self.file_list = file_list
        self.k = k
        # super().__init__ must come AFTER setting instance attributes,
        # because it calls process() which uses self.file_list and self.k
        super().__init__(root, transform, pre_transform)
        self.data_pts = []
        for i, fpath in enumerate(self.file_list):
            df = pd.read_csv(fpath)

            # removing blank probes - used for False-positive detection
            blank_cols = [c for c in df.columns if c.startswith('Blank-')]
            drop_cols = set(blank_cols) | META_COLS
            gene_cols = [c for c in df.columns if c not in drop_cols]

            # separating features from x,y coords and labels (BRAAK)
            # + converting to tensors
            x = torch.tensor(df[gene_cols].values, dtype=torch.float)
            pos = torch.tensor(df[['x', 'y']].values, dtype=torch.float)
            y = F.one_hot(
                torch.tensor(int(df['BRAAK_score'].iloc[0]), dtype=torch.long),
                num_classes=7,
            )
            edge_index = knn_graph(pos, k=self.k, loop=False)

            data = Data(x=x, edge_index=edge_index, y=y.unsqueeze(0).to(torch.float), pos=pos)
            data = Distance()(data)
            if device is not None:
                self.data_pts.append(data.to(device))
            else:
                self.data_pts.append(data.to('cuda' if torch.cuda.is_available() else 'cpu'))

    @property
    def processed_file_names(self) -> list[str]:
        # One .pt file per CSV: {subclass}_{section}.pt
        names = []
        for fpath in self.file_list:
            parts = fpath.replace('\\', '/').split('/')
            subclass = parts[-2].replace(' ', '_')
            section = os.path.splitext(parts[-1])[0]
            names.append(f"{subclass}_{section}.pt")
        return names


    # --- Standard access ---

    def len(self) -> int:
        return len(self.file_list)

    def get(self, idx: int) -> Data:
        return self.data_pts[idx]

    # --- Split helpers ---

    @staticmethod
    def collect_files(data_root: str, subclasses: list[str] | None = None) -> list[str]:
        """Return sorted list of all CSV paths under data_root/{subclass}/*.csv."""
        files = []
        subclass_dirs = subclasses if subclasses is not None else sorted(os.listdir(data_root))
        for subclass in subclass_dirs:
            subclass_path = os.path.join(data_root, subclass)
            if not os.path.isdir(subclass_path):
                continue
            for fname in sorted(os.listdir(subclass_path)):
                if fname.endswith('.csv'):
                    files.append(os.path.join(subclass_path, fname))
        return files

    @staticmethod
    def donor_split(
        file_list: list[str],
        samplesheet: str,
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Split file_list into train/val/test using a samplesheet CSV.

        Expected samplesheet columns (in order):
            donor_id  |  section_id  |  split
        where split is one of: training, val, test

        Each file is matched to a split via its section_id (the CSV filename
        without extension). Files whose section_id is not in the samplesheet
        are silently skipped with a warning.
        """
        ss = pd.read_csv(samplesheet, header=0)
        ss.columns = ['donor_id', 'section_id', 'split']
        section_to_split = dict(zip(ss['section_id'].astype(str), ss['split'].str.strip()))

        train_files, val_files, test_files = [], [], []
        for fpath in file_list:
            section = os.path.splitext(os.path.basename(fpath))[0]
            split = section_to_split.get(section)
            if split is None:
                print(f"Warning: {section} not found in samplesheet — skipping")
                continue
            if split == 'training':
                train_files.append(fpath)
            elif split == 'val':
                val_files.append(fpath)
            elif split == 'test':
                test_files.append(fpath)
            else:
                print(f"Warning: unknown split value '{split}' for {section} — skipping")

        return sorted(train_files), sorted(val_files), sorted(test_files)