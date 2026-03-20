import sys
import re
import torch
from utils import MERFISHDataset, Trainer, AllMERFISHDataset
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path
from torch_geometric.loader import DataLoader


def get_next_run_name(work_dir: str) -> str:
    work = Path(work_dir)
    if not work.exists():
        return "run_1"
    existing = [d.name for d in work.iterdir() if d.is_dir() and re.match(r"run_\d+", d.name)]
    if not existing:
        return "run_1"
    nums = [int(re.search(r"\d+", name).group()) for name in existing]
    return f"run_{max(nums) + 1}"


def get_config() -> ListConfig | DictConfig:
    """
    Merge chain: default_config.yaml → base config (argv[1]) → experiment config (argv[2])
    Each layer overrides only the keys it specifies.
    Usage:
        python train.py                                         # defaults only
        python train.py configs/config.yaml                    # base override
        python train.py configs/config.yaml configs/exp.yaml   # base + experiment
    """
    default_config_path = "configs/default_config.yaml"

    cfg = OmegaConf.load(default_config_path)

    for arg in sys.argv[1:]:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(arg))

    OmegaConf.update(cfg, "run_name", get_next_run_name(cfg.work_dir))

    return cfg

class MERFISHTrainer(Trainer):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def model_forward(self, batch):
        # batch = batch.to(self.cfg.device)
        # print(batch.x.shape)
        y_hat = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = self.loss_fn(y_hat, batch.y)
        
        return loss, batch.num_graphs 

        

if __name__ == '__main__':
    
    cfg = get_config()
    for cell_type in Path('data/braak_xy_expression_by_celltype_donor_lognorm').iterdir():
        trainer = MERFISHTrainer(cfg)
        print(cell_type.name)
        try:
            all_files = MERFISHDataset.collect_files(cfg.data.root, [cell_type.name])
            train_files, val_files, test_files = MERFISHDataset.donor_split(all_files, cfg.data.samplesheet)

            train_loader = DataLoader(MERFISHDataset(cfg.data.root, train_files, k=cfg.data.k, device=cfg.device),
                                    batch_size=cfg.batch_size, shuffle=True)
            val_loader   = DataLoader(MERFISHDataset(cfg.data.root, val_files, k=cfg.data.k, device=cfg.device),
                                    batch_size=cfg.batch_size, shuffle=False)
            test_loader  = DataLoader(MERFISHDataset(cfg.data.root, test_files, k=cfg.data.k, device=cfg.device),
                                    batch_size=cfg.batch_size, shuffle=False)
            # train_loader = DataLoader(AllMERFISHDataset(cfg.data.root, train_files, k=cfg.data.k),
            #                         batch_size=cfg.batch_size, shuffle=True)
            # val_loader   = DataLoader(AllMERFISHDataset(cfg.data.root, val_files, mean=train_loader.dataset.mean, std=train_loader.dataset.std, k=cfg.data.k),
            #                         batch_size=cfg.batch_size, shuffle=False)
            # test_loader  = DataLoader(AllMERFISHDataset(cfg.data.root, test_files, k=cfg.data.k),
            #                         batch_size=cfg.batch_size, shuffle=False)

        except Exception:
            continue
        
        with tqdm(range(cfg.num_epochs)) as pbar:
            for final_model_epochs in pbar:
                trainer.train(train_loader)
                pbar.set_postfix({"train_loss": f"{trainer.loss_epoch[-1]:.4f}", "val_loss": f"{trainer.last_val_loss:.4f}"}, refresh=False)
                with torch.no_grad():
                    should_stop = trainer.val(val_loader)
                if should_stop:
                    print(f'Stopped after {final_model_epochs} epochs due to early stopping.')
                    break        
        trainer.training_summary(final_model_epochs)