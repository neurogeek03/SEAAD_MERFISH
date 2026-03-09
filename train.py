import sys
import torch
from utils.data_split import random_donor_split
from utils.callbacks import EarlyStopping
from utils import get_optim, get_scheduler, get_model, get_loss_fn, MERFISHDataset
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path

from torch_geometric.loader import DataLoader


def get_config() -> ListConfig | DictConfig:
    
    default_config_path = "configs/default_config.yaml"

    cfg = OmegaConf.load(default_config_path)

    if len(sys.argv) > 1:
        user_cfg = OmegaConf.load(sys.argv[1])
        cfg = OmegaConf.merge(cfg, user_cfg)

    return cfg


class Train():
    
    def __init__(self, cfg: ListConfig | DictConfig):
        self.model = get_model(
            cfg.model.name, 
            **cfg.model.kwargs 
        )
        self.optimizer = get_optim(
            cfg.optim.name, 
            self.model.parameters(), 
            lr=cfg.optim.lr,
            **cfg.optim.kwargs
        )
        self.loss_fn = get_loss_fn(
            cfg.loss.name,
            **cfg.loss.kwargs
        )
        
        self.scheduler = None
        if cfg.scheduler is not None:
            self.scheduler = get_scheduler(
                cfg.scheduler.name, 
                self.optimizer, 
                **cfg.scheduler.kwargs
            )
        
        self.cfg = cfg
        self.loss_epoch = []
        self.val_loss = []
        if cfg.device is not None:
            self.model.to(cfg.device)
        else:
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def after_train_batch(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()

    
    def train(self, train_loader: DataLoader):
        loss_iters = 0
        self.model.train()
        for x in train_loader:
            y_hat = self.model(x.x, x.edge_index, x.batch)
            loss = self.loss_fn(y_hat, x.y)
            loss_iters += loss.item()
            self.after_train_batch(loss)
        self.loss_epoch.append(loss_iters / len(train_loader))
        
        
    def val(self, val_loader: DataLoader):
        self.model.eval()
        loss_iters = 0
        for x in val_loader:
            y_hat = self.model(x.x, x.edge_index, x.batch)
            loss = self.loss_fn(y_hat, x.y)
            loss_iters += loss.item()
        self.val_loss.append(loss_iters / len(val_loader))
        
    @property
    def last_val_loss(self):
        return self.val_loss[-1] if self.val_loss else 0





if __name__ == '__main__':
    
    cfg = get_config()
    trainer = Train(cfg)

    if cfg.callbacks.use_early_stopping:
        stopper = EarlyStopping(cfg.callbacks.early_stopping_epochs, use=True)
    else: 
        stopper = EarlyStopping(use=False)
    
    train, val, test = random_donor_split(
        file_list=[str(x) for x in Path('data/braak_xy_expression_by_celltype_donor_lognorm_zscore/Astrocyte').glob('*.csv')],
        val_frac=0.1,
        test_frac=0.2
    )
    
    train_loader = DataLoader(MERFISHDataset(cfg.data.root, train, k=cfg.data.k, device=cfg.device),
                            batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(MERFISHDataset(cfg.data.root, val, k=cfg.data.k, device=cfg.device),
                            batch_size=cfg.batch_size, shuffle=False)

    
    with tqdm(range(cfg.num_epochs)) as pbar:
        for final_model_epochs in pbar:
            trainer.train(train_loader)
            pbar.set_postfix({"train_loss": f"{trainer.loss_epoch[-1]:.4f}", "val_loss": f"{trainer.last_val_loss:.4f}"}, refresh=False)
            with torch.no_grad():
                trainer.val(val_loader)        
            stopper.check_if_stop(trainer.last_val_loss)
    