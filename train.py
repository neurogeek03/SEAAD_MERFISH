import sys
import torch
from utils.callbacks import EarlyStopping
from utils import get_optim, get_scheduler, get_model, get_loss_fn
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
from tqdm import tqdm
from typing import Callable


from torch_geometric.datasets import TUDataset 
from torch_geometric.loader import DataLoader


def get_config() -> ListConfig | DictConfig:
    
    default_config_path = "configs/config.yaml"

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
        self.scheduler = get_scheduler(
            cfg.scheduler.name, 
            self.optimizer, 
            **cfg.scheduler.kwargs
        )
        self.loss_fn = get_loss_fn(
            cfg.loss.name,
            **cfg.loss.kwargs
        )
        
        self.cfg = cfg
        self.loss_epoch = []
        self.val_loss = []
    
    
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
        for x in tqdm(val_loader):
            y_hat = self.model(x.x, x.edge_index, x.batch)
            loss = self.loss_fn(y_hat, x.y)
            loss_iters += loss.item()
        self.val_loss.append(loss_iters / len(val_loader))
        
    @property
    def last_val_loss(self):
        return self.val_loss[-1]
        




if __name__ == '__main__':
    
    cfg = get_config()
    
    trainer = Train(cfg)

    
    if cfg.callbacks.use_early_stopping:
        stopper = EarlyStopping(cfg.callbacks.early_stopping_epochs, use=True)
    else: 
        stopper = EarlyStopping(use=False)
    
    
    
    
    
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    train_dataset = dataset[:150]
    val_dataset = dataset[150:]
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    
    
    
    
    
    
    # train_loader: torch.utils.data.DataLoader = None# TODO
    # val_loader: torch.utils.data.DataLoader = None # TODO
    # loss_fn: Callable = None # TODO
    
    train_losses = []
    val_losses = []
    for final_model_epochs in tqdm(range(cfg.num_epochs)):
        
        trainer.train(train_loader)
        with torch.no_grad():
            trainer.val(val_loader)
        
        stopper.check_if_stop(trainer.last_val_loss)
        