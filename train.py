import sys
import torch
from torch import nn, optim
from utils.callbacks import EarlyStopping
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
from tqdm import tqdm
from typing import Callable

def get_config() -> ListConfig | DictConfig:
    
    default_config_path = "config/config.yaml"

    cfg = OmegaConf.load(default_config_path)

    if len(sys.argv) > 1:
        user_cfg = OmegaConf.load(sys.argv[1])
        cfg = OmegaConf.merge(cfg, user_cfg)

    return cfg

if __name__ == '__main__':
    
    cfg = get_config()
    
    model: nn.Module = None# TODO write get model
    optimizer: optim.Optimizer = None # TODO write get optim
    scheduler: optim.lr_scheduler.LRScheduler = None # TODO write get scheduler
    train_loader: torch.utils.data.DataLoader = None# TODO
    val_loader: torch.utils.data.DataLoader = None # TODO
    loss_fn: Callable = None # TODO
    
    
    if cfg.use_early_stopping:
        stopper = EarlyStopping(cfg.early_stopping_epochs, use=True)
    else: 
        stopper = EarlyStopping(use=False)
    
    train_losses = []
    val_losses = []
    for final_model_epochs in tqdm(range(cfg.num_epochs)):
        loss_iters = 0
        model.train()
        for x, y in train_loader:
            y_hat = model(x)
            loss = loss_fn(y, y_hat)
            loss_iters += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        train_losses.append(loss_iters / len(train_loader))


        with torch.no_grad():
            model.eval()
            loss_iters = 0
            for x, y in tqdm(val_loader):
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                loss_iters += loss.item()
                val_losses.append(loss_iters / len(val_loader))

        if stopper.check_if_stop(val_losses[-1]):
            break
        
        