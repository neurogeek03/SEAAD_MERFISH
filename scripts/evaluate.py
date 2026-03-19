"""
Evaluate a saved checkpoint on the val set.
Usage:
    uv run python scripts/evaluate.py <path/to/checkpoint.pt>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import torch
import numpy as np
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader
from utils import get_model, MERFISHDataset
from utils.dataset import BRAAK_TO_GROUP

GROUP_NAMES = {0: 'BRAAK 2-3', 1: 'BRAAK 4', 2: 'BRAAK 5-6'}


def confusion_matrix_str(cm, labels):
    """Pretty-print a confusion matrix."""
    header = f"{'':>12}" + "".join(f"{l:>12}" for l in labels)
    lines = [header]
    for i, row in enumerate(cm):
        line = f"{'pred ' + labels[i]:>12}" + "".join(f"{v:>12}" for v in row)
        lines.append(line)
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/evaluate.py <checkpoint.pt>")
        sys.exit(1)

    ckpt_path = Path(sys.argv[1])
    cfg_path  = Path("configs/config.yaml")
    exp_path  = Path("configs/exp_001_reduce_overfit.yaml")

    # Load config
    cfg = OmegaConf.load("configs/default_config.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg_path))
    cfg = OmegaConf.merge(cfg, OmegaConf.load(exp_path))

    device = cfg.device if cfg.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model and load weights
    model = get_model(cfg.model.name, **cfg.model.kwargs)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    # Load val data
    all_files = MERFISHDataset.collect_files(cfg.data.root)
    _, val_files, _ = MERFISHDataset.donor_split(all_files, cfg.data.samplesheet)
    val_loader = DataLoader(
        MERFISHDataset(cfg.data.root, val_files, k=cfg.data.k, device=device),
        batch_size=cfg.batch_size, shuffle=False
    )

    # Evaluate
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            y_hat  = model(batch.x, batch.edge_index, batch.batch)
            preds  = y_hat.argmax(dim=1).cpu().numpy()
            labels = batch.y.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    n_classes  = cfg.model.kwargs.num_classes

    # Accuracy
    acc = (all_preds == all_labels).mean()
    print(f"\nCheckpoint : {ckpt_path.name}")
    print(f"Val loss   : {ckpt['val_loss']:.4f}")
    print(f"Val acc    : {acc:.4f} ({acc*100:.1f}%)")
    print(f"Baseline   : {max(np.bincount(all_labels)) / len(all_labels):.4f} (majority class)")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for c in range(n_classes):
        mask = all_labels == c
        if mask.sum() == 0:
            continue
        c_acc = (all_preds[mask] == c).mean()
        print(f"  {GROUP_NAMES[c]:>12}: {c_acc:.4f}  ({mask.sum()} samples)")

    # Confusion matrix (rows=true, cols=pred)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1
    labels = [GROUP_NAMES[i] for i in range(n_classes)]
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix_str(cm, labels))


if __name__ == "__main__":
    main()
