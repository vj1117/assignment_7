import math
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# utility: count params and compute receptive field analytically

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def compute_receptive_field(layers):
    """
    layers: list of dicts with keys: k (kernel), s (stride), d (dilation)
    returns RF. Assumes symmetrical square kernels.
    """
    rf = 1
    stride_prod = 1
    for L in layers:
        k = L.get('k', 1)
        s = L.get('s', 1)
        d = L.get('d', 1)
        k_eff = k + (k - 1) * (d - 1)
        rf = rf + (k_eff - 1) * stride_prod
        stride_prod *= s
    return rf

def get_data_loader(root,args=None):
    # --------------------------
    # Albumentations transforms
    # --------------------------

    # Standard CIFAR-10 mean (RGB)
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
        A.CoarseDropout(
            num_holes=1,
            max_h_size=16, max_w_size=16,
            min_h_size=16, min_w_size=16,
            fill_value=tuple(int(x * 255) for x in CIFAR10_MEAN),
            p=0.5
        ),
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2()
    ])


    val_transform = A.Compose([
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2()
    ])

    train_set = AlbCIFAR10(root, train=True, transform=train_transform, download=True) # Changed download to True
    val_set = AlbCIFAR10(root, train=False, transform=val_transform, download=True) # Changed download to True

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader

# wrapper dataset to apply albumentations
class AlbCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__(root=root, train=train, download=download)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # img is numpy array HWC uint8
        img = Image.fromarray(img)
        img = np.array(img)
        if self.transform:
            aug = self.transform(image=img)
            img = aug['image']
        return img, target


# --------------------------
# Training loop
# --------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # >>> OneCycle step per batch
        if scheduler is not None:
            scheduler.step()
            last_lr = scheduler.get_last_lr()  # list of LRs per param group
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total, (last_lr if last_lr is not None else [pg['lr'] for pg in optimizer.param_groups])


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total


# --------------------------
# Main
# --------------------------

def main(args, model=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    params = count_params(model)
    print(f"Total parameters: {params:,}")

    # compute receptive field for the 4 conv layers
    layers = [
        {'k':3, 's':1, 'd':1},    # C1
        {'k':3, 's':1, 'd':1},    # C1
        {'k':3, 's':1, 'd':1},    # C2 
        {'k':3, 's':1, 'd':1},    # C2 
        {'k':1, 's':2, 'd':1},    # C2 
        {'k':3, 's':1, 'd':1},    # C3
        {'k':3, 's':1, 'd':1},    # C3
        {'k':1, 's':2, 'd':1},    # C3
        {'k':2, 's':1, 'd':1},    # C4
        {'k':3, 's':1, 'd':3},    # C4
    ]
    rf = compute_receptive_field(layers)
    print(f"Estimated receptive field: {rf}")

    # Data
    root = args.data_dir
    train_loader, val_loader = get_data_loader(root, args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=args.pct_start,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
        anneal_strategy='cos',          # works well in practice
        cycle_momentum=True,            # momentum up/down with LR (SGD)
        base_momentum=0.85,
        max_momentum=0.95
    )


    # --- tracking containers ---
    history = {
        "epoch": [],
        "lr": [],            # if multiple param groups, will store a list per epoch
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    
    best_acc = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):

        train_loss, train_acc, batch_lrs = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # record metrics
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        #print(f"Epoch {epoch}/{args.epochs}: Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")

        # log the **last batch** LR(s) of this epoch (typical summary)
        history["lr"].append(batch_lrs if len(batch_lrs) > 1 else batch_lrs[0])
        lr_str = ", ".join(f"{lr:.6f}" for lr in batch_lrs)
        
        # pretty print: show LR(s)
        #lr_str = ", ".join(f"{lr:.6f}" for lr in (current_lrs if isinstance(current_lrs, list) else [current_lrs]))
        print(
            f"Epoch {epoch}/{args.epochs} | LR(s): [{lr_str}] "
            f"| Train loss {train_loss:.4f} acc {train_acc:.4f} "
            f"| Val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state': model.state_dict(), 'acc': best_acc}, save_dir / 'best2.pth')

    print(f"Best val acc: {best_acc:.4f}")
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--save-dir', default='./checkpoints2', type=str)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max-lr', default=0.02, type=float)      # peak LR in the cycle
    parser.add_argument('--pct-start', default=0.3, type=float)    # % of steps to reach max LR
    parser.add_argument('--div-factor', default=25.0, type=float)  # initial_lr = max_lr/div_factor
    parser.add_argument('--final-div-factor', default=1000, type=float)  # min lr = max_lr/final_div_factor

    args = parser.parse_args(args=[])

    history = main(args)

    save_dir = "./checkpoints2"
    # --- plots ---
    # 1) Loss curves
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train loss")
    plt.plot(history["epoch"], history["val_loss"], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    #plt.savefig(save_dir / "loss_curve.png", dpi=150)
    
    # 2) Accuracy curves
    plt.figure()
    plt.plot(history["epoch"], history["train_acc"], label="train acc")
    plt.plot(history["epoch"], history["val_acc"], label="val acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    #plt.savefig(save_dir / "accuracy_curve.png", dpi=150)