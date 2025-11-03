#!/usr/bin/env python3
"""
train_fabric.py

Usage example:
    python train_fabric.py --data-dir ./fabric_dataset --epochs 25 --batch-size 32 --output model_resnet18.pth
Dataset layout (expected):
    fabric_dataset/
        class_0/
            img001.jpg
            ...
        class_1/
            ...
        ...
        class_13/
"""

import argparse
import os
from pathlib import Path
import time
import copy

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -------------------------
# Arguments
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train fabric classifier (transfer learning, ResNet18)")
    p.add_argument("--data-dir", type=str, required=True, help="Root dataset directory (class subfolders)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", type=str, default="fabric_resnet18.pth")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    return p.parse_args()

# -------------------------
# Helpers
# -------------------------
def get_dataloaders(data_dir, img_size=224, batch_size=32, workers=4):
    # Standard transforms + augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # Use ImageFolder; structure: root/<class_name>/*.jpg
    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    # split into train/val (80/20)
    n = len(full_dataset)
    indices = list(range(n))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    def subset_from_indices(dataset, idxs, transform):
        subset = torch.utils.data.Subset(dataset, idxs)
        # monkey patch to set transform on underlying dataset
        subset.dataset.transform = transform
        return subset

    train_ds = subset_from_indices(full_dataset, train_idx, train_transform)
    val_ds = subset_from_indices(full_dataset, val_idx, val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader, class_names

# -------------------------
# Model setup
# -------------------------
def build_model(num_classes, device):
    model = models.resnet18(pretrained=True)
    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)
    return model

# -------------------------
# Training + validation
# -------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=100. * correct / total)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Val", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            pbar.set_postfix(val_loss=running_loss/total, val_acc=100. * correct / total)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_labels, all_preds

# -------------------------
# Utilities
# -------------------------
def save_checkpoint(state, filename):
    torch.save(state, filename)

def plot_metrics(history, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(); plt.plot(epochs, history['train_loss'], label='train_loss'); plt.plot(epochs, history['val_loss'], label='val_loss'); plt.legend(); plt.xlabel('epoch'); plt.savefig(os.path.join(out_dir, 'loss.png')); plt.close()
    plt.figure(); plt.plot(epochs, history['train_acc'], label='train_acc'); plt.plot(epochs, history['val_acc'], label='val_acc'); plt.legend(); plt.xlabel('epoch'); plt.savefig(os.path.join(out_dir, 'acc.png')); plt.close()

# -------------------------
# Main training loop
# -------------------------
def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(args.data_dir, img_size=args.img_size,
                                                           batch_size=args.batch_size, workers=args.workers)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_labels, val_preds = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}  time={elapsed:.1f}s")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'class_names': class_names,
                'val_acc': val_acc,
            }, args.output.replace('.pth', '_best.pth'))
            print(f"Saved new best model (val_acc={val_acc:.4f}) -> {args.output.replace('.pth','_best.pth')}")

        # periodic checkpoint
        if epoch % args.save_every == 0:
            cp_name = args.output.replace('.pth', f'_epoch{epoch}.pth')
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'class_names': class_names,
                'val_acc': val_acc,
            }, cp_name)
            print(f"Saved checkpoint -> {cp_name}")

    # final save
    save_checkpoint({
        'epoch': args.epochs,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'class_names': class_names,
        'val_acc': val_acc,
    }, args.output)
    print(f"Training complete. Final model saved -> {args.output}")

    # Post-training report
    plot_metrics(history)

    # Print classification report & confusion matrix for validation set
    print("\nValidation classification report:")
    print(classification_report(val_labels, val_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(val_labels, val_preds)
    print("Confusion matrix shape:", cm.shape)

if __name__ == "__main__":
    main()
  
