"""Train the ResNet, keep the best epoch by validation accuracy, write metrics and test predictions."""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm

from data import NUM_CLASSES, make_loaders
from model import ResNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += images.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, total = 0.0, 0
    all_targets, all_probs = [], []
    for images, targets in tqdm(loader, desc="val", leave=False):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss_sum += criterion(logits, targets).item() * images.size(0)
        total += images.size(0)
        all_targets.append(targets.cpu())
        all_probs.append(logits.softmax(dim=1).cpu())
    y_true, probs = torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy()
    return loss_sum / total, float((y_true == probs.argmax(axis=1)).mean()), y_true, probs


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ids, preds = [], []
    for images, img_ids in tqdm(loader, desc="test"):
        logits = model(images.to(device))
        preds.extend((logits.argmax(dim=1).cpu().numpy() + 1).tolist())  # back to labels 1 to 5
        ids.extend(img_ids)
    return pd.DataFrame({"id": ids, "label": preds})


def save_val_predictions(path, ids, y_true, probs):
    """Labels 1 to 5 plus the softmax probability of every class, one row per image."""
    df = pd.DataFrame({"id": ids, "label": y_true + 1, "pred": probs.argmax(axis=1) + 1})
    for c in range(NUM_CLASSES):
        df[f"prob_{c + 1}"] = probs[:, c]
    df.to_csv(path, index=False)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--epochs", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = f"{args.output_dir}/best_model.pth"
    predictions_path = f"{args.output_dir}/predictions.csv"
    metrics_path = f"{args.output_dir}/metrics.csv"
    val_predictions_path = f"{args.output_dir}/val_predictions.csv"
    val_probs_path = f"{args.output_dir}/val_probs.npy"

    train_loader, val_loader, test_loader, weights, counts = make_loaders(
        args.data_dir, args.batch_size, args.num_workers, args.val_split, args.seed
    )
    print(f"Train class counts: {counts.tolist()}")
    print(f"Class weights: {weights.numpy().round(3).tolist()}")

    model = ResNet(num_classes=NUM_CLASSES, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    with open(f"{args.output_dir}/config.json", "w") as f:
        json.dump({**vars(args), "device": device, "parameters": n_params}, f, indent=2)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_acc, best_epoch = -1.0, 0
    history, val_probs = [], []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, epoch_probs = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        val_probs.append(epoch_probs)
        np.save(val_probs_path, np.stack(val_probs))  # (epochs, val images, classes), any val metric per epoch
        print(
            f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} "
            f"| val loss {val_loss:.4f} acc {val_acc:.4f}"
        )
        history.append(
            dict(epoch=epoch, lr=lr, train_loss=train_loss, train_acc=train_acc,
                 val_loss=val_loss, val_acc=val_acc, seconds=time.time() - start)
        )
        pd.DataFrame(history).to_csv(metrics_path, index=False)  # rewritten each epoch, survives a crash
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), model_path)

    print(f"Best validation accuracy {best_acc:.4f} at epoch {best_epoch}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    _, _, y_true, probs = evaluate(model, val_loader, criterion, device)
    val_set = val_loader.dataset  # Subset of the train set, same order as the loader
    val_ids = val_set.dataset.df["id"].to_numpy()[val_set.indices]
    save_val_predictions(val_predictions_path, val_ids, y_true, probs)

    predict(model, test_loader, device).to_csv(predictions_path, index=False)
    print(f"Saved {metrics_path}, {val_predictions_path}, {predictions_path}")


if __name__ == "__main__":
    main()
