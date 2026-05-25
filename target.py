# Kaggle-ready CNN for 5-class radio-signal object counting
# Expected data layout:
#   DATA_DIR/train.csv with columns: id,label
#   DATA_DIR/test.csv with column: id
#   DATA_DIR/train/<id>
#   DATA_DIR/test/<id>

import os
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
from torchvision import transforms


# ---------------- Config helpers ----------------


def default_data_dir():
    candidates = [
        "/kaggle/input/datasets/robupetrurazvan/kg-sgn-obj/data",
        "/kaggle/input/kg-sgn-obj/data",
        "/kaggle/input/data",
        "./data",
    ]
    for path in candidates:
        if Path(path, "train.csv").exists() and Path(path, "test.csv").exists():
            return path
    return candidates[0]


def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        # Faster on Kaggle when all images have same shape.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


# ---------------- Image loading / augmentation ----------------


def load_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img.astype(np.float32) / 255.0


class AddGaussianNoise:
    def _init_(self, std=0.02, p=0.25):
        self.std = std
        self.p = p

    def _call_(self, image):
        if torch.rand(1).item() < self.p:
            image = image + self.std * torch.randn_like(image)
            image = torch.clamp(image, 0.0, 1.0)
        return image


class RandomShift2D:
    """Small translation augmentation. Avoids rotations/flips, which may be invalid for signals."""

    def _init_(self, max_shift=2, p=0.25):
        self.max_shift = max_shift
        self.p = p

    def _call_(self, image):
        if torch.rand(1).item() >= self.p:
            return image
        dy = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
        dx = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
        return torch.roll(image, shifts=(dy, dx), dims=(-2, -1))


# ---------------- Dataset ----------------


class ObjDetTorchDataset(Dataset):
    def _init_(self, csv_file, img_dir, has_label=True, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.has_label = has_label
        self.transform = transform

    def _len_(self):
        return len(self.df)

    def _getitem_(self, index):
        row = self.df.iloc[index]
        img_id = str(row["id"])
        image = load_image(self.img_dir / img_id)

        # OpenCV returns [H, W]. Tensor becomes [1, H, W].
        image = torch.from_numpy(image).float().unsqueeze(0)

        if self.transform is not None:
            image = self.transform(image)

        if self.has_label:
            # Convert labels 1-5 to 0-4 for CrossEntropyLoss.
            label = torch.tensor(int(row["label"]) - 1, dtype=torch.long)
            return image, label

        return image, img_id


def stratified_split_indices(labels, val_split=0.2, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_indices, val_indices = [], []

    for label in np.unique(labels):
        class_indices = np.where(labels == label)[0]
        rng.shuffle(class_indices)
        n_val = max(1, int(round(len(class_indices) * val_split)))
        val_indices.extend(class_indices[:n_val].tolist())
        train_indices.extend(class_indices[n_val:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def make_class_weights(labels, num_classes=5):
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32), counts.astype(int)


def make_loader(dataset, batch_size, shuffle, num_workers, seed=None):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if seed is not None:
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def load_dataset(data_dir, batch_size=64, num_workers=2, val_split=0.2, seed=42):
    data_dir = Path(data_dir)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    train_img_dir = data_dir / "train"
    test_img_dir = data_dir / "test"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Could not find train.csv/test.csv in {data_dir}. Set --data-dir correctly."
        )

    train_transform = transforms.Compose([
        AddGaussianNoise(std=0.02, p=0.25),
        RandomShift2D(max_shift=2, p=0.20),
    ])

    full_train_aug = ObjDetTorchDataset(train_csv, train_img_dir, has_label=True, transform=train_transform)
    full_train_eval = ObjDetTorchDataset(train_csv, train_img_dir, has_label=True, transform=None)
    test_data = ObjDetTorchDataset(test_csv, test_img_dir, has_label=False, transform=None)

    labels_0_based = full_train_eval.df["label"].astype(int).values - 1
    train_idx, val_idx = stratified_split_indices(labels_0_based, val_split=val_split, seed=seed)

    train_data = Subset(full_train_aug, train_idx)
    val_data = Subset(full_train_eval, val_idx)

    class_weights, class_counts = make_class_weights(labels_0_based[train_idx], num_classes=5)

    train_loader = make_loader(train_data, batch_size, True, num_workers, seed=seed)
    val_loader = make_loader(val_data, batch_size, False, num_workers)
    test_loader = make_loader(test_data, batch_size, False, num_workers)

    return test_data, train_data, val_data, test_loader, train_loader, val_loader, class_weights, class_counts


# ---------------- Model ----------------


class CNN(nn.Module):
    def _init_(self, in_channels=1, num_classes=5, dropout=0.25):
        super()._init_()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)


# ---------------- Train / validation / prediction ----------------


def amp_context(enabled):
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def make_scaler(enabled):
    if hasattr(torch, "amp"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, targets in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_context(use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        running_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += bs

    return running_loss / total, correct / total


def validate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_targets, all_preds = [], []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with amp_context(use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            preds = logits.argmax(dim=1)
            bs = images.size(0)
            running_loss += loss.item() * bs
            correct += (preds == targets).sum().item()
            total += bs

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    return running_loss / total, correct / total, np.array(all_targets), np.array(all_preds)


def predict_test(model, loader, device, use_amp):
    model.eval()
    test_ids, test_preds = [], []

    with torch.no_grad():
        for images, img_ids in tqdm(loader, desc="Predicting test"):
            images = images.to(device, non_blocking=True)
            with amp_context(use_amp):
                logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy() + 1
            test_ids.extend(list(img_ids))
            test_preds.extend(preds.tolist())

    return pd.DataFrame({"id": test_ids, "label": test_preds})


def print_confusion_matrix(y_true, y_pred, num_classes=5):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    print("Validation confusion matrix, rows=true 1-5, cols=pred 1-5:")
    print(pd.DataFrame(cm, index=range(1, num_classes + 1), columns=range(1, num_classes + 1)))


# ---------------- Main ----------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=default_data_dir())
    parser.add_argument("--output-dir", type=str, default="/kaggle/working" if Path("/kaggle").exists() else ".")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--deterministic", action="store_true")

    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Ignoring unknown notebook arguments: {unknown}")
    return args

def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pth"
    submission_path = output_dir / "submission.csv"

    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")

    test_data, train_data, val_data, test_loader, train_loader, val_loader, class_weights, class_counts = load_dataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )

    images, labels = next(iter(train_loader))
    print(f"Image batch shape: {tuple(images.shape)}")
    print(f"Label batch shape: {tuple(labels.shape)}")
    print(f"Train size: {len(train_data)} | Validation size: {len(val_data)} | Test size: {len(test_data)}")
    print(f"Train class counts 1-5: {class_counts.tolist()}")
    print(f"Class weights: {class_weights.numpy().round(4).tolist()}")

    model = CNN(in_channels=1, num_classes=5, dropout=args.dropout).to(device)

    weight = None if args.no_class_weights else class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=4, factor=0.5)
    scaler = make_scaler(use_amp)

    best_acc = -1.0
    best_epoch = -1

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_acc, y_true, y_pred = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step(val_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | LR: {lr:.6f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_acc": best_acc,
                "epoch": best_epoch,
                "args": vars(args),
            }, best_model_path)
            print(f"Saved best model: epoch {best_epoch}, val_acc={best_acc:.4f}")

    print(f"\nBest validation accuracy: {best_acc:.4f} at epoch {best_epoch}")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, final_val_acc, y_true, y_pred = validate(model, val_loader, criterion, device, use_amp)
    print(f"Final loaded best-model validation accuracy: {final_val_acc:.4f}")
    print_confusion_matrix(y_true, y_pred, num_classes=5)

    submission = predict_test(model, test_loader, device, use_amp)
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission: {submission_path}")
    print(submission.head())


if _name_ == "_main_":
    main()