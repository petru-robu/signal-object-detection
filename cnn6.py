# CNNS 0.6948387096774193
import random
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torchmetrics import Accuracy

# ------ Helpers -------

def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------ Image manipulation stuff -------
def imshow(img):
    npimg = img.numpy()
    # convert RGB ro grayscale
    npimg = np.transpose(npimg, (1, 2, 0))
    npimg = npimg.mean(axis=2)

    plt.imshow(npimg, cmap='viridis')
    plt.show()


def load_image(image_path):
    # image size is 55, 128
    # tensor shape later is: 1, 128, 55
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = img.astype(np.float32) / 255.0
    return img

# Some custom methods to transform an image
class AddGaussianNoise:
    def __init__(self, std=0.02, p=0.25):
        self.std = std
        self.p = p

    def __call__(self, image):
        if torch.rand(1).item() < self.p:
            image = image + self.std * torch.randn_like(image)
            image = torch.clamp(image, 0.0, 1.0)
        return image

class RandomShift2D:
    """Small translation."""

    def __init__(self, max_shift=2, p=0.25):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, image):
        if torch.rand(1).item() >= self.p:
            return image
        dy = int(torch.randint(-self.max_shift,
                 self.max_shift + 1, (1,)).item())
        dx = int(torch.randint(-self.max_shift,
                 self.max_shift + 1, (1,)).item())
        return torch.roll(image, shifts=(dy, dx), dims=(-2, -1))

# ---- Dataset stuff ----

class ObjDetTorchDataset(Dataset):
    def __init__(self, csv_file, has_label=True, img_dir=None, transform=None):
        self.df = pd.read_csv(csv_file)
        self.has_label = has_label
        self.transform = transform
        self.img_dir = img_dir

        if img_dir is None:
            self.img_dir = './data/train' if "train" in csv_file else './data/test'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row["id"]

        img_path = self.img_dir + f"/{img_id}"
        image = load_image(img_path)

        # transform from np to tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if self.transform is not None:
            image = self.transform(image)

        if self.has_label:
            label = int(row["label"]) - 1
            label = torch.tensor(label, dtype=torch.long)
            return image, label

        return image, img_id


def stratified_split_indices(labels, val_split=0.2, seed=42):
    """ Get idx for a stratified split, so classes are equally dsitributed. """
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


def make_class_weights(labels, num_classes=5, penalty_multipliers=None):
    """
    Calculates base weights based on data distribution, then applies a manual multiplier
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    base_weights = counts.sum() / (num_classes * counts)
    
    if penalty_multipliers is not None:
        penalty_array = np.array(penalty_multipliers, dtype=np.float32)
        weights = base_weights * penalty_array
    else:
        weights = base_weights
        
    # normalize weights
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32), counts.astype(int)


def load_dataset(data_dir="./data", batch_size=32, num_workers=2, val_split=0.2, seed=42):
    """ Loads the dataset but for torch required format, data and loader """
    train_csv, train_img_dir = data_dir + "/train.csv", data_dir + "/train"
    test_csv, test_img_dir = data_dir + "/test.csv", data_dir + "/test"

    # data augumentation - my own
    train_transform = transforms.Compose([
        AddGaussianNoise(std=0.02, p=0.25),
        RandomShift2D(max_shift=2, p=0.20)
    ])

    # data augumentation
    # train_transform = v2.Compose([
    #     v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.2)
    # ])

    # torch datasets
    full_train_aug = ObjDetTorchDataset(
        csv_file=train_csv,
        has_label=True,
        img_dir=train_img_dir,
        transform=train_transform
    )

    full_train_eval = ObjDetTorchDataset(
        csv_file=train_csv,
        has_label=True,
        img_dir=train_img_dir,
        transform=None
    )

    test_data = ObjDetTorchDataset(
        csv_file=test_csv,
        has_label=False,
        img_dir=test_img_dir,
        transform=None
    )

    # split
    labels_0_based = full_train_eval.df["label"].astype(int).values - 1

    train_idx, val_idx = stratified_split_indices(
        labels_0_based, val_split=val_split, seed=seed)
    
    train_data = Subset(full_train_aug, train_idx)

    val_data = Subset(full_train_eval, val_idx)

    # data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # weights
    class_weights, class_counts = make_class_weights(labels_0_based[train_idx], num_classes=5)
    return test_data, train_data, val_data, test_loader, train_loader, val_loader, class_weights, class_counts


def check_dataset():
    """ Show images from the dataset. """
    # Loading dataset
    print("Loading dataset...")
    test_data, train_data, val_data, test_loader, train_loader, val_loader = load_dataset()
    images, labels = next(iter(train_loader))
    print("Dataset loaded succesfully!")
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    # show images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))


# ---------------- CNN STUFF ---------------------
class ResBlock(nn.Module):
    """
    A standard Residual Block. 
    If stride > 1, it downsamples the spatial dimensions while increasing the channel count.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        # Main pathway
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut pathway (Skip Connection)
        self.shortcut = nn.Sequential()
        # If we are changing the number of channels or the spatial dimensions, 
        # the shortcut needs a 1x1 convolution to match shapes so we can add them together.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Pass through main pathway
        out = F.silu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        
        # Add the original input (shortcut) back to the output
        out += self.shortcut(x)
        
        # Final activation
        out = F.silu(out, inplace=True)
        return out


class CNN(nn.Module):
    """
    Upgraded ResNet-style CNN for 1-channel images.
    """
    def __init__(self, in_channels=1, num_classes=5, dropout=0.25):
        super(CNN, self).__init__()

        # Initial feature extraction: 1 -> 16 channels (No downsampling yet)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Residual Stages
        # Instead of MaxPool, we use stride=2 in the blocks to learn the downsampling.
        self.layer1 = ResBlock(16, 16, stride=1)           # Spatial size stays the same
        self.layer2 = ResBlock(16, 32, stride=2)           # Downsamples by 2x
        self.layer3 = ResBlock(32, 64, stride=2)           # Downsamples by 2x
        self.layer4 = ResBlock(64, 128, stride=2)          # Downsamples by 2x

        # Global Average Pooling flattens whatever spatial dimensions are left into 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier (Kept identical to your original for consistency)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # 1. Initial Stem
        x = F.silu(self.bn1(self.conv1(x)), inplace=True)

        # 2. Residual Feature Extraction
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 3. Pooling and Classification
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x
    

# ----- Train / Eval / Predict ------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, target in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        targets = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True) 

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward() 


        # prevent spikes?
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


        optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += bs
        
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_targets, all_preds = [], []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

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


def predict_test(model, loader, device):
    model.eval()
    test_ids, test_preds = [], []

    with torch.no_grad():
        for images, img_ids in tqdm(loader, desc="Predicting test"):
            images = images.to(device, non_blocking=True)

            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy() + \
                1  # go back to 1-indexing

            test_ids.extend(list(img_ids))
            test_preds.extend(preds.tolist())

    return pd.DataFrame({"id": test_ids, "label": test_preds})


def confusion_matrix(y_true, y_pred, num_classes=5):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    print("Validation confusion matrix, rows=true 1-5, cols=pred 1-5:")
    print(pd.DataFrame(cm, index=range(1, num_classes + 1),
          columns=range(1, num_classes + 1)))


# ------ ARGS ------
def parse_args():
    """
        To test the model easier we use an arg parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Ignoring unknown notebook arguments: {unknown}")
    return args

def main():
    # ARGS AND SETTINGS
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device is {device}!")

    best_model_path = args.output_dir + "/best_model.pth"
    submission_path = args.output_dir + "/submission.csv"

    # LOADING DATASET
    print("Loading dataset...")
    test_data, train_data, val_data, test_loader, train_loader, val_loader, class_weights, class_counts = load_dataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed
    )
    images, labels = next(iter(train_loader))
    print("Dataset loaded succesfully!")
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print(f"Train class counts 1-5: {class_counts.tolist()}")
    print(f"Class weights: {class_weights.numpy().round(4).tolist()}")

    # MODEL
    model = CNN(in_channels=1, num_classes=5, dropout=args.dropout).to(device)
    print("Model is: \n", model)

    weights = class_weights.to(device) # weights

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # TRAINING LOOP
    best_acc, best_epoch = -1, -1
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc, y_true, y_pred = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train loss: {train_loss}, Train acc: {train_acc}"
            f"Val loss: {val_loss}, Val acc: {val_acc}, Learning rate: {lr}"
        )

        if val_acc > best_acc:
            # save best model
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "epoch": best_epoch,
                    "args": vars(args),
                },
                best_model_path
            )
            print(f"Saved best model: epoch {best_epoch}, val_acc={best_acc}")

    # USE BEST MODEL FOR FINAL PREDICTION
    print(f"\nBest validation accuracy: {best_acc} at epoch {best_epoch}")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, final_val_acc, y_true, y_pred = validate(model, val_loader, criterion, device)

    print(f"Final loaded best-model validation accuracy: {final_val_acc:.4f}")
    confusion_matrix(y_true, y_pred, num_classes=5)

    submission = predict_test(model, test_loader, device)
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission: {submission_path}")
    print(submission.head())


if __name__ == '__main__':
    main()
