# HYPERPARAMETER TUNING

import random
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna # <-- Added for hyperparameter tuning

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F

import torchvision
from torchvision import transforms

# ------ Helpers -------
def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ------ Image manipulation stuff -------
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = img.astype(np.float32) / 255.0
    return img

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
    def __init__(self, max_shift=2, p=0.25):
        self.max_shift = max_shift
        self.p = p
    def __call__(self, image):
        if torch.rand(1).item() >= self.p:
            return image
        dy = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
        dx = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
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
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.has_label:
            label = int(row["label"]) - 1
            label = torch.tensor(label, dtype=torch.long)
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

def make_class_weights(labels, num_classes=5, penalty_multipliers=None):
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    base_weights = counts.sum() / (num_classes * counts)
    if penalty_multipliers is not None:
        penalty_array = np.array(penalty_multipliers, dtype=np.float32)
        weights = base_weights * penalty_array
    else:
        weights = base_weights
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32), counts.astype(int)

def load_dataset_objects(data_dir="./data", val_split=0.2, seed=42):
    """ Loads the dataset objects only, loaders are handled in Optuna trials to allow dynamic batch sizes """
    train_csv, train_img_dir = data_dir + "/train.csv", data_dir + "/train"
    test_csv, test_img_dir = data_dir + "/test.csv", data_dir + "/test"

    train_transform = transforms.Compose([
        AddGaussianNoise(std=0.02, p=0.25),
        RandomShift2D(max_shift=2, p=0.20)
    ])

    full_train_aug = ObjDetTorchDataset(csv_file=train_csv, has_label=True, img_dir=train_img_dir, transform=train_transform)
    full_train_eval = ObjDetTorchDataset(csv_file=train_csv, has_label=True, img_dir=train_img_dir, transform=None)
    test_data = ObjDetTorchDataset(csv_file=test_csv, has_label=False, img_dir=test_img_dir, transform=None)

    labels_0_based = full_train_eval.df["label"].astype(int).values - 1
    train_idx, val_idx = stratified_split_indices(labels_0_based, val_split=val_split, seed=seed)
    
    train_data = Subset(full_train_aug, train_idx)
    val_data = Subset(full_train_eval, val_idx)
    class_weights, class_counts = make_class_weights(labels_0_based[train_idx], num_classes=5)
    
    return test_data, train_data, val_data, class_weights, class_counts

# ----- CNN stuff ------
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=5, dropout=0.25):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

# ----- Train / Eval / Predict ------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, target in loader: # Removed tqdm inside epochs during tuning for cleaner logs
        images, targets = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True) 
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward() 
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
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            
            preds = logits.argmax(dim=1)
            bs = images.size(0)
            running_loss += loss.item() * bs
            correct += (preds == targets).sum().item()
            total += bs
    return running_loss / total, correct / total

def predict_test(model, loader, device):
    model.eval()
    test_ids, test_preds = [], []
    with torch.no_grad():
        for images, img_ids in tqdm(loader, desc="Predicting test"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy() + 1
            test_ids.extend(list(img_ids))
            test_preds.extend(preds.tolist())
    return pd.DataFrame({"id": test_ids, "label": test_preds})

# ------ Optuna Objective Function ------
def objective(trial, args, train_data, val_data, class_weights, device):
    # 1. Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # 2. Setup DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    # 3. Initialize Model, Loss, Optimizer
    model = CNN(in_channels=1, num_classes=5, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. Train Loop with TQDM Progress Bar
    best_acc = 0.0
    
    # Create a progress bar for the epochs in THIS trial. 
    # leave=False ensures it disappears when the trial is done.
    epoch_pbar = tqdm(range(args.epochs), desc=f"Trial {trial.number}", leave=False)
    
    for epoch in epoch_pbar:
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            
        # Update the progress bar text with the current metrics
        epoch_pbar.set_postfix(val_acc=f"{val_acc:.4f}", best_acc=f"{best_acc:.4f}")
            
        # Report for Optuna pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            # If pruned, close the bar cleanly before raising the exception
            epoch_pbar.close() 
            raise optuna.exceptions.TrialPruned()

    return best_acc

# ------ ARGS & MAIN ------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dataset objects once
    print("Loading data for tuning...")
    test_data, train_data, val_data, class_weights, _ = load_dataset_objects(
        data_dir=args.data_dir, val_split=args.val_split, seed=args.seed
    )

    print("\n--- Starting Optuna Hyperparameter Tuning ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    
    # Run optimization
    study.optimize(lambda trial: objective(trial, args, train_data, val_data, class_weights, device), n_trials=args.n_trials)
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Validation Accuracy: {trial.value}")
    print("  Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("\n--- Retraining Best Model for Submission ---")
    best_params = trial.params
    train_loader = DataLoader(train_data, batch_size=best_params["batch_size"], shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=best_params["batch_size"], shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=best_params["batch_size"], shuffle=False, num_workers=args.num_workers)

    final_model = CNN(in_channels=1, num_classes=5, dropout=best_params["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=best_params["label_smoothing"], weight=class_weights.to(device))
    optimizer = optim.AdamW(final_model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_final_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(final_model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(final_model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs} - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_final_acc:
            best_final_acc = val_acc
            torch.save(final_model.state_dict(), f"{args.output_dir}/best_model.pth")
    
    # Load best state and predict
    final_model.load_state_dict(torch.load(f"{args.output_dir}/best_model.pth"))
    submission = predict_test(final_model, test_loader, device)
    submission.to_csv(f"{args.output_dir}/submission.csv", index=False)
    print(f"Saved optimal submission to {args.output_dir}/submission.csv")

if __name__ == '__main__':
    main()