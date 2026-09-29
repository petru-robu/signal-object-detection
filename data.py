"""Dataset, augmentation, stratified split and data loaders."""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

NUM_CLASSES = 5


def load_image(path):
    """Read a PNG as grayscale and scale it to [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img.astype(np.float32) / 255.0


class Augment:
    """Random Gaussian noise and a small random shift, applied independently."""

    def __init__(self, noise_std=0.02, noise_p=0.25, max_shift=2, shift_p=0.20):
        self.noise_std, self.noise_p = noise_std, noise_p
        self.max_shift, self.shift_p = max_shift, shift_p

    def __call__(self, image):
        if torch.rand(1).item() < self.noise_p:
            image = (image + self.noise_std * torch.randn_like(image)).clamp(0.0, 1.0)
        if torch.rand(1).item() < self.shift_p:
            dy, dx = torch.randint(-self.max_shift, self.max_shift + 1, (2,)).tolist()
            image = torch.roll(image, shifts=(dy, dx), dims=(-2, -1))
        return image


class SignalDataset(Dataset):
    """Yields (image, label) for labeled data and (image, id) for test data."""

    def __init__(self, csv_path, img_dir, labeled=True, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.labeled = labeled
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = torch.from_numpy(load_image(f"{self.img_dir}/{row['id']}")).unsqueeze(0)
        if self.transform is not None:
            image = self.transform(image)
        if self.labeled:
            return image, torch.tensor(int(row["label"]) - 1, dtype=torch.long)
        return image, row["id"]


def stratified_split(labels, val_split, seed):
    """Split indices so every class has the same train/validation ratio."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        n_val = max(1, round(len(idx) * val_split))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return rng.permutation(train_idx), rng.permutation(val_idx)


def class_weights(labels):
    """Inverse-frequency weights, scaled to sum to the number of classes."""
    counts = np.maximum(np.bincount(labels, minlength=NUM_CLASSES), 1)
    w = 1.0 / counts
    return torch.tensor(w / w.sum() * NUM_CLASSES, dtype=torch.float32), counts


def make_loaders(data_dir, batch_size, num_workers, val_split, seed):
    """Return train, validation and test loaders plus class weights and counts."""
    train_csv, test_csv = f"{data_dir}/train.csv", f"{data_dir}/test.csv"
    train_dir, test_dir = f"{data_dir}/train", f"{data_dir}/test"

    train_full = SignalDataset(train_csv, train_dir, transform=Augment())
    val_full = SignalDataset(train_csv, train_dir)
    test_set = SignalDataset(test_csv, test_dir, labeled=False)

    labels = val_full.df["label"].to_numpy().astype(int) - 1
    train_idx, val_idx = stratified_split(labels, val_split, seed)
    weights, counts = class_weights(labels[train_idx])

    def loader(dataset, shuffle):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return (
        loader(Subset(train_full, train_idx), shuffle=True),
        loader(Subset(val_full, val_idx), shuffle=False),
        loader(test_set, shuffle=False),
        weights,
        counts,
    )
