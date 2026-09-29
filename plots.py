"""Plots and summary metrics from the files written by train.py."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from data import NUM_CLASSES, load_image

CLASSES = list(range(1, NUM_CLASSES + 1))
PROB_COLS = [f"prob_{c}" for c in CLASSES]


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    np.add.at(cm, (y_true - 1, y_pred - 1), 1)
    return cm


def divide(a, b):
    return np.divide(a, b, out=np.zeros(a.shape, dtype=float), where=b > 0)


def class_metrics(cm):
    tp = np.diag(cm)
    precision, recall = divide(tp, cm.sum(axis=0)), divide(tp, cm.sum(axis=1))
    f1 = divide(2 * precision * recall, precision + recall)
    return pd.DataFrame(
        {"class": CLASSES, "precision": precision, "recall": recall, "f1": f1, "support": cm.sum(axis=1)}
    )


def plot_curves(m, path):
    best = int(m.loc[m["val_acc"].idxmax(), "epoch"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, name in zip(axes[:2], ("loss", "acc")):
        ax.plot(m["epoch"], m[f"train_{name}"], label="train")
        ax.plot(m["epoch"], m[f"val_{name}"], label="validation")
        ax.axvline(best, color="gray", linestyle="--", label=f"best epoch ({best})")
        ax.set(title="Loss" if name == "loss" else "Accuracy", xlabel="Epoch")
        ax.legend()
    axes[2].plot(m["epoch"], m["lr"], color="C2")
    axes[2].set(title="Learning rate", xlabel="Epoch", yscale="log")
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    save(fig, path)


def plot_confusion(cm, path):
    normalized = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data, title, fmt in (
        (axes[0], cm, "Counts", "d"),
        (axes[1], normalized, "Row-normalized (recall)", ".2f"),
    ):
        im = ax.imshow(data, cmap="Blues")
        ax.set(title=title, xlabel="Predicted", ylabel="True", xticks=range(NUM_CLASSES),
               yticks=range(NUM_CLASSES), xticklabels=CLASSES, yticklabels=CLASSES)
        for i, j in np.ndindex(data.shape):
            color = "white" if data[i, j] > data.max() / 2 else "black"
            ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", color=color)
        fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, path)


def plot_class_metrics(stats, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    x, width = np.arange(NUM_CLASSES), 0.27
    for k, name in enumerate(("precision", "recall", "f1")):
        ax.bar(x + (k - 1) * width, stats[name], width, label=name)
    ax.set(title="Per-class metrics", xlabel="Class (number of objects)", ylim=(0, 1),
           xticks=x, xticklabels=CLASSES)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, path)


def plot_errors(v, path):
    error = v["pred"] - v["label"]
    span = NUM_CLASSES - 1
    counts = error.value_counts().reindex(range(-span, span + 1), fill_value=0)
    confidence = v[PROB_COLS].max(axis=1)
    correct = error == 0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(counts.index, counts.values)
    axes[0].set(title="Count error (predicted minus true)", xlabel="Error", ylabel="Images",
                xticks=counts.index)
    axes[1].hist([confidence[correct], confidence[~correct]], bins=np.linspace(0.2, 1.0, 17),
                 stacked=True, color=["C0", "C3"], label=["correct", "wrong"])
    axes[1].set(title="Prediction confidence", xlabel="Highest class probability", ylabel="Images")
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    save(fig, path)


def plot_misclassified(v, data_dir, path, n=10):
    """The most confident mistakes."""
    wrong = v[v["pred"] != v["label"]].copy()
    wrong["confidence"] = wrong[PROB_COLS].max(axis=1)
    wrong = wrong.nlargest(n, "confidence")
    if wrong.empty:
        return
    fig, axes = plt.subplots(1, len(wrong), figsize=(1.6 * len(wrong), 3.6), squeeze=False)
    for ax, (_, r) in zip(axes[0], wrong.iterrows()):
        ax.imshow(load_image(f"{data_dir}/train/{r['id']}"), cmap="viridis")
        ax.set_title(f"true {r['label']}\npred {r['pred']}\n{r['confidence']:.0%}", fontsize=9)
        ax.axis("off")
    save(fig, path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output-dir", default="./output")
    return p.parse_args()


def main():
    args = parse_args()
    plot_dir = f"{args.output_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)

    metrics = pd.read_csv(f"{args.output_dir}/metrics.csv")
    val = pd.read_csv(f"{args.output_dir}/val_predictions.csv")
    cm = confusion_matrix(val["label"].to_numpy(), val["pred"].to_numpy())
    stats = class_metrics(cm)

    plot_curves(metrics, f"{plot_dir}/curves.png")
    plot_confusion(cm, f"{plot_dir}/confusion_matrix.png")
    plot_class_metrics(stats, f"{plot_dir}/class_metrics.png")
    plot_errors(val, f"{plot_dir}/errors.png")
    plot_misclassified(val, args.data_dir, f"{plot_dir}/misclassified.png")
    stats.to_csv(f"{plot_dir}/class_metrics.csv", index=False)

    error = (val["pred"] - val["label"]).abs()
    best = metrics.loc[metrics["val_acc"].idxmax()]
    print(f"Best epoch {int(best['epoch'])}, validation accuracy {best['val_acc']:.4f}")
    print(f"Macro F1 {stats['f1'].mean():.4f}")
    print(f"Within one object {(error <= 1).mean():.4f}, mean absolute count error {error.mean():.4f}")
    print(stats.round(4).to_string(index=False))
    print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
