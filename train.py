#!/usr/bin/env python3
"""Train the letter classifier on landmark CSV files.

Examples
  python train.py                                   # data/landmarks.csv -> models/asl_classifier.pt
  python train.py --data data/landmarks.csv data/kaggle.csv
  python train.py --data data/train.csv --val-data data/val.csv --test-data data/test.csv

Validation data is split off by recording take (group), not by frame, so the
reported accuracy reflects recordings the model hasn't seen. Neighbouring
video frames are near-duplicates, and a frame-level split would give an
inflated number.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from asl.config import ALPHABET, DEFAULT_DATA_CSV, DEFAULT_MODEL_PATH
from asl.data import LandmarkDataset, load_dataset
from asl.features import FEATURE_DIM, augment, features, normalize
from asl.model import LandmarkMLP, save_classifier


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, nargs="+", default=[DEFAULT_DATA_CSV],
                   help="training CSV file(s) (default: %(default)s)")
    p.add_argument("--val-data", type=Path, nargs="+",
                   help="separate validation CSV(s); otherwise --val-frac of the takes is held out")
    p.add_argument("--test-data", type=Path, nargs="+", help="optional test CSV(s), evaluated at the end")
    p.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH, help="model file to write")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40, help="early-stopping patience in epochs")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--no-augment", action="store_true", help="disable landmark augmentation")
    p.add_argument("--min-samples", type=int, default=10,
                   help="drop classes with fewer samples than this")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plots", action="store_true", help="skip the confusion-matrix / curve PNGs")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def order_labels(labels: Sequence[str]) -> List[str]:
    """Alphabetical letters first, then anything else (e.g. 'nothing')."""
    uniq = set(labels)
    letters = [c for c in ALPHABET if c in uniq]
    return letters + sorted(uniq - set(letters))


def group_split(labels: np.ndarray, groups: np.ndarray, val_frac: float,
                rng: np.random.Generator) -> Tuple[np.ndarray, List[str]]:
    """Boolean validation mask chosen per class, whole groups at a time.

    Classes recorded as a single group fall back to a random sample-level
    split (and are reported, since their validation score is optimistic).
    """
    val = np.zeros(len(labels), dtype=bool)
    frame_level = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        uniq = np.unique(groups[idx])
        if len(uniq) >= 2:
            rng.shuffle(uniq)
            target, taken, chosen = val_frac * len(idx), 0, []
            for g in uniq[:-1]:                     # never move every group to validation
                if taken >= target:
                    break
                chosen.append(g)
                taken += int(np.sum(groups[idx] == g))
            val[idx[np.isin(groups[idx], chosen)]] = True
        else:
            k = int(round(val_frac * len(idx)))
            if len(idx) >= 5 and k > 0:
                val[rng.choice(idx, k, replace=False)] = True
                frame_level.append(str(c))
    return val, frame_level


def to_arrays(ds: LandmarkDataset, labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    index = {name: i for i, name in enumerate(labels)}
    keep = np.array([l in index for l in ds.labels])
    norm = normalize(ds.points[keep])
    y = np.array([index[l] for l in ds.labels[keep]], dtype=np.int64)
    return norm, y


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def report(cm: np.ndarray, labels: List[str], title: str) -> Dict:
    total = cm.sum()
    acc = float(np.trace(cm) / max(total, 1))
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    support = cm.sum(1)
    present = support > 0
    print(f"\n{title}: accuracy {acc:.2%} on {total} samples "
          f"(macro F1 {f1[present].mean():.3f})")
    print(f"  {'label':>8} {'prec':>6} {'recall':>6} {'f1':>6} {'n':>6}   most confused with")
    for i, name in enumerate(labels):
        if not present[i]:
            continue
        row = cm[i].copy()
        row[i] = 0
        confused = ", ".join(f"{labels[j]}({row[j]})" for j in np.argsort(row)[::-1][:2] if row[j] > 0)
        print(f"  {name:>8} {precision[i]:6.2f} {recall[i]:6.2f} {f1[i]:6.2f} {support[i]:6d}   {confused}")
    return {"accuracy": acc, "macro_f1": float(f1[present].mean()), "samples": int(total)}


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(x[i:i + batch])).argmax(1).numpy() for i in range(0, len(x), batch)]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def save_plots(out_dir: Path, history: Dict[str, List[float]], cm: np.ndarray,
               labels: List[str], title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["val_acc"], color="tab:green")
    axes[1].set_title("validation accuracy"); axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close(fig)

    norm_cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    size = max(6, 0.38 * len(labels))
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if norm_cm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace) -> Dict:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    for path in list(args.data) + list(args.val_data or []) + list(args.test_data or []):
        if not Path(path).exists():
            sys.exit(f"{path} not found. Record samples with collect.py or build a CSV with "
                     "extract_images.py first.")
    data = load_dataset(args.data)
    counts = {l: int(n) for l, n in zip(*np.unique(data.labels, return_counts=True))}
    small = sorted(l for l, n in counts.items() if n < args.min_samples)
    if small:
        print(f"Dropping classes with < {args.min_samples} samples: {small}")
    labels = order_labels([l for l in counts if l not in small])
    if len(labels) < 2:
        sys.exit("Need at least two classes with data. Record more with collect.py.")
    print(f"Loaded {len(data)} samples, {len(labels)} classes from {', '.join(map(str, args.data))}")
    missing = [c for c in ALPHABET if c not in labels and c not in "JZ"]
    if missing:
        print(f"  note: no data for {' '.join(missing)} (the model won't know these letters)")

    if args.val_data:
        train_ds, val_ds = data, load_dataset(args.val_data)
        frame_level: List[str] = []
    else:
        val_mask, frame_level = group_split(data.labels, data.groups, args.val_frac, rng)
        train_ds, val_ds = data.subset(~val_mask), data.subset(val_mask)
        if frame_level:
            print(f"  note: {', '.join(frame_level)} have a single take, so their validation "
                  "samples come from the same take (score is optimistic). Record more takes.")
    x_train_norm, y_train = to_arrays(train_ds, labels)
    x_val_norm, y_val = to_arrays(val_ds, labels)
    if len(y_val) == 0:
        sys.exit("No validation samples. Record more takes or pass --val-data.")
    print(f"  train {len(y_train)}  |  validation {len(y_val)}")

    x_val = features(x_val_norm)
    train_feats = features(x_train_norm)
    model = LandmarkMLP(FEATURE_DIM, len(labels), args.hidden, args.dropout)
    model.set_feature_stats(train_feats.mean(0), train_feats.std(0))

    # Inverse-frequency class weights (clipped) so rare letters aren't ignored.
    freq = np.bincount(y_train, minlength=len(labels)).astype(np.float64)
    weights = np.clip(freq.sum() / (len(labels) * np.maximum(freq, 1)), 0.25, 4.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32),
                                    label_smoothing=args.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                           eta_min=args.lr * 0.02)
    xv, yv = torch.from_numpy(x_val), torch.from_numpy(y_val)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    best = {"acc": -1.0, "loss": float("inf"), "epoch": 0, "state": None}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(y_train))
        losses = []
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch = x_train_norm[idx]
            if not args.no_augment:
                batch = augment(batch, rng)
            xb = torch.from_numpy(features(batch))
            yb = torch.from_numpy(y_train[idx])
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(xv)
            val_loss = val_criterion(logits, yv).item()
            val_acc = (logits.argmax(1) == yv).float().mean().item()
        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_acc > best["acc"] + 1e-9 or (abs(val_acc - best["acc"]) <= 1e-9 and val_loss < best["loss"]):
            best = {"acc": val_acc, "loss": val_loss, "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:4d}  train loss {history['train_loss'][-1]:.4f}  "
                  f"val loss {val_loss:.4f}  val acc {val_acc:.2%}  (best {best['acc']:.2%} @ {best['epoch']})")
        if epoch - best["epoch"] >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best["state"])
    print(f"Training took {time.time() - t0:.0f}s; best validation accuracy "
          f"{best['acc']:.2%} at epoch {best['epoch']}")

    cm_val = confusion_matrix(y_val, predict(model, x_val), len(labels))
    val_metrics = report(cm_val, labels, "Validation")
    results = {"validation": val_metrics}

    cm_plot, plot_title = cm_val, "Validation confusion matrix"
    if args.test_data:
        test_ds = load_dataset(args.test_data)
        x_test_norm, y_test = to_arrays(test_ds, labels)
        unknown = sorted(set(test_ds.labels) - set(labels))
        if unknown:
            print(f"\nTest labels not in the model (skipped): {unknown}")
        cm_test = confusion_matrix(y_test, predict(model, features(x_test_norm)), len(labels))
        results["test"] = report(cm_test, labels, "Test")
        cm_plot, plot_title = cm_test, "Test confusion matrix"

    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": [str(p) for p in args.data],
        "train_samples": int(len(y_train)),
        "best_epoch": int(best["epoch"]),
        "results": results,
    }
    save_classifier(args.out, model, labels, args.hidden, args.dropout, meta)
    print(f"\nSaved model to {args.out}")
    if not args.no_plots:
        save_plots(args.out.parent, history, cm_plot, labels, plot_title)
        print(f"Saved {args.out.parent / 'training_curves.png'} and "
              f"{args.out.parent / 'confusion_matrix.png'}")
    (args.out.parent / "metrics.json").write_text(json.dumps(meta, indent=2))
    return results


if __name__ == "__main__":
    train(parse_args())
