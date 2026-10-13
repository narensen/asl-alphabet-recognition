#!/usr/bin/env python3
"""Build a landmark CSV from a folder of labelled hand images.

Expected layout: one sub-folder per letter, for example
    dataset/A/*.jpg, dataset/B/*.jpg, ...
This matches the Kaggle "ASL Alphabet" dataset (asl_alphabet_train/) and most
other image datasets. Folder names are matched without regard to case; J and Z
are skipped (they need motion), a folder named nothing/none/blank becomes the
"nothing" class, and other folders (space, del, ...) are skipped.

Examples
  python extract_images.py path/to/asl_alphabet_train --out data/kaggle.csv --limit 600 --group-size 100
  python extract_images.py path/to/my_photos --out data/photos.csv --workers 4

Datasets that are cut from videos contain long runs of near-identical frames.
Use --group-size so consecutive frames share one group; train.py then keeps
each group entirely in training or entirely in validation.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from asl.config import HAND_MODEL_PATH, MOTION_LETTERS, NOTHING, STATIC_LETTERS
from asl.data import append_rows, hand_to_row
from asl.landmarks import Hand, HandDetector, ensure_hand_model, read_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NOTHING_NAMES = {"nothing", "none", "blank", "no_sign", "background"}
# Augmentation suffixes some dataset tools append to copies of the same photo.
_AUG_SUFFIX = re.compile(r"(\.rf\.[0-9a-f]+|_aug_?\d+|_augmented\d*)$", re.IGNORECASE)

_detector: Optional[HandDetector] = None
_pad = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="folder containing one sub-folder per letter")
    p.add_argument("--out", type=Path, required=True, help="CSV file to write (appends if it exists)")
    p.add_argument("--limit", type=int, default=0, help="max images per letter (0 = all)")
    p.add_argument("--group-size", type=int, default=1,
                   help="consecutive images (in natural filename order) per group (default: 1)")
    p.add_argument("--pad", type=float, default=0.25,
                   help="add a black border of this fraction of the image size first; "
                        "helps MediaPipe find hands that touch the image edge (default: 0.25)")
    p.add_argument("--min-confidence", type=float, default=0.5, help="hand detection threshold")
    p.add_argument("--include-jz", action="store_true",
                   help="also keep J and Z folders as static poses (not recommended)")
    p.add_argument("--workers", type=int, default=1, help="parallel processes (default: 1)")
    p.add_argument("--seed", type=int, default=0, help="random seed used with --limit")
    return p.parse_args()


def natural_key(path: Path):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", path.name)]


def folder_label(name: str, include_jz: bool) -> Optional[str]:
    n = name.strip()
    if n.lower() in NOTHING_NAMES:
        return NOTHING
    if len(n) == 1 and n.upper() in STATIC_LETTERS:
        return n.upper()
    if include_jz and len(n) == 1 and n.upper() in MOTION_LETTERS:
        return n.upper()
    return None


def find_images(root: Path, include_jz: bool, limit: int, group_size: int,
                seed: int) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """Return [(image path, label, group)] and the list of skipped folder names."""
    rng = np.random.default_rng(seed)
    items, skipped = [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        label = folder_label(folder.name, include_jz)
        if label is None:
            skipped.append(folder.name)
            continue
        files = sorted((f for f in folder.rglob("*") if f.suffix.lower() in IMAGE_EXTS),
                       key=natural_key)
        # Group consecutive files (video frames) and copies of the same photo.
        groups = {}
        for i, f in enumerate(files):
            source = _AUG_SUFFIX.sub("", f.stem)
            groups[f] = f"{label}/{source}" if group_size <= 1 else f"{label}/g{i // group_size}"
        if limit and len(files) > limit:
            keep = sorted(rng.choice(len(files), limit, replace=False))
            files = [files[i] for i in keep]
        items += [(str(f), label, groups[f]) for f in files]
    return items, skipped


def _init_worker(min_confidence: float, pad: float) -> None:
    global _detector, _pad
    _detector = HandDetector(mode="image", num_hands=1,
                             min_detection_confidence=min_confidence)
    _pad = pad


def _process(item: Tuple[str, str, str]):
    path, label, group = item
    image = read_image(path)
    if image is None:
        return item, None, "unreadable"
    if _pad > 0:
        h, w = image.shape[:2]
        b = int(round(_pad * max(h, w)))
        image = cv2.copyMakeBorder(image, b, b, b, b, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    hands = _detector.detect(image)
    if not hands:
        return item, None, "no hand"
    hand = hands[0]
    return item, (hand.landmarks, hand.handedness, hand.score, hand.image_size), "ok"


def main() -> None:
    args = parse_args()
    if not args.root.is_dir():
        sys.exit(f"Not a folder: {args.root}")
    items, skipped = find_images(args.root, args.include_jz, args.limit, args.group_size, args.seed)
    if skipped:
        print(f"Skipping folders: {', '.join(skipped)}")
    if not items:
        sys.exit("No labelled images found. Expected sub-folders named A, B, C, ...")
    print(f"Processing {len(items)} images from {args.root} with {args.workers} worker(s) ...")
    ensure_hand_model(HAND_MODEL_PATH)  # download once, before workers start

    total, found = Counter(), Counter()
    reasons = Counter()
    rows = []
    t0 = time.time()
    if args.workers > 1:
        pool = mp.Pool(args.workers, initializer=_init_worker,
                       initargs=(args.min_confidence, args.pad))
        results = pool.imap_unordered(_process, items, chunksize=16)
    else:
        pool = None
        _init_worker(args.min_confidence, args.pad)
        results = map(_process, items)
    try:
        for n, (item, hand_data, status) in enumerate(results, 1):
            path, label, group = item
            total[label] += 1
            if hand_data is not None:
                landmarks, handedness, score, size = hand_data
                hand = Hand(np.asarray(landmarks), handedness, score, tuple(size))
                rows.append(hand_to_row(label, group, Path(path).name, hand))
                found[label] += 1
            else:
                reasons[status] += 1
            if n % 200 == 0 or n == len(items):
                rate = n / max(time.time() - t0, 1e-6)
                print(f"  {n}/{len(items)}  hands found: {sum(found.values())}  ({rate:.1f} img/s)",
                      flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    rows.sort(key=lambda r: (r[0], r[1]))
    append_rows(args.out, rows)
    print(f"\nWrote {len(rows)} samples to {args.out}")
    print("Detection rate per letter:")
    print("  " + "  ".join(f"{k}:{found[k]}/{total[k]}" for k in sorted(total)))
    if reasons:
        print("Skipped images: " + ", ".join(f"{v} {k}" for k, v in reasons.items()))
    low = [k for k in total if total[k] and found[k] / total[k] < 0.5]
    if low:
        print(f"Low detection rate for {' '.join(sorted(low))}: try --pad 0.4 or "
              "--min-confidence 0.3")


if __name__ == "__main__":
    main()
