"""Reading and writing the landmark CSV files.

One row = one hand sample. Coordinates are stored exactly as MediaPipe
returns them (normalised x, y, z) together with the image size, so the
feature pipeline can change later without re-recording anything.
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Union

import numpy as np

from .landmarks import Hand, to_pixels

META_COLUMNS = ["label", "take", "source", "handedness", "width", "height"]
COORD_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
CSV_COLUMNS = META_COLUMNS + COORD_COLUMNS

PathLike = Union[str, Path]


@dataclass
class LandmarkDataset:
    points: np.ndarray   # (N, 21, 3) aspect-correct pixel coordinates
    labels: np.ndarray   # (N,) str
    groups: np.ndarray   # (N,) str - recording take; samples of one take stay together

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, mask: np.ndarray) -> "LandmarkDataset":
        return LandmarkDataset(self.points[mask], self.labels[mask], self.groups[mask])


def hand_to_row(label: str, take: str, source: str, hand: Hand) -> List[str]:
    width, height = hand.image_size
    coords = [f"{v:.6f}" for v in hand.landmarks.reshape(-1)]
    return [label, take, source, hand.handedness, str(width), str(height)] + coords


def append_rows(path: PathLike, rows: Iterable[Sequence[str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def _read(path: PathLike) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = set(CSV_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)[:5]} ...")
        return list(reader)


def label_counts(path: PathLike) -> Counter:
    path = Path(path)
    if not path.exists():
        return Counter()
    with open(path, newline="", encoding="utf-8") as fh:
        return Counter(row["label"] for row in csv.DictReader(fh))


def remove_take(path: PathLike, take: str) -> int:
    """Delete every row of one take (used for 'undo'). Returns rows removed."""
    path = Path(path)
    if not path.exists():
        return 0
    rows = _read(path)
    kept = [r for r in rows if r["take"] != take]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    return len(rows) - len(kept)


def load_dataset(paths: Union[PathLike, Sequence[PathLike]]) -> LandmarkDataset:
    """Load one or more CSV files into pixel-space landmark arrays."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    points, labels, groups = [], [], []
    for file_index, path in enumerate(paths):
        rows = _read(path)
        if not rows:
            continue
        raw = np.array([[float(r[c]) for c in COORD_COLUMNS] for r in rows], dtype=np.float32)
        size = np.array([[float(r["width"]), float(r["height"])] for r in rows], dtype=np.float32)
        raw = raw.reshape(-1, 21, 3)
        points.append(to_pixels(raw, size[:, 0, None], size[:, 1, None]))
        labels += [r["label"] for r in rows]
        # Prefix with the file index so take names never collide across files.
        groups += [f"{file_index}:{r['take']}" for r in rows]
    if not points:
        raise ValueError(f"No samples found in {list(map(str, paths))}")
    return LandmarkDataset(np.concatenate(points), np.array(labels), np.array(groups))
