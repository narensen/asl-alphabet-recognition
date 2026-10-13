"""Shared test helpers: a simple synthetic hand so tests don't need a camera."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asl.landmarks import Hand  # noqa: E402

# (knuckle x offset, knuckle height, segment lengths) in palm units; fingers point up (-y).
_FINGERS = {
    "index": (-0.35, 1.00, (0.45, 0.28, 0.22)),
    "middle": (-0.10, 1.05, (0.50, 0.32, 0.24)),
    "ring": (0.15, 1.00, (0.46, 0.30, 0.22)),
    "pinky": (0.38, 0.90, (0.36, 0.22, 0.20)),
}
_FINGER_START = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def canonical_hand(extended: Iterable[str] = ("index", "middle", "ring", "pinky"),
                   thumb_out: bool = True) -> np.ndarray:
    """(21, 3) hand in palm units with the wrist at the origin, fingers up."""
    ext = set(extended)
    pts = np.zeros((21, 3))
    thumb = [(-0.30, -0.30, 0.0), (-0.52, -0.52, 0.0), (-0.66, -0.74, 0.0), (-0.75, -0.95, 0.0)]
    if not thumb_out:
        thumb = [(-0.30, -0.30, 0.0), (-0.40, -0.55, -0.1), (-0.25, -0.75, -0.2), (-0.05, -0.80, -0.25)]
    pts[1:5] = thumb
    for name, (x, h, (a, b, c)) in _FINGERS.items():
        i = _FINGER_START[name]
        mcp = np.array([x, -h, 0.0])
        pts[i] = mcp
        if name in ext:
            pts[i + 1] = mcp + (0, -a, 0)
            pts[i + 2] = mcp + (0, -a - b, 0)
            pts[i + 3] = mcp + (0, -a - b - c, 0)
        else:  # curled forward and down into the palm
            pts[i + 1] = mcp + (0, -0.25, -0.35)
            pts[i + 2] = mcp + (0, 0.02, -0.40)
            pts[i + 3] = mcp + (0, 0.12, -0.22)
    return pts


def place(canonical: np.ndarray, center: Sequence[float] = (640, 360), size: float = 100.0,
          angle_deg: float = 0.0, mirror: bool = False) -> np.ndarray:
    """Scale to pixels, rotate in the image plane and move the wrist to `center`."""
    p = canonical.copy()
    if mirror:
        p[:, 0] *= -1
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    p = (p @ rot.T) * size
    p[:, :2] += np.asarray(center, dtype=float)
    return p.astype(np.float32)


def hand_from_points(points: np.ndarray, image_size: Tuple[int, int] = (1280, 720)) -> Hand:
    """Wrap pixel points in a Hand (converting back to normalised coordinates)."""
    w, h = image_size
    norm = np.array(points, dtype=np.float32, copy=True)
    norm[:, 0] /= w
    norm[:, 1] /= h
    norm[:, 2] /= w
    return Hand(norm, "Right", 0.99, image_size)


def moving_hand_frames(canonical: np.ndarray, tip_index: int, path: np.ndarray,
                       fps: float = 30.0, t0: float = 0.0, size: float = 100.0,
                       noise: float = 0.0, rng: Optional[np.random.Generator] = None,
                       mirror: bool = False):
    """Yield (t, points) with the hand translated so `tip_index` follows `path` (pixels)."""
    rng = rng or np.random.default_rng(0)
    base = place(canonical, (0, 0), size, mirror=mirror)
    offset = base[tip_index, :2]
    for k, target in enumerate(path):
        pts = base.copy()
        pts[:, :2] += np.asarray(target) - offset
        if noise:
            pts[:, :2] += rng.normal(0, noise, (21, 2))
        yield t0 + k / fps, pts


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
