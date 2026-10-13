"""Landmark normalisation, feature extraction and training-time augmentation.

Every function works on arrays shaped (..., 21, 3), so one hand (21, 3) and a
batch (N, 21, 3) take the same code path.

Design notes
------------
* Translation and scale are removed: the wrist goes to the origin and
  coordinates are divided by the palm size (mean 3D length of rigid palm
  segments, which barely changes when the fingers move).
* Rotation is deliberately NOT removed. Several ASL letters differ mainly by
  orientation (H vs U, K vs P, G vs Q), so the in-image orientation carries
  information.
* Left and right hands are mirror images. Training randomly mirrors the x axis
  so one model handles both hands without trusting MediaPipe's
  handedness label.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

FEATURE_VERSION = 1

WRIST = 0
PALM_POINTS = (0, 5, 9, 13, 17)
PALM_SEGMENTS: Tuple[Tuple[int, int], ...] = ((0, 5), (0, 9), (0, 13), (0, 17), (5, 17))

# Distance features that make the classifier's job easier with little data.
_THUMB_PAIRS = tuple((4, j) for j in range(5, 21))        # thumb placement (A/S/T/M/N/E)
_SPREAD_PAIRS = ((8, 12), (12, 16), (16, 20))             # finger spread (U/V, R)
_REACH_PAIRS = ((0, 4), (0, 8), (0, 12), (0, 16), (0, 20))  # tip-to-wrist reach
_CURL_PAIRS = ((5, 8), (9, 12), (13, 16), (17, 20))       # MCP-to-tip: finger curl
DISTANCE_PAIRS = _THUMB_PAIRS + _SPREAD_PAIRS + _REACH_PAIRS + _CURL_PAIRS
_PAIR_I = np.array([a for a, _ in DISTANCE_PAIRS])
_PAIR_J = np.array([b for _, b in DISTANCE_PAIRS])
_SEG_I = np.array([a for a, _ in PALM_SEGMENTS])
_SEG_J = np.array([b for _, b in PALM_SEGMENTS])

FEATURE_DIM = 21 * 3 + len(DISTANCE_PAIRS)  # 63 + 28 = 91


def palm_size(points: np.ndarray) -> np.ndarray:
    """Mean 3D length of the palm segments, shape (...)."""
    p = np.asarray(points, dtype=np.float32)
    return np.linalg.norm(p[..., _SEG_I, :] - p[..., _SEG_J, :], axis=-1).mean(-1)


def palm_center(points: np.ndarray) -> np.ndarray:
    """Mean of wrist and knuckle positions, shape (..., 3)."""
    return np.asarray(points, dtype=np.float32)[..., PALM_POINTS, :].mean(-2)


def normalize(points: np.ndarray) -> np.ndarray:
    """Pixel landmarks (..., 21, 3) -> wrist at the origin, palm size 1."""
    p = np.asarray(points, dtype=np.float32)
    p = p - p[..., :1, :]
    size = np.maximum(palm_size(p), 1e-6)
    return p / size[..., None, None]


def features(normalized: np.ndarray) -> np.ndarray:
    """Normalised landmarks (..., 21, 3) -> feature vectors (..., FEATURE_DIM)."""
    n = np.asarray(normalized, dtype=np.float32)
    coords = n.reshape(*n.shape[:-2], 63)
    dists = np.linalg.norm(n[..., _PAIR_I, :] - n[..., _PAIR_J, :], axis=-1)
    return np.concatenate([coords, dists], axis=-1).astype(np.float32)


def mirror(normalized: np.ndarray) -> np.ndarray:
    """Mirror a hand left <-> right (flip the x axis)."""
    out = np.array(normalized, dtype=np.float32, copy=True)
    out[..., 0] *= -1
    return out


def _rotation_matrices(n: int, rng: np.random.Generator,
                       roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(n, 3, 3) random rotations; angles drawn uniformly within ±limits (degrees)."""
    def angles(limit: float) -> np.ndarray:
        return np.deg2rad(rng.uniform(-limit, limit, n))

    a, b, c = angles(roll), angles(pitch), angles(yaw)
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    zeros, ones = np.zeros(n), np.ones(n)
    rz = np.stack([ca, -sa, zeros, sa, ca, zeros, zeros, zeros, ones], -1).reshape(n, 3, 3)
    rx = np.stack([ones, zeros, zeros, zeros, cb, -sb, zeros, sb, cb], -1).reshape(n, 3, 3)
    ry = np.stack([cc, zeros, sc, zeros, ones, zeros, -sc, zeros, cc], -1).reshape(n, 3, 3)
    return (rz @ ry @ rx).astype(np.float32)


def augment(normalized: np.ndarray, rng: Optional[np.random.Generator] = None, *,
            roll: float = 15.0, pitch: float = 12.0, yaw: float = 15.0,
            stretch: float = 0.08, jitter: float = 0.02,
            mirror_prob: float = 0.5) -> np.ndarray:
    """Randomly perturb a batch (N, 21, 3) of normalised hands.

    Small 3D rotations (roll = in-image), anisotropic x/y stretch (camera
    perspective), Gaussian jitter (landmark noise) and random mirroring
    (left/right hand). The result is re-normalised so it matches what
    `normalize` produces at inference time.
    """
    rng = rng or np.random.default_rng()
    p = np.asarray(normalized, dtype=np.float32)
    n = p.shape[0]
    p = p @ _rotation_matrices(n, rng, roll, pitch, yaw).transpose(0, 2, 1)
    scale = rng.uniform(1 - stretch, 1 + stretch, (n, 1, 2)).astype(np.float32)
    p[..., :2] *= scale
    p = p + rng.normal(0.0, jitter, p.shape).astype(np.float32)
    flip = rng.random(n) < mirror_prob
    p[flip, :, 0] *= -1
    return normalize(p)
