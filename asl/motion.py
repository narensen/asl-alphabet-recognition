"""Recognise the motion letters J and Z from fingertip trajectories.

In ASL, J is the I handshape (pinky up) with the pinky drawing a J, and Z is
the index finger drawing a Z. One frame can't separate them from I and D, so
this module tracks the fingertip over time:

1. Each frame, check the handshape: only the pinky extended (J) or only the
   index extended (Z).
2. While the handshape holds, record the fingertip path in image space.
3. Cut off the still parts at the start and end, then check duration, path
   length and size (in palm widths, so distance from the camera doesn't
   matter).
4. Resample the path to a fixed number of points, normalise its position and
   scale, and compare it with template shapes (a template-matching idea
   borrowed from the "$1 unistroke recogniser"). Mirrored templates cover
   left-handed signers and un-mirrored video.

Coordinates are image pixels with y pointing down, as seen in the mirrored
(selfie) preview, where a right-handed signer's J and Z look like the written
letters.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np

from .features import normalize, palm_size

# finger name -> (PIP joint, fingertip) landmark indices
FINGER_JOINTS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}

# letter -> (the finger that must be the only one extended, its tip landmark)
MOTION_SHAPES = {"J": ("pinky", 20), "Z": ("index", 8)}


def extended_fingers(points: np.ndarray, ratio: float = 1.1) -> Dict[str, bool]:
    """A finger counts as extended when its tip is clearly farther from the
    wrist than its middle (PIP) joint. Works in 3D, so fingers pointing at
    the camera still count."""
    n = normalize(points)  # wrist at the origin
    return {name: bool(np.linalg.norm(n[tip]) > ratio * np.linalg.norm(n[pip]))
            for name, (pip, tip) in FINGER_JOINTS.items()}


def only_extended(ext: Dict[str, bool], finger: str) -> bool:
    return ext[finger] and not any(v for k, v in ext.items() if k != finger)


# --------------------------------------------------------------------------- #
# Path utilities and templates
# --------------------------------------------------------------------------- #
def resample(path: np.ndarray, n: int) -> np.ndarray:
    """Resample a 2D polyline to n points spaced evenly along its length."""
    path = np.asarray(path, dtype=np.float64)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    path, seg = path[keep], seg[seg > 1e-9]
    if len(path) < 2:
        return np.repeat(path[:1], n, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, cum[-1], n)
    return np.stack([np.interp(targets, cum, path[:, 0]),
                     np.interp(targets, cum, path[:, 1])], axis=1)


def normalize_path(path: np.ndarray) -> np.ndarray:
    """Centre on the centroid and scale the longer bounding-box side to 1
    (uniform scaling keeps the aspect ratio: J is tall, Z is wide)."""
    p = path - path.mean(axis=0)
    extent = float((p.max(axis=0) - p.min(axis=0)).max())
    return p / max(extent, 1e-9)


def _rotate(path: np.ndarray, degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return path @ rot.T


def z_template() -> np.ndarray:
    # top-left -> top-right -> bottom-left -> bottom-right (y points down)
    return np.array([(0, 0), (1, 0), (0, 1), (1, 1)], dtype=np.float64)


def j_template(stem: float, radius: float, sweep: float) -> np.ndarray:
    """A downward stem followed by a hook curling to the left."""
    x = 2 * radius
    angles = np.deg2rad(np.linspace(0.0, sweep, 25))
    hook = np.stack([radius + radius * np.cos(angles), stem + radius * np.sin(angles)], axis=1)
    return np.vstack([[(x, 0.0)], hook])


def _prepare(shapes: Sequence[np.ndarray], n: int) -> List[np.ndarray]:
    out = []
    for shape in shapes:
        for s in (shape, shape * np.array([-1.0, 1.0])):  # + mirror image
            out.append(normalize_path(resample(s, n)))
    return out


def default_templates(n: int = 32) -> Dict[str, List[np.ndarray]]:
    j_shapes = [j_template(0.6, 0.30, 180), j_template(0.35, 0.35, 180),
                j_template(0.8, 0.25, 150), j_template(0.2, 0.40, 150)]
    return {"J": _prepare(j_shapes, n), "Z": _prepare([z_template()], n)}


def match_distance(path: np.ndarray, templates: Sequence[np.ndarray], n: int = 32,
                   angles: Sequence[float] = (-15.0, -7.5, 0.0, 7.5, 15.0)) -> float:
    """Smallest mean point-to-point distance between the path and any template,
    allowing a small tilt (0 = perfect match; distances are in units of the
    path's bounding-box size)."""
    p = resample(path, n)
    best = math.inf
    for angle in angles:
        q = normalize_path(_rotate(p, angle))
        for template in templates:
            best = min(best, float(np.linalg.norm(q - template, axis=1).mean()))
    return best


def turn_angle(path: np.ndarray, n: int = 32) -> float:
    """Angle in degrees between the path's opening and closing directions.
    A J turns sharply at the hook; a straight swipe hardly turns."""
    p = resample(path, n)
    start = p[int(0.35 * (n - 1))] - p[0]
    end = p[-1] - p[int(0.75 * (n - 1))]
    denom = np.linalg.norm(start) * np.linalg.norm(end)
    if denom < 1e-9:
        return 0.0
    return math.degrees(math.acos(float(np.clip(start @ end / denom, -1.0, 1.0))))


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
@dataclass
class MotionEvent:
    letter: str
    t_start: float    # when the drawing movement started
    t_end: float
    distance: float   # template match distance (lower = cleaner)
    run_start: float  # when the J/Z handshape was first held


@dataclass
class MotionConfig:
    history_s: float = 2.5      # how much trajectory to keep
    min_duration_s: float = 0.25
    max_duration_s: float = 2.5
    min_path: float = 1.5       # fingertip path length, in palm sizes
    min_extent: float = 0.9     # bounding-box diagonal of the path, in palm sizes
    move_speed: float = 1.0     # palm sizes per second that count as "moving"
    max_pause_s: float = 0.3    # pauses longer than this split movements apart
    max_gap_frames: int = 3     # wrong-handshape frames tolerated inside a gesture
    # Template-match thresholds (lower = stricter). Calibrated in
    # tests/test_motion.py: true J/Z paths score <= ~0.12 / ~0.19 even with
    # noise and tilt, while other shapes score >= ~0.29 against Z.
    max_distance_j: float = 0.13
    max_distance_z: float = 0.22
    min_turn_j: float = 45.0    # degrees; rejects a plain downward swipe
    cooldown_s: float = 0.8
    lost_hand_s: float = 0.3    # drop the trajectory if the hand is gone this long
    n_points: int = 32


@dataclass
class _Sample:
    t: float
    tips: Dict[str, np.ndarray]
    palm: float
    gates: Dict[str, bool] = field(default_factory=dict)


class MotionLetterDetector:
    """Feed it one hand per frame; it returns a MotionEvent when J or Z is drawn."""

    def __init__(self, config: Optional[MotionConfig] = None,
                 letters: Sequence[str] = ("J", "Z")):
        self.cfg = config or MotionConfig()
        self.letters = tuple(letters)
        self.templates = default_templates(self.cfg.n_points)
        self._hist: Deque[_Sample] = deque()
        self._cooldown_until = -math.inf

    def reset(self) -> None:
        self._hist.clear()

    def update(self, t: float, points: Optional[np.ndarray]) -> Optional[MotionEvent]:
        """points: (21, 3) pixel landmarks, or None when no hand is visible."""
        if points is None:
            if self._hist and t - self._hist[-1].t > self.cfg.lost_hand_s:
                self._hist.clear()
            return None

        ext = extended_fingers(points)
        sample = _Sample(t=t, palm=float(palm_size(points)), tips={}, gates={})
        for letter in self.letters:
            finger, tip = MOTION_SHAPES[letter]
            sample.tips[letter] = np.asarray(points[tip, :2], dtype=np.float64)
            sample.gates[letter] = only_extended(ext, finger)
        self._hist.append(sample)
        while self._hist and t - self._hist[0].t > self.cfg.history_s:
            self._hist.popleft()

        if t < self._cooldown_until:
            return None
        for letter in self.letters:
            event = self._check(letter)
            if event is not None:
                self._cooldown_until = t + self.cfg.cooldown_s
                self._hist.clear()
                return event
        return None

    def trail(self, letter: str) -> np.ndarray:
        """Fingertip path of the current handshape run (for drawing)."""
        run = self._current_run(letter)
        if not run:
            return np.zeros((0, 2))
        return np.array([s.tips[letter] for s in run])

    # ------------------------------------------------------------------ #
    def _current_run(self, letter: str) -> List[_Sample]:
        hist = list(self._hist)
        gates = [s.gates.get(letter, False) for s in hist]
        if not any(gates):
            return []
        last = len(gates) - 1 - gates[::-1].index(True)
        if len(gates) - 1 - last > self.cfg.max_gap_frames:
            return []  # the handshape ended a while ago
        start, gap = last, 0
        for i in range(last - 1, -1, -1):
            if gates[i]:
                start, gap = i, 0
            else:
                gap += 1
                if gap > self.cfg.max_gap_frames:
                    break
        return hist[start:last + 1]

    def _check(self, letter: str) -> Optional[MotionEvent]:
        cfg = self.cfg
        run = self._current_run(letter)
        if len(run) < 5:
            return None
        times = np.array([s.t for s in run])
        tips = np.array([s.tips[letter] for s in run])
        palm = float(np.median([s.palm for s in run]))
        if palm <= 1e-6:
            return None
        # Light smoothing so landmark jitter doesn't count as motion.
        if len(tips) >= 3:
            kernel = np.ones(3) / 3
            smooth = np.stack([np.convolve(tips[:, k], kernel, mode="same") for k in range(2)], 1)
            smooth[0], smooth[-1] = tips[0], tips[-1]
        else:
            smooth = tips
        step = np.linalg.norm(np.diff(smooth, axis=0), axis=1) / palm
        speed = step / np.maximum(np.diff(times), 1e-3)
        moving_steps = np.flatnonzero(speed > cfg.move_speed)  # step i spans frames i..i+1
        if len(moving_steps) == 0:
            return None
        # The gesture is the latest stretch of movement. Short pauses (slowing
        # down at a corner of the Z) don't split it; longer ones do, so moving
        # the hand into position before drawing isn't part of the path.
        last = first = int(moving_steps[-1])
        for j in moving_steps[-2::-1]:
            if times[first] - times[j + 1] > cfg.max_pause_s:
                break
            first = int(j)
        segment = smooth[first:last + 2]
        t0, t1 = float(times[first]), float(times[last + 1])
        if not cfg.min_duration_s <= t1 - t0 <= cfg.max_duration_s:
            return None
        if step[first:last + 1].sum() < cfg.min_path:
            return None
        if np.linalg.norm(segment.max(0) - segment.min(0)) / palm < cfg.min_extent:
            return None
        distance = match_distance(segment, self.templates[letter], cfg.n_points)
        if letter == "J":
            if distance > cfg.max_distance_j or turn_angle(segment, cfg.n_points) < cfg.min_turn_j:
                return None
        elif distance > cfg.max_distance_z:
            return None
        return MotionEvent(letter, t0, t1, distance, run_start=float(times[0]))
