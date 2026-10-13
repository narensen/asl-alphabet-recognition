"""Hand detection with MediaPipe's HandLandmarker, plus drawing helpers."""
from __future__ import annotations

import shutil
import ssl
import threading
import time
import urllib.error
import urllib.request
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .config import HAND_MODEL_PATH, HAND_MODEL_URL

# 21-point hand topology used by MediaPipe.
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),             # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),             # index
    (5, 9), (9, 10), (10, 11), (11, 12),        # middle
    (9, 13), (13, 14), (14, 15), (15, 16),      # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm edge
)
FINGERTIPS = (4, 8, 12, 16, 20)


@dataclass
class Hand:
    """One detected hand.

    ``landmarks`` are MediaPipe's normalised image coordinates: x and y in
    [0, 1] relative to the image width and height, z is depth relative to the
    wrist on roughly the same scale as x.
    """

    landmarks: np.ndarray            # (21, 3) float32
    handedness: str                  # "Left" / "Right" as reported by MediaPipe
    score: float                     # handedness confidence
    image_size: Tuple[int, int]      # (width, height)

    @property
    def points(self) -> np.ndarray:
        """(21, 3) landmarks in pixels (x right, y down, z on the x scale)."""
        return to_pixels(self.landmarks, *self.image_size)

    def bbox(self, pad: float = 0.0) -> Tuple[int, int, int, int]:
        """Integer bounding box (x0, y0, x1, y1), optionally padded by a
        fraction of the box size and clipped to the image."""
        pts = self.points[:, :2]
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        w, h = self.image_size
        return (int(max(0, x0 - px)), int(max(0, y0 - py)),
                int(min(w - 1, x1 + px)), int(min(h - 1, y1 + py)))


def to_pixels(landmarks: np.ndarray, width: float, height: float) -> np.ndarray:
    """Convert normalised MediaPipe coordinates to aspect-correct pixels."""
    pts = np.array(landmarks, dtype=np.float32, copy=True)
    pts[..., 0] *= width
    pts[..., 1] *= height
    pts[..., 2] *= width
    return pts


# --------------------------------------------------------------------------- #
# Model download
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path, context: Optional[ssl.SSLContext]) -> None:
    with urllib.request.urlopen(url, timeout=60, context=context) as response, \
            open(dest, "wb") as fh:
        shutil.copyfileobj(response, fh)


def ensure_hand_model(path: Union[str, Path] = HAND_MODEL_PATH,
                      url: str = HAND_MODEL_URL) -> Path:
    """Return the path to ``hand_landmarker.task``, downloading it if needed."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    print(f"Downloading the MediaPipe hand model (~8 MB) to {path} ...", flush=True)
    try:
        try:
            _download(url, tmp, ssl.create_default_context())
        except (ssl.SSLError, urllib.error.URLError) as exc:
            # Some Python installs (e.g. python.org builds on macOS) ship
            # without root certificates; certifi comes with mediapipe.
            if "CERTIFICATE" not in str(exc).upper():
                raise
            import certifi
            _download(url, tmp, ssl.create_default_context(cafile=certifi.where()))
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001 - show a helpful message instead
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Could not download the hand model ({exc}).\n"
            f"Download it manually from\n  {url}\nand save it as\n  {path}"
        ) from exc
    return path


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class HandDetector:
    """Thin wrapper around MediaPipe's HandLandmarker (Tasks API).

    mode="video" tracks the hand between frames (smoother and faster, needs
    increasing timestamps); mode="image" treats every image independently.
    """

    def __init__(self, mode: str = "video", num_hands: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_path: Union[str, Path] = HAND_MODEL_PATH):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - environment specific
            raise ImportError(
                "MediaPipe is not installed. Run: pip install -r requirements.txt"
            ) from exc

        if mode not in ("video", "image"):
            raise ValueError("mode must be 'video' or 'image'")
        self._mp = mp
        self._video = mode == "video"
        self._last_ts = -1

        model_path = ensure_hand_model(model_path)
        options = vision.HandLandmarkerOptions(
            # Passing the bytes (not the path) avoids MediaPipe's problems with
            # non-ASCII characters in Windows paths.
            base_options=mp_tasks.BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=(vision.RunningMode.VIDEO if self._video
                          else vision.RunningMode.IMAGE),
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        # Close the landmarker exactly once: when this object is garbage
        # collected, when close() is called, or at interpreter exit. Recent
        # MediaPipe versions run the landmarker on a thread pool that Python
        # shuts down *before* ordinary atexit handlers, so the exit hook goes
        # into threading's own (earlier) atexit list when that exists.
        self._finalizer = weakref.finalize(self, self._landmarker.close)
        self._finalizer.atexit = False
        register = getattr(threading, "_register_atexit", None)
        try:
            if register is not None:
                register(self._finalizer)
            else:  # pragma: no cover - very old Python
                self._finalizer.atexit = True
        except RuntimeError:  # interpreter already shutting down
            pass

    def detect(self, frame_bgr: np.ndarray,
               timestamp_ms: Optional[int] = None) -> List[Hand]:
        """Detect hands in a BGR image. Returns hands sorted by size (largest first)."""
        if frame_bgr.ndim == 2:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
        elif frame_bgr.shape[2] == 4:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(rgb))
        if self._video:
            if timestamp_ms is None:
                timestamp_ms = int(time.monotonic() * 1000)
            # MediaPipe rejects timestamps that do not strictly increase.
            timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
            self._last_ts = timestamp_ms
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        else:
            result = self._landmarker.detect(image)

        hands: List[Hand] = []
        for i, landmarks in enumerate(result.hand_landmarks):
            coords = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
            category = None
            if i < len(result.handedness) and result.handedness[i]:
                category = result.handedness[i][0]
            hands.append(Hand(
                landmarks=coords,
                handedness=category.category_name if category else "Unknown",
                score=float(category.score) if category else 0.0,
                image_size=(width, height),
            ))
        hands.sort(key=_hand_area, reverse=True)
        return hands

    def close(self) -> None:
        self._finalizer()  # runs landmarker.close() once

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _hand_area(hand: Hand) -> float:
    x0, y0, x1, y1 = hand.bbox()
    return float((x1 - x0) * (y1 - y0))


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def draw_hand(image: np.ndarray, hand: Hand,
              color: Sequence[int] = (80, 220, 100), thickness: int = 2) -> None:
    """Draw the hand skeleton in place."""
    pts = np.round(hand.points[:, :2]).astype(int)
    scale = max(1.0, min(image.shape[:2]) / 480)
    t = max(1, int(round(thickness * scale)))
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, tuple(pts[a]), tuple(pts[b]), tuple(color), t, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        r = int(round((5 if i in FINGERTIPS else 3) * scale))
        cv2.circle(image, (x, y), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), r, tuple(color), max(1, t // 2), cv2.LINE_AA)


def read_image(path: Union[str, Path]) -> Optional[np.ndarray]:
    """cv2.imread that also works with non-ASCII paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
