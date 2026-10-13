"""Camera helpers."""
from __future__ import annotations

import sys
from typing import Optional

import cv2


def open_camera(index: int = 0, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """Open a webcam, trying the backends that work best on each OS."""
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]  # DSHOW opens much faster
    else:
        backends = [cv2.CAP_ANY]
    for api in backends:
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    raise RuntimeError(
        f"Could not open camera {index}. Try another index (--camera 1), close other "
        "apps using the camera, and on macOS allow camera access for your terminal "
        "(System Settings > Privacy & Security > Camera).")


class FPSMeter:
    """Exponentially smoothed frames per second."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.fps = 0.0
        self._last: Optional[float] = None

    def tick(self, t: float) -> float:
        if self._last is not None and t > self._last:
            inst = 1.0 / (t - self._last)
            self.fps = inst if self.fps == 0 else (1 - self.alpha) * self.fps + self.alpha * inst
        self._last = t
        return self.fps


def create_window(name: str, width: int = 1280, height: int = 720) -> None:
    """Resizable window without OpenCV's Qt toolbar, sized to fit most laptop screens."""
    flags = cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0) | getattr(cv2, "WINDOW_GUI_NORMAL", 0)
    cv2.namedWindow(name, flags)
    scale = min(1.0, 1280 / max(width, 1), 720 / max(height, 1))
    cv2.resizeWindow(name, int(width * scale), int(height * scale))


def window_closed(name: str) -> bool:
    """True once the user closes the OpenCV window with the mouse."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True
