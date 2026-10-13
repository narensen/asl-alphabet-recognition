"""OpenCV drawing helpers for the collection and recognition windows."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .landmarks import draw_hand

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

WHITE = (255, 255, 255)
GREY = (160, 160, 160)
GREEN = (90, 220, 110)
YELLOW = (60, 210, 250)
RED = (80, 80, 240)
CYAN = (230, 200, 60)
MAGENTA = (220, 90, 230)
PANEL = (28, 28, 28)

Color = Tuple[int, int, int]


def ui_scale(img: np.ndarray) -> float:
    return max(0.5, img.shape[0] / 720.0)


def panel(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
          color: Color = PANEL, alpha: float = 0.65) -> None:
    """Semi-transparent filled rectangle (blends only the region, so it's cheap)."""
    h, w = img.shape[:2]
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * (1 - alpha)
              + np.array(color, np.float32) * alpha).astype(np.uint8)


def text(img: np.ndarray, s: str, org: Tuple[int, int], scale: float,
         color: Color = WHITE, thickness: int = 1, font: int = FONT) -> None:
    x, y = int(org[0]), int(org[1])
    t = max(1, int(round(thickness)))
    # Drop shadow with the *same* thickness: OpenCV 5 switches glyph widths
    # with thickness, so a thicker shadow would drift out of line.
    off = 1 if t < 3 else 2
    cv2.putText(img, s, (x + off, y + off), font, scale, (0, 0, 0), t, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), font, scale, color, t, cv2.LINE_AA)


def text_width(s: str, scale: float, thickness: int = 1, font: int = FONT) -> int:
    return cv2.getTextSize(s, font, scale, max(1, int(round(thickness))))[0][0]


def bar(img: np.ndarray, x: int, y: int, w: int, h: int, frac: float,
        color: Color, bg: Color = (70, 70, 70)) -> None:
    frac = float(np.clip(frac, 0.0, 1.0))
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), bg, -1)
    if frac > 0:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w * frac), int(y + h)), color, -1)


def fit_tail(s: str, max_width: int, scale: float, thickness: int, font: int = FONT) -> str:
    """Keep the end of s that fits in max_width pixels."""
    if text_width(s, scale, thickness, font) <= max_width:
        return s
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi) // 2
        if text_width("..." + s[mid:], scale, thickness, font) <= max_width:
            hi = mid
        else:
            lo = mid + 1
    return "..." + s[lo:]


# --------------------------------------------------------------------------- #
# Recognition window
# --------------------------------------------------------------------------- #
def draw_recognition(frame: np.ndarray, result, typed_text: str, *, fps: float,
                     motion_on: bool, debug: bool, t: float,
                     trails: Iterable[np.ndarray] = (), flash: Optional[str] = None,
                     speed: float = 0.0, message: Optional[str] = None) -> None:
    """Draw everything for recognize.py onto `frame` in place.

    `result` is an asl.pipeline.FrameResult.
    """
    s = ui_scale(frame)
    h, w = frame.shape[:2]
    pad = int(16 * s)

    # Fingertip trail while a J/Z handshape is held.
    for trail in trails:
        if len(trail) >= 2:
            cv2.polylines(frame, [np.round(trail).astype(np.int32)], False, MAGENTA,
                          max(2, int(3 * s)), cv2.LINE_AA)

    label, conf = result.label, result.confidence
    confident = label is not None and conf >= 0.7
    if result.hand is not None:
        color = (GREEN if result.still else YELLOW) if confident else GREY
        draw_hand(frame, result.hand, color=color)
        x0, y0, x1, y1 = result.hand.bbox(pad=0.12)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, max(1, int(2 * s)), cv2.LINE_AA)
        if label is not None:
            tag = f"{label} {conf:.0%}"
            text(frame, tag, (x0, max(int(24 * s), y0 - int(8 * s))), 0.8 * s, color, 2 * s)

    # Letter panel.
    pw, ph = int(210 * s), int(230 * s)
    panel(frame, pad, pad, pad + pw, pad + ph)
    text(frame, "LETTER", (pad + int(12 * s), pad + int(28 * s)), 0.6 * s, GREY, 1 * s)
    big = label if label is not None else "-"
    big_scale = (4.2 if len(big) == 1 else 1.6) * s
    bw = text_width(big, big_scale, 7 * s, FONT_BOLD)
    big_color = (GREEN if result.still else YELLOW) if confident else GREY
    text(frame, big, (pad + (pw - bw) // 2, pad + int(150 * s)), big_scale, big_color, 7 * s, FONT_BOLD)
    bx, bwid = pad + int(12 * s), pw - int(24 * s)
    bar(frame, bx, pad + int(172 * s), bwid, int(12 * s), conf, big_color)
    text(frame, f"confidence {conf:.0%}", (bx, pad + int(202 * s)), 0.5 * s, WHITE, 1 * s)
    bar(frame, bx, pad + int(212 * s), bwid, int(6 * s), result.hold_progress, CYAN)

    # Debug: top-3 and hand info.
    if debug:
        y = pad + ph + int(10 * s)
        panel(frame, pad, y, pad + pw, y + int(150 * s))
        yy = y + int(28 * s)
        for name, p in result.top:
            text(frame, f"{name}", (bx, yy), 0.6 * s, WHITE, 1 * s)
            bar(frame, bx + int(80 * s), yy - int(12 * s), bwid - int(80 * s), int(12 * s), p, CYAN)
            yy += int(28 * s)
        info = f"speed {speed:.2f} palm/s" if np.isfinite(speed) else "speed -"
        text(frame, info, (bx, yy), 0.5 * s, GREY, 1 * s)
        if result.hand is not None:
            text(frame, f"hand {result.hand.handedness} {result.hand.score:.2f}",
                 (bx, yy + int(24 * s)), 0.5 * s, GREY, 1 * s)

    # Status (top right).
    status = [f"FPS {fps:4.1f}", f"J/Z {'on' if motion_on else 'off'}"]
    for i, line in enumerate(status):
        tw = text_width(line, 0.6 * s, 1 * s)
        text(frame, line, (w - pad - tw, pad + int(24 * s) + i * int(26 * s)), 0.6 * s, WHITE, 1 * s)

    # Motion-letter banner.
    if flash:
        msg = f"{flash}  (motion)"
        tw = text_width(msg, 1.6 * s, 3 * s, FONT_BOLD)
        cx = (w - tw) // 2
        panel(frame, cx - pad, pad, cx + tw + pad, pad + int(64 * s), (60, 20, 60), 0.7)
        text(frame, msg, (cx, pad + int(48 * s)), 1.6 * s, MAGENTA, 3 * s, FONT_BOLD)

    # Typed text (bottom).
    bh = int(92 * s)
    panel(frame, 0, h - bh, w, h, alpha=0.7)
    hints = "[SPACE] space  [BKSP] delete  [C] clear  [M] J/Z  [D] debug  [S] save  [Q] quit"
    text(frame, message or hints, (pad, h - bh + int(24 * s)), 0.5 * s,
         YELLOW if message else GREY, 1 * s)
    cursor = "_" if int(t * 2) % 2 == 0 else " "
    shown = fit_tail(typed_text + cursor, w - 2 * pad, 1.3 * s, 2 * s, FONT_BOLD)
    text(frame, shown, (pad, h - int(24 * s)), 1.3 * s, WHITE, 2 * s, FONT_BOLD)


# --------------------------------------------------------------------------- #
# Collection window
# --------------------------------------------------------------------------- #
def draw_collection(frame: np.ndarray, *, status: str, status_color: Color,
                    progress: Optional[float], labels: Sequence[str],
                    counts: Dict[str, int], target: int, current: Optional[str],
                    message: Optional[str], fps: float) -> None:
    s = ui_scale(frame)
    h, w = frame.shape[:2]
    pad = int(16 * s)

    # Status panel (top).
    panel(frame, 0, 0, w, int(92 * s), alpha=0.7)
    text(frame, status, (pad, int(38 * s)), 0.9 * s, status_color, 2 * s)
    help_line = ("Press a letter key to record it  |  0 = 'nothing' (non-letter poses)  |  "
                 "BKSP = undo last take  |  SPACE = cancel  |  ESC = quit")
    text(frame, fit_tail(help_line, w - 2 * pad, 0.5 * s, 1 * s), (pad, int(70 * s)), 0.5 * s, GREY, 1 * s)
    fps_s = f"FPS {fps:4.1f}"
    text(frame, fps_s, (w - pad - text_width(fps_s, 0.5 * s, 1 * s), int(38 * s)), 0.5 * s, GREY, 1 * s)
    if progress is not None:
        bar(frame, 0, int(88 * s), w, int(6 * s), progress, RED)

    # Per-label counts (bottom grid).
    cols = 13
    rows = (len(labels) + cols - 1) // cols
    cell_w, cell_h = (w - 2 * pad) / cols, int(44 * s)
    top = h - rows * cell_h - pad
    panel(frame, 0, top - int(34 * s), w, h, alpha=0.7)
    text(frame, f"samples per letter (aim for {target}+, from several takes)",
         (pad, top - int(10 * s)), 0.5 * s, GREY, 1 * s)
    for i, name in enumerate(labels):
        r, c = divmod(i, cols)
        x = int(pad + c * cell_w)
        y = top + r * cell_h
        n = counts.get(name, 0)
        color = GREEN if n >= target else (YELLOW if n >= target // 3 else RED)
        if name == current:
            cv2.rectangle(frame, (x, y), (int(x + cell_w - 4 * s), y + cell_h - int(4 * s)), CYAN, 2)
        short = "0:nothing" if name == "nothing" else name
        text(frame, short, (x + int(6 * s), y + int(18 * s)), 0.55 * s, WHITE, 1 * s)
        text(frame, str(n), (x + int(6 * s), y + int(38 * s)), 0.5 * s, color, 1 * s)

    if message:
        tw = text_width(message, 0.8 * s, 2 * s)
        panel(frame, (w - tw) // 2 - pad, int(110 * s), (w + tw) // 2 + pad, int(160 * s), alpha=0.7)
        text(frame, message, ((w - tw) // 2, int(145 * s)), 0.8 * s, YELLOW, 2 * s)
