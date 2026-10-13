"""Turning noisy per-frame predictions into typed text.

* ProbabilitySmoother - exponential moving average over class probabilities.
* StillnessMeter      - is the hand holding still? (static letters are held still)
* TextBuilder         - holding a letter still for `hold_s` seconds types it once.
                        To type a double letter ("LL"), bounce or slide the hand
                        a little and hold again, as signers do.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from .features import palm_center, palm_size


class ProbabilitySmoother:
    def __init__(self, alpha: float = 0.5):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._p: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._p = None

    def update(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        if self._p is None or self._p.shape != probs.shape:
            self._p = probs.copy()
        else:
            self._p = self.alpha * probs + (1 - self.alpha) * self._p
        return self._p


class StillnessMeter:
    """Palm speed in palm sizes per second, measured over a short window so
    landmark jitter doesn't register as movement."""

    def __init__(self, threshold: float = 0.8, window_s: float = 0.2):
        self.threshold = threshold
        self.window_s = window_s
        self.speed = float("inf")
        self._hist: Deque[Tuple[float, np.ndarray, float]] = deque()

    def reset(self) -> None:
        self._hist.clear()
        self.speed = float("inf")

    def update(self, t: float, points: np.ndarray) -> bool:
        self._hist.append((t, palm_center(points)[:2], float(palm_size(points))))
        while len(self._hist) > 2 and t - self._hist[1][0] >= self.window_s:
            self._hist.popleft()
        t0, c0, _ = self._hist[0]
        if t - t0 < 0.5 * self.window_s:
            return False  # hand just appeared: not enough history yet
        palm = max(np.median([h[2] for h in self._hist]), 1e-6)
        self.speed = float(np.linalg.norm(self._hist[-1][1] - c0) / palm / (t - t0))
        return self.speed < self.threshold


class TextBuilder:
    def __init__(self, hold_s: float = 0.8, min_conf: float = 0.7,
                 release_s: float = 0.2, bounce_shift: float = 0.25,
                 auto_space_s: float = 0.0, motion_margin_s: float = 0.3,
                 max_len: int = 500):
        self.hold_s = hold_s
        self.min_conf = min_conf
        self.release_s = release_s              # how long a letter must be "gone" before it can repeat
        self.bounce_shift = bounce_shift        # palm sizes the hand must move to repeat a letter
        self.auto_space_s = auto_space_s        # >0: type a space after this long without a hand
        self.motion_margin_s = motion_margin_s
        self.max_len = max_len
        self.text = ""
        self._log: List[Tuple[float, str]] = []  # (time, source) of each character in self.text
        self._cand: Optional[str] = None
        self._since = 0.0
        self._locked: Optional[str] = None
        self._lock_needs_new_label = False      # set after J/Z: moving doesn't release the lock
        self._release_since: Optional[float] = None
        self._no_hand_since: Optional[float] = None
        self._anchor: Optional[np.ndarray] = None  # palm position when the letter was typed

    # ----------------------------------------------------------------- #
    def update(self, t: float, label: Optional[str], conf: float, still: bool,
               hand_present: bool = True, center: Optional[np.ndarray] = None,
               palm: Optional[float] = None) -> Optional[str]:
        """Feed one frame. Returns the character typed on this frame, if any.

        `center` / `palm` (palm centre and palm size in pixels) are optional;
        with them, a small bounce of the hand is enough to repeat a letter.
        """
        if hand_present:
            self._no_hand_since = None
        elif self._no_hand_since is None:
            self._no_hand_since = t

        valid = label is not None and conf >= self.min_conf
        if self._locked is not None:
            self._update_lock(t, valid and label == self._locked, still, center, palm)

        if not hand_present and self.auto_space_s > 0 and self._no_hand_since is not None \
                and t - self._no_hand_since >= self.auto_space_s \
                and self.text and not self.text.endswith(" "):
            return self._append(" ", t, "auto")

        if not (valid and still) or label == self._locked:
            self._cand = None
            return None
        if label != self._cand:
            self._cand, self._since = label, t
            return None
        if t - self._since >= self.hold_s:
            self._cand = None
            self._locked, self._release_since = label, None
            self._lock_needs_new_label = False
            self._anchor = None if center is None else np.asarray(center, dtype=np.float64)
            return self._append(label, t, "static")
        return None

    def _update_lock(self, t: float, same: bool, still: bool,
                     center: Optional[np.ndarray], palm: Optional[float]) -> None:
        """Decide whether the last typed letter may be typed again.

        The lock is released when the letter has been gone (another label, no
        hand, low confidence, or moving) for release_s seconds, or - for static
        letters - when the hand bounces: a quick movement that takes it at
        least bounce_shift palm sizes from where the letter was typed. (Slow
        drift while holding a letter never counts, because the hand stays "still".)
        """
        if not self._lock_needs_new_label:
            if (not still and same and self._anchor is not None and center is not None
                    and palm and np.linalg.norm(np.asarray(center) - self._anchor) / palm
                    >= self.bounce_shift):
                self._locked = self._release_since = None
                return
            same = same and still  # after a static letter, moving counts as "gone"
        if same:
            self._release_since = None
            return
        if self._release_since is None:
            self._release_since = t
        if t - self._release_since >= self.release_s:
            self._locked = self._release_since = None

    def hold_progress(self, t: float) -> float:
        """0..1 progress of the letter currently being held."""
        if self._cand is None or self.hold_s <= 0:
            return 0.0
        return float(min(1.0, (t - self._since) / self.hold_s))

    @property
    def candidate(self) -> Optional[str]:
        return self._cand

    # ----------------------------------------------------------------- #
    def add_motion_letter(self, letter: str, run_start: float, t: float,
                          current_label: Optional[str] = None) -> str:
        """Type a motion letter (J/Z).

        `run_start` is when the J/Z handshape was first held. A static letter
        typed while holding it (the I before drawing a J, the D before a Z) is
        replaced by the motion letter; earlier letters are left alone.
        """
        cutoff = run_start - self.motion_margin_s
        while self._log and self._log[-1][1] == "static" and self._log[-1][0] >= cutoff:
            self.text, self._log = self.text[:-1], self._log[:-1]
        self._cand = None
        # The hand usually rests in the final J/Z pose; don't type it as a static letter.
        self._locked, self._release_since = current_label, None
        self._lock_needs_new_label = True
        return self._append(letter, t, "motion")

    def space(self, t: float = 0.0) -> None:
        if self.text and not self.text.endswith(" "):
            self._append(" ", t, "manual")

    def backspace(self) -> None:
        self.text, self._log = self.text[:-1], self._log[:-1]

    def clear(self) -> None:
        self.text, self._log = "", []
        self._cand = self._locked = self._release_since = None
        self._lock_needs_new_label = False

    def _append(self, s: str, t: float, source: str) -> str:
        self.text += s
        self._log.append((t, source))
        if len(self.text) > self.max_len:
            self.text, self._log = self.text[-self.max_len:], self._log[-self.max_len:]
        return s
