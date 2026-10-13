"""Per-frame recognition logic, kept separate from the camera and UI so it can
be tested without a webcam."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import NOTHING
from .features import palm_center, palm_size
from .landmarks import Hand
from .model import LetterClassifier
from .motion import MotionConfig, MotionEvent, MotionLetterDetector
from .text import ProbabilitySmoother, StillnessMeter, TextBuilder


@dataclass
class FrameResult:
    hand: Optional[Hand]
    label: Optional[str]                  # best static letter (None: no hand or "nothing")
    confidence: float
    top: List[Tuple[str, float]]          # top-3 (label, probability)
    still: bool
    hold_progress: float                  # 0..1 towards typing `label`
    typed: Optional[str] = None           # character typed on this frame
    motion: Optional[MotionEvent] = None  # J/Z detected on this frame


class Recognizer:
    def __init__(self, classifier: LetterClassifier, *, detect_motion: bool = True,
                 hold_s: float = 0.8, min_conf: float = 0.7, smoothing: float = 0.5,
                 still_speed: float = 0.8, auto_space_s: float = 0.0,
                 motion_config: Optional[MotionConfig] = None):
        self.classifier = classifier
        self.labels = list(classifier.labels)
        self.smoother = ProbabilitySmoother(alpha=smoothing)
        self.stillness = StillnessMeter(threshold=still_speed)
        self.text = TextBuilder(hold_s=hold_s, min_conf=min_conf, auto_space_s=auto_space_s)
        # Only look for motion letters the classifier doesn't already know.
        motion_letters = [c for c in ("J", "Z") if c not in self.labels]
        self.motion: Optional[MotionLetterDetector] = (
            MotionLetterDetector(motion_config, motion_letters)
            if detect_motion and motion_letters else None)
        self.motion_enabled = self.motion is not None

    def process(self, hand: Optional[Hand], t: float) -> FrameResult:
        """Update with the hand seen at time t (seconds), or None if no hand."""
        if hand is None:
            self.smoother.reset()
            self.stillness.reset()
            if self.motion is not None:
                self.motion.update(t, None)
            typed = self.text.update(t, None, 0.0, still=False, hand_present=False)
            return FrameResult(None, None, 0.0, [], False, 0.0, typed=typed)

        points = hand.points
        probs = self.smoother.update(self.classifier.predict_proba(points))
        order = np.argsort(probs)[::-1][:3]
        top = [(self.labels[i], float(probs[i])) for i in order]
        label, conf = top[0]
        if label == NOTHING:
            label = None
        still = self.stillness.update(t, points)

        event = None
        if self.motion is not None and self.motion_enabled:
            event = self.motion.update(t, points)
        if event is not None:
            typed = self.text.add_motion_letter(event.letter, event.run_start, t, current_label=label)
        else:
            typed = self.text.update(t, label, conf, still,
                                     center=palm_center(points)[:2], palm=float(palm_size(points)))
        return FrameResult(hand, label, conf, top, still, self.text.hold_progress(t),
                           typed=typed, motion=event)
