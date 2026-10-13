"""Project-wide paths and constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

DEFAULT_DATA_CSV = DATA_DIR / "landmarks.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "asl_classifier.pt"

# MediaPipe hand landmark model (downloaded automatically on first run).
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# J and Z are signed with motion, so the static classifier covers the other 24
# letters and asl/motion.py recognises J and Z from fingertip trajectories.
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MOTION_LETTERS = ("J", "Z")
STATIC_LETTERS = tuple(c for c in ALPHABET if c not in MOTION_LETTERS)

# Optional "no sign" class: record it with key 0 in collect.py so the model can
# say "this is not a letter" during transitions.
NOTHING = "nothing"

NUM_LANDMARKS = 21
