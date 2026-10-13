#!/usr/bin/env python3
"""Classify the ASL letter in one or more image files.

  python predict_image.py photo.jpg
  python predict_image.py images/*.jpg --save-dir predictions/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from asl.config import DEFAULT_MODEL_PATH
from asl.landmarks import HandDetector, draw_hand, read_image
from asl.model import LetterClassifier
from asl.ui import GREEN, text, ui_scale


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", type=Path, nargs="+")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--save-dir", type=Path, help="write annotated copies here")
    p.add_argument("--pad", type=float, default=0.25,
                   help="black border (fraction of image size) added before detection (default: 0.25)")
    args = p.parse_args()

    classifier = LetterClassifier(args.model)
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    with HandDetector(mode="image", num_hands=1) as detector:
        for path in args.images:
            image = read_image(path)
            if image is None:
                print(f"{path}: could not read image")
                continue
            if args.pad > 0:  # helps MediaPipe with hands that touch the image edge
                b = int(round(args.pad * max(image.shape[:2])))
                image = cv2.copyMakeBorder(image, b, b, b, b, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            hands = detector.detect(image)
            if not hands:
                print(f"{path}: no hand found")
                continue
            probs = classifier.predict_proba(hands[0].points)
            top = probs.argsort()[::-1][:3]
            guesses = "  ".join(f"{classifier.labels[i]} {probs[i]:.0%}" for i in top)
            print(f"{path}: {guesses}")
            if args.save_dir:
                draw_hand(image, hands[0], color=GREEN)
                s = ui_scale(image)
                label = f"{classifier.labels[top[0]]} {probs[top[0]]:.0%}"
                text(image, label, (int(10 * s), int(50 * s)), 1.5 * s, GREEN, 3 * s)
                cv2.imwrite(str(args.save_dir / path.name), image)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"Error: {exc}")
