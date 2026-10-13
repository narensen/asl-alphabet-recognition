#!/usr/bin/env python3
"""Real-time ASL fingerspelling recognition from your webcam (or a video file).

Hold a letter still for a moment to type it. J and Z are recognised from
their movement: hold the I handshape and draw a J with your pinky, or point
your index finger and draw a Z.

Keys
  SPACE      type a space          BACKSPACE  delete the last character
  C          clear the text        S          save the text to --text-file
  M          toggle J/Z detection  D          toggle the debug overlay
  Q / ESC    quit

Examples
  python recognize.py
  python recognize.py --camera 1 --hold 0.6
  python recognize.py --video clip.mp4 --save-video annotated.mp4 --headless
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import torch

from asl.config import DEFAULT_MODEL_PATH
from asl.landmarks import HandDetector
from asl.model import LetterClassifier
from asl.pipeline import Recognizer
from asl.ui import draw_recognition
from asl.video import FPSMeter, create_window, open_camera, window_closed

WINDOW = "ASL alphabet recognition"
KEY_ESC, KEY_SPACE, KEY_NONE = 27, 32, 255
KEYS_BACKSPACE = (8, 127)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    p.add_argument("--video", type=Path, help="run on a video file instead of the webcam")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--mirror", choices=["auto", "yes", "no"], default="auto",
                   help="mirror the image (auto: webcam yes, video no)")
    p.add_argument("--hold", type=float, default=0.8,
                   help="seconds a letter must be held still to be typed (default: 0.8)")
    p.add_argument("--min-conf", type=float, default=0.7,
                   help="minimum smoothed confidence to type a letter (default: 0.7)")
    p.add_argument("--smoothing", type=float, default=0.5,
                   help="weight of the newest frame in the probability average (1 = none)")
    p.add_argument("--auto-space", type=float, default=0.0,
                   help="type a space after this many seconds without a hand (0 = off)")
    p.add_argument("--no-motion", action="store_true", help="disable J/Z motion detection")
    p.add_argument("--debug", action="store_true", help="start with the debug overlay on")
    p.add_argument("--save-video", type=Path, help="also write the annotated frames to this .mp4")
    p.add_argument("--headless", action="store_true", help="no window (for --video processing)")
    p.add_argument("--text-file", type=Path, default=Path("typed_text.txt"),
                   help="where S saves the typed text (default: %(default)s)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.headless and not args.video:
        sys.exit("--headless needs --video (there would be no way to stop the webcam loop).")
    torch.set_num_threads(1)  # a tiny MLP: threads only add latency

    classifier = LetterClassifier(args.model)
    recognizer = Recognizer(classifier, detect_motion=not args.no_motion, hold_s=args.hold,
                            min_conf=args.min_conf, smoothing=args.smoothing,
                            auto_space_s=args.auto_space)
    detector = HandDetector(mode="video", num_hands=1)
    print(f"Model: {args.model} ({len(classifier.labels)} classes: {' '.join(classifier.labels)})")
    if recognizer.motion is not None:
        print(f"Motion letters: {' '.join(recognizer.motion.letters)}")

    if args.video:
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            sys.exit(f"Could not open video {args.video}")
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        cap = open_camera(args.camera, args.width, args.height)
        video_fps = 0.0
    mirror = args.mirror == "yes" or (args.mirror == "auto" and not args.video)

    if not args.headless:
        create_window(WINDOW, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width,
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height)
    writer: Optional[cv2.VideoWriter] = None
    fps = FPSMeter()
    debug = args.debug
    flash, flash_until = None, 0.0
    message, message_until = None, 0.0
    frame_index = failures = 0
    shown = ""  # text last printed to the console

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.video:
                    break
                failures += 1
                if failures > 30:
                    raise RuntimeError("The camera stopped delivering frames.")
                continue
            failures = 0
            if mirror:
                frame = cv2.flip(frame, 1)
            # Video files use their own clock so timing doesn't depend on processing speed.
            t = frame_index / video_fps if args.video else time.monotonic()
            frame_index += 1

            hands = detector.detect(frame, int(t * 1000))
            result = recognizer.process(hands[0] if hands else None, t)
            if recognizer.text.text != shown:  # mirror the typed text in the console
                pad = " " * max(0, len(shown) - len(recognizer.text.text))
                shown = recognizer.text.text
                print(f"\rTyped: {shown}{pad}", end="", flush=True)
            if result.motion is not None:
                flash, flash_until = result.motion.letter, t + 1.0

            trails = []
            if recognizer.motion is not None and recognizer.motion_enabled:
                trails = [recognizer.motion.trail(letter) for letter in recognizer.motion.letters]
            draw_recognition(frame, result, recognizer.text.text, fps=fps.tick(time.monotonic()),
                             motion_on=recognizer.motion_enabled, debug=debug, t=t, trails=trails,
                             flash=flash if t < flash_until else None,
                             speed=recognizer.stillness.speed,
                             message=message if t < message_until else None)

            if args.save_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(args.save_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                             video_fps or 30.0, (w, h))
                writer.write(frame)
            if args.headless:
                continue

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (KEY_ESC, ord("q"), ord("Q")) or window_closed(WINDOW):
                break
            if key == KEY_NONE:
                continue
            text = recognizer.text
            if key == KEY_SPACE:
                text.space(t)
            elif key in KEYS_BACKSPACE:
                text.backspace()
            elif key in (ord("c"), ord("C")):
                text.clear()
            elif key in (ord("d"), ord("D")):
                debug = not debug
            elif key in (ord("m"), ord("M")):
                if recognizer.motion is None:
                    message = "J/Z detection unavailable (disabled, or the model already has J/Z)"
                else:
                    recognizer.motion_enabled = not recognizer.motion_enabled
                    recognizer.motion.reset()
                    message = f"J/Z detection {'on' if recognizer.motion_enabled else 'off'}"
                message_until = t + 2.0
            elif key in (ord("s"), ord("S")):
                with open(args.text_file, "a", encoding="utf-8") as fh:
                    fh.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{text.text}\n")
                message, message_until = f"Saved to {args.text_file}", t + 2.0
    finally:
        cap.release()
        detector.close()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"\n\nTyped text: {recognizer.text.text!r}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"Error: {exc}")
