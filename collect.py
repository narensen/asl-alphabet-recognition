#!/usr/bin/env python3
"""Record hand-landmark samples for each letter from your webcam.

Press a letter key to record one "take" of that letter: after a short
countdown, the next --samples frames that contain a hand are saved to the CSV.
Move your hand a little while recording (closer and farther away, slight
tilts) so the model sees some variety.

Keys
  A-Y (not J)  record that letter (J and Z are detected from motion, no samples needed)
  0            record "nothing": relaxed hands and in-between poses, which helps the
               recogniser ignore transitions (optional, recommended)
  SPACE        cancel the take in progress
  BACKSPACE    cancel the take in progress, or undo the last saved take
  ESC          quit

Tip: record 3-5 takes per letter at different times, positions and lighting.
train.py keeps whole takes together when it sets aside validation data, so
the accuracy it reports reflects new recordings, not near-duplicate frames.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from asl.config import DEFAULT_DATA_CSV, MOTION_LETTERS, NOTHING, STATIC_LETTERS
from asl.data import append_rows, hand_to_row, label_counts, remove_take
from asl.landmarks import HandDetector, draw_hand
from asl.ui import GREEN, RED, WHITE, YELLOW, draw_collection
from asl.video import FPSMeter, create_window, open_camera, window_closed

WINDOW = "ASL data collection"
KEY_ESC, KEY_SPACE, KEY_NONE = 27, 32, 255
KEYS_BACKSPACE = (8, 127)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_DATA_CSV,
                   help="CSV file to append samples to (default: %(default)s)")
    p.add_argument("--samples", type=int, default=100, help="samples per take (default: 100)")
    p.add_argument("--every", type=int, default=2,
                   help="keep every n-th frame, for more variety per sample (default: 2)")
    p.add_argument("--countdown", type=float, default=1.5,
                   help="seconds to get ready before a take starts (default: 1.5)")
    p.add_argument("--target", type=int, default=300,
                   help="suggested samples per letter, used for the on-screen colours")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-mirror", action="store_true", help="don't mirror the preview")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = list(STATIC_LETTERS) + [NOTHING]
    counts = dict(label_counts(args.out))
    detector = HandDetector(mode="video", num_hands=1)
    cap = open_camera(args.camera, args.width, args.height)
    create_window(WINDOW, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width,
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height)
    fps = FPSMeter()

    state = "idle"            # idle -> countdown -> recording -> idle
    label = take = None
    rows: list = []
    frames = 0
    start_at = 0.0
    take_counter = 0
    saved_takes: list = []    # (take id, label) saved this session, for undo
    message, message_until = None, 0.0
    failures = 0

    def say(msg: str, seconds: float = 2.5) -> None:
        nonlocal message, message_until
        message, message_until = msg, time.monotonic() + seconds
        print(msg)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures > 30:
                    raise RuntimeError("The camera stopped delivering frames.")
                continue
            failures = 0
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()
            hands = detector.detect(frame, int(now * 1000))
            hand = hands[0] if hands else None
            if hand is not None:
                draw_hand(frame, hand, color=RED if state == "recording" else GREEN)

            if state == "countdown" and now >= start_at:
                state = "recording"
            if state == "recording" and hand is not None:
                frames += 1
                if frames % max(1, args.every) == 0:
                    rows.append(hand_to_row(label, take, "webcam", hand))
                if len(rows) >= args.samples:
                    append_rows(args.out, rows)
                    counts[label] = counts.get(label, 0) + len(rows)
                    saved_takes.append((take, label))
                    say(f"Saved {len(rows)} samples of '{label}'")
                    state, rows = "idle", []

            if state == "idle":
                status, color, progress = "Press a letter key to record it", WHITE, None
            elif state == "countdown":
                status, color, progress = f"Get ready: '{label}' in {start_at - now:.1f}s", YELLOW, None
            else:
                status = f"Recording '{label}': {len(rows)}/{args.samples}"
                status += "  - move your hand a little" if hand is not None else "  - no hand visible!"
                color, progress = RED, len(rows) / args.samples
            draw_collection(frame, status=status, status_color=color, progress=progress,
                            labels=labels, counts=counts, target=args.target,
                            current=label if state != "idle" else None,
                            message=message if now < message_until else None,
                            fps=fps.tick(now))
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == KEY_ESC or window_closed(WINDOW):
                break
            if key == KEY_NONE:
                continue
            if key in KEYS_BACKSPACE or key == KEY_SPACE:
                if state != "idle":
                    state, rows = "idle", []
                    say("Take cancelled")
                elif key in KEYS_BACKSPACE:
                    if saved_takes:
                        take_id, take_label = saved_takes.pop()
                        removed = remove_take(args.out, take_id)
                        counts[take_label] = max(0, counts.get(take_label, 0) - removed)
                        say(f"Removed the last take of '{take_label}' ({removed} samples)")
                    else:
                        say("Nothing to undo in this session")
            elif state == "idle" and key < 128:
                ch = chr(key).upper()
                if ch in MOTION_LETTERS:
                    say(f"{ch} is a motion letter: it is recognised from movement, no samples needed")
                elif ch == "0" or ch in STATIC_LETTERS:
                    label = NOTHING if ch == "0" else ch
                    take_counter += 1
                    take = f"{label}-{datetime.now():%Y%m%d-%H%M%S}-{take_counter}"
                    rows, frames = [], 0
                    state, start_at = "countdown", now + args.countdown
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()

    total = sum(counts.values())
    print(f"\n{args.out}: {total} samples")
    print("  " + "  ".join(f"{k}:{counts.get(k, 0)}" for k in labels))
    missing = [k for k in STATIC_LETTERS if counts.get(k, 0) == 0]
    if missing:
        print(f"  letters without samples: {' '.join(missing)}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        sys.exit(f"Error: {exc}")
