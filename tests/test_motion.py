import math

import numpy as np
import pytest

from asl.motion import (MotionLetterDetector, extended_fingers, j_template, match_distance,
                        default_templates, resample, z_template)
from conftest import canonical_hand, moving_hand_frames, place

PINKY_ONLY = canonical_hand(["pinky"], thumb_out=False)   # the I handshape (J starts here)
INDEX_ONLY = canonical_hand(["index"], thumb_out=False)   # the Z handshape
OPEN_HAND = canonical_hand()
PINKY_TIP, INDEX_TIP = 20, 8


def shape_path(shape, height=250.0, start=(640.0, 200.0), n=24, mirror=False, ease=False):
    """Unit shape -> pixel path `height` px on its longer side, n samples."""
    p = resample(np.asarray(shape, float), 200)
    if mirror:
        p[:, 0] *= -1
    p = (p - p.min(0)) / (p.max(0) - p.min(0)).max() * height
    u = np.linspace(0, 1, n)
    if ease:  # slow at the start and end, like a real stroke
        u = 0.5 - 0.5 * np.cos(math.pi * u)
    idx = np.round(u * (len(p) - 1)).astype(int)
    return p[idx] + np.asarray(start)


def hold_frames(canonical, tip, at, seconds, t0, fps=30.0, mirror=False):
    n = int(seconds * fps)
    return list(moving_hand_frames(canonical, tip, np.repeat([at], n, axis=0), fps, t0, mirror=mirror))


def gesture(canonical, tip, path, fps=30.0, noise=0.0, mirror=False, pre=0.5, post=0.5):
    """Hold still, move the fingertip along `path`, hold still again."""
    frames = hold_frames(canonical, tip, path[0], pre, 0.0, fps, mirror)
    t = len(frames) / fps
    frames += list(moving_hand_frames(canonical, tip, path, fps, t, noise=noise, mirror=mirror))
    t += len(path) / fps
    frames += hold_frames(canonical, tip, path[-1], post, t, fps, mirror)
    return frames


def run(frames, detector=None):
    detector = detector or MotionLetterDetector()
    return [e for e in (detector.update(t, pts) for t, pts in frames) if e is not None]


# --------------------------------------------------------------------------- #
def test_extended_fingers_is_rotation_and_scale_invariant():
    for angle, size in [(0, 100), (90, 60), (-35, 250), (180, 120)]:
        ext = extended_fingers(place(INDEX_ONLY, (300, 300), size, angle))
        assert ext == {"index": True, "middle": False, "ring": False, "pinky": False}
        ext = extended_fingers(place(OPEN_HAND, (300, 300), size, angle))
        assert all(ext.values())


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("noise", [0.0, 2.0])
@pytest.mark.parametrize("shape", [j_template(0.6, 0.3, 180), j_template(0.4, 0.3, 120),
                                   j_template(0.2, 0.45, 170)])
def test_detects_j(shape, noise, mirror):
    path = shape_path(shape, mirror=mirror, ease=True)
    events = run(gesture(PINKY_ONLY, PINKY_TIP, path, noise=noise, mirror=mirror))
    assert [e.letter for e in events] == ["J"]


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("noise", [0.0, 2.0])
@pytest.mark.parametrize("shape", [z_template(), [(0, 0), (1, 0.1), (0.1, 1), (1.1, 1.05)],
                                   z_template() * [1.0, 0.7]])
def test_detects_z(shape, noise, mirror):
    path = shape_path(shape, n=30, mirror=mirror, ease=True)
    events = run(gesture(INDEX_ONLY, INDEX_TIP, path, noise=noise, mirror=mirror))
    assert [e.letter for e in events] == ["Z"]


def test_slow_gesture_is_detected():
    path = shape_path(z_template(), n=45)  # 1.5 s at 30 fps
    assert [e.letter for e in run(gesture(INDEX_ONLY, INDEX_TIP, path))] == ["Z"]


@pytest.mark.parametrize("name, canonical, tip, shape", [
    ("I moving straight down", PINKY_ONLY, PINKY_TIP, [(0, 0), (0, 1)]),
    ("I moving diagonally", PINKY_ONLY, PINKY_TIP, [(0, 0), (0.6, 1)]),
    ("index swipe right", INDEX_ONLY, INDEX_TIP, [(0, 0), (1, 0)]),
    ("index draws a 7", INDEX_ONLY, INDEX_TIP, [(0, 0), (1, 0), (0, 1)]),
    ("index draws an N", INDEX_ONLY, INDEX_TIP, [(0, 1), (0, 0), (1, 1), (1, 0)]),
    ("open hand draws a Z", OPEN_HAND, INDEX_TIP, z_template()),
    ("open hand draws a J", OPEN_HAND, PINKY_TIP, j_template(0.6, 0.3, 180)),
])
def test_rejects_other_motions(name, canonical, tip, shape):
    path = shape_path(shape, n=30)
    assert run(gesture(canonical, tip, path, noise=1.0)) == [], name


def test_rejects_circle():
    a = np.linspace(0, 2 * math.pi, 40)
    path = np.stack([640 + 120 * np.cos(a), 300 + 120 * np.sin(a)], 1)
    assert run(gesture(INDEX_ONLY, INDEX_TIP, path)) == []


def test_rejects_tiny_z():
    path = shape_path(z_template(), height=40.0, n=30)  # well under one palm size
    assert run(gesture(INDEX_ONLY, INDEX_TIP, path)) == []


def test_still_hand_with_jitter_never_fires():
    rng = np.random.default_rng(0)
    frames = []
    for k in range(150):
        pts = place(INDEX_ONLY, (640, 360), 100)
        pts[:, :2] += rng.normal(0, 2.0, (21, 2))
        frames.append((k / 30, pts))
    assert run(frames) == []


def test_one_event_per_gesture_and_lost_hand_resets():
    detector = MotionLetterDetector()
    path = shape_path(z_template(), n=30)
    frames = gesture(INDEX_ONLY, INDEX_TIP, path, post=2.0)
    assert len(run(frames, detector)) == 1
    # Half a Z, hand disappears for a while, then the other half: no event.
    t = frames[-1][0] + 1.0
    half = gesture(INDEX_ONLY, INDEX_TIP, path[:15], pre=0.2, post=0.0)
    run([(t + ft, p) for ft, p in half], detector)
    t_gap = t + half[-1][0]
    detector.update(t_gap + 0.1, None)
    detector.update(t_gap + 0.6, None)
    rest = gesture(INDEX_ONLY, INDEX_TIP, path[15:], pre=0.0, post=0.3)
    assert run([(t_gap + 0.7 + ft, p) for ft, p in rest], detector) == []


def test_template_scores_separate_z_from_other_shapes():
    templates = default_templates()
    z_scores = [match_distance(resample(z_template(), 40), templates["Z"])]
    others = [np.array([(0, 0), (1, 0), (0, 1)]), np.array([(0, 0), (0, 1)]),
              np.array([(0, 1), (0, 0), (1, 1), (1, 0)])]
    other_scores = [match_distance(resample(o, 40), templates["Z"]) for o in others]
    assert max(z_scores) < 0.05 < 0.25 < min(other_scores)


def test_approach_movement_is_not_part_of_the_gesture():
    """Moving into position, pausing, then drawing: only the drawing counts."""
    path = shape_path(z_template(), n=30, ease=True)
    approach = np.linspace(path[0] + (-150, 120), path[0], 9)
    frames = list(moving_hand_frames(INDEX_ONLY, INDEX_TIP, approach, 30.0, 0.0))
    frames += hold_frames(INDEX_ONLY, INDEX_TIP, path[0], 0.8, 0.3)
    frames += list(moving_hand_frames(INDEX_ONLY, INDEX_TIP, path, 30.0, 1.1))
    frames += hold_frames(INDEX_ONLY, INDEX_TIP, path[-1], 0.5, 2.1)
    events = run(frames)
    assert [e.letter for e in events] == ["Z"]
    assert events[0].t_start >= 1.0 and events[0].run_start == 0.0
