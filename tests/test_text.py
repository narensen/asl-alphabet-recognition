import numpy as np

from asl.motion import j_template
from asl.pipeline import Recognizer
from asl.text import ProbabilitySmoother, StillnessMeter, TextBuilder
from conftest import canonical_hand, hand_from_points, moving_hand_frames, place

FPS = 30.0


def feed(tb: TextBuilder, t0: float, seconds: float, label, conf=0.95, still=True, hand=True):
    typed = ""
    n = int(round(seconds * FPS))
    for k in range(n):
        out = tb.update(t0 + k / FPS, label, conf, still, hand_present=hand)
        typed += out or ""
    return typed, t0 + n / FPS


def test_holding_types_once():
    tb = TextBuilder(hold_s=0.8)
    typed, t = feed(tb, 0.0, 0.7, "A")
    assert typed == ""
    typed, t = feed(tb, t, 3.0, "A")
    assert typed == "A" and tb.text == "A"


def test_short_glitch_does_not_repeat_letter():
    tb = TextBuilder(hold_s=0.8)
    _, t = feed(tb, 0.0, 1.0, "A")
    _, t = feed(tb, t, 2 / FPS, "S")          # two noisy frames
    _, t = feed(tb, t, 2.0, "A")
    assert tb.text == "A"


def test_bounce_types_double_letter():
    tb = TextBuilder(hold_s=0.8)
    _, t = feed(tb, 0.0, 1.0, "L")
    _, t = feed(tb, t, 0.3, "L", still=False)  # small bounce
    _, t = feed(tb, t, 1.0, "L")
    assert tb.text == "LL"


def test_needs_confidence_and_stillness():
    tb = TextBuilder(hold_s=0.5, min_conf=0.7)
    feed(tb, 0.0, 2.0, "B", conf=0.5)
    feed(tb, 2.0, 2.0, "B", still=False)
    feed(tb, 4.0, 2.0, None)
    assert tb.text == ""


def test_sequence_of_letters():
    tb = TextBuilder(hold_s=0.5)
    t = 0.0
    for letter in "HELLO":
        _, t = feed(tb, t, 0.3, letter, still=False)  # moving into the letter
        _, t = feed(tb, t, 0.7, letter)
    assert tb.text == "HELLO"


def test_auto_space_and_editing():
    tb = TextBuilder(hold_s=0.5, auto_space_s=1.0)
    _, t = feed(tb, 0.0, 0.8, "H")
    _, t = feed(tb, t, 0.8, "I")
    _, t = feed(tb, t, 2.0, None, hand=False)
    assert tb.text == "HI "
    tb.backspace()
    tb.space()
    tb.space()
    assert tb.text == "HI "
    tb.clear()
    assert tb.text == ""


def test_motion_letter_replaces_the_handshape_letter():
    tb = TextBuilder(hold_s=0.5)
    _, t = feed(tb, 0.0, 1.0, "A")
    _, t = feed(tb, t, 0.3, None, still=False)
    run_start = t
    _, t = feed(tb, t, 1.0, "I")               # holding the I handshape types I...
    assert tb.text == "AI"
    tb.add_motion_letter("J", run_start=run_start, t=t + 0.8, current_label="I")
    assert tb.text == "AJ"                     # ...which the J then replaces
    # The hand rests in the final pose: must not type "I" again.
    feed(tb, t + 0.8, 2.0, "I")
    assert tb.text == "AJ"


def test_motion_letters_keep_earlier_letters():
    tb = TextBuilder(hold_s=0.5)
    _, t = feed(tb, 0.0, 1.0, "A")             # typed before the Z handshape appeared
    tb.add_motion_letter("Z", run_start=t + 0.1, t=t + 1.5, current_label="D")
    tb.add_motion_letter("Z", run_start=t + 1.6, t=t + 3.0, current_label="D")
    assert tb.text == "AZZ"


def test_smoother_and_stillness():
    sm = ProbabilitySmoother(alpha=0.5)
    sm.update(np.array([1.0, 0.0]))
    p = sm.update(np.array([0.0, 1.0]))
    assert np.allclose(p, [0.5, 0.5])

    meter = StillnessMeter(threshold=0.8)
    hand = canonical_hand()
    rng = np.random.default_rng(0)
    states = [meter.update(k / FPS, place(hand, (640, 360), 100) + rng.normal(0, 1.5, (21, 3)))
              for k in range(30)]
    assert all(states[10:])                   # jitter isn't movement
    moving = [meter.update(1 + k / FPS, place(hand, (640 + 15 * k, 360), 100)) for k in range(10)]
    assert not any(moving[3:])                # 15 px/frame ~ 4.6 palm/s


class FakeClassifier:
    """Predicts from finger extension, standing in for a trained model."""
    labels = ["B", "D", "I"]

    def predict_proba(self, points):
        from asl.motion import extended_fingers
        ext = extended_fingers(points)
        if all(ext.values()):
            return np.array([0.96, 0.02, 0.02])
        if ext["index"]:
            return np.array([0.02, 0.96, 0.02])
        return np.array([0.02, 0.02, 0.96])


def test_recognizer_types_static_and_motion_letters():
    rec = Recognizer(FakeClassifier(), hold_s=0.6)
    t = 0.0

    def show(canonical, seconds, center=(640, 360)):
        nonlocal t
        for _ in range(int(seconds * FPS)):
            rec.process(hand_from_points(place(canonical, center, 100)), t)
            t += 1 / FPS

    def nothing(seconds):
        nonlocal t
        for _ in range(int(seconds * FPS)):
            rec.process(None, t)
            t += 1 / FPS

    show(canonical_hand(), 1.0)                    # B
    nothing(0.5)
    show(canonical_hand(["index"], False), 1.0)    # D
    nothing(0.5)
    pinky = canonical_hand(["pinky"], False)
    show(pinky, 1.0, center=(640, 300))            # I ... then draw a J with the pinky
    path = np.array(j_template(0.6, 0.3, 180)) * 250
    path = np.array([np.interp(np.linspace(0, len(path) - 1, 24), np.arange(len(path)), path[:, k])
                     for k in range(2)]).T
    start = place(pinky, (640, 300), 100)[20, :2]
    path += start - path[0]
    for tt, pts in moving_hand_frames(pinky, 20, path, FPS, t):
        rec.process(hand_from_points(pts), tt)
    t = tt + 1 / FPS
    end_center = place(pinky, (640, 300), 100)[0, :2] + (path[-1] - path[0])
    show(pinky, 1.5, center=end_center)            # resting in the final pose
    assert rec.text.text == "BDJ"


def _hold_at(tb, t, label, center, palm, seconds, still=True):
    for _ in range(int(round(seconds * FPS))):
        tb.update(t, label, 0.95, still, center=np.asarray(center, float), palm=palm)
        t += 1 / FPS
    return t


def test_quick_small_bounce_repeats_letter():
    """A bounce too short to count as 0.2 s of movement still types a double letter."""
    tb = TextBuilder(hold_s=0.8)
    c0, palm = np.array([640.0, 360.0]), 200.0
    t = _hold_at(tb, 0.0, "L", c0, palm, 1.0)
    assert tb.text == "L"
    for k, dy in enumerate([20, 40, 60, 40, 20, 0]):   # 0.3 palm down and back, flickering
        tb.update(t, "L", 0.95, still=(k % 2 == 1), center=c0 + (0, dy), palm=palm)
        t += 1 / FPS
    _hold_at(tb, t, "L", c0, palm, 1.0)
    assert tb.text == "LL"


def test_slow_drift_while_holding_does_not_repeat():
    tb = TextBuilder(hold_s=0.8)
    palm, t = 200.0, 0.0
    for k in range(int(4 * FPS)):                     # drifts 0.5 palm over 4 s, always "still"
        tb.update(t, "A", 0.95, True, center=np.array([640.0 + 100 * k / (4 * FPS), 360.0]), palm=palm)
        t += 1 / FPS
    assert tb.text == "A"
