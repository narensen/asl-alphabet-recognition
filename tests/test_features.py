import numpy as np

from asl.features import FEATURE_DIM, augment, features, mirror, normalize, palm_size
from conftest import canonical_hand, place


def test_normalize_removes_position_and_scale():
    hand = canonical_hand(["index", "middle"])
    a = normalize(place(hand, (100, 200), 80))
    b = normalize(place(hand, (900, 500), 240))
    assert np.allclose(a, b, atol=1e-5)
    assert np.allclose(a[0], 0)
    assert np.isclose(palm_size(a), 1.0, atol=1e-5)


def test_orientation_is_kept():
    hand = canonical_hand(["index", "middle"])  # U vs H differ only by orientation
    up = features(normalize(place(hand, (300, 300), 100, 0)))
    sideways = features(normalize(place(hand, (300, 300), 100, 90)))
    assert np.abs(up - sideways).max() > 0.5


def test_features_shape_and_batching():
    hands = np.stack([place(canonical_hand(), (300, 300), 100, a) for a in (0, 10, 20)])
    f = features(normalize(hands))
    assert f.shape == (3, FEATURE_DIM)
    assert np.allclose(f[1], features(normalize(hands[1])))


def test_mirror_is_an_involution():
    n = normalize(place(canonical_hand(), (300, 300), 100, 15))
    assert np.allclose(mirror(mirror(n)), n)
    assert not np.allclose(mirror(n), n)


def test_augment_keeps_normalization(rng):
    batch = normalize(np.stack([place(canonical_hand(), (300, 300), 100)] * 64))
    out = augment(batch, rng)
    assert out.shape == batch.shape and np.isfinite(out).all()
    assert np.allclose(out[:, 0], 0, atol=1e-5)
    assert np.allclose(palm_size(out), 1.0, atol=1e-4)
    assert not np.allclose(out, batch)
    # Augmentation is mild: augmented hands stay close to the original.
    assert np.abs(out - batch).mean() < 0.3
