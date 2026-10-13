import numpy as np
import pytest
import torch

import train as train_script
from asl.data import append_rows, hand_to_row, label_counts, load_dataset, remove_take
from asl.features import FEATURE_DIM
from asl.model import LandmarkMLP, LetterClassifier, load_classifier, save_classifier
from conftest import canonical_hand, hand_from_points, place

SHAPES = {
    "B": canonical_hand(),
    "D": canonical_hand(["index"], thumb_out=False),
    "I": canonical_hand(["pinky"], thumb_out=False),
    "V": canonical_hand(["index", "middle"], thumb_out=False),
}


def synthetic_rows(rng, takes_per_label=4, per_take=30):
    rows = []
    for label, canonical in SHAPES.items():
        for take in range(takes_per_label):
            center = rng.uniform([200, 150], [1000, 550])
            size, angle = rng.uniform(60, 160), rng.uniform(-20, 20)
            for _ in range(per_take):
                pts = place(canonical, center + rng.normal(0, 5, 2), size * rng.uniform(0.95, 1.05),
                            angle + rng.normal(0, 3))
                pts = pts + rng.normal(0, 1.5, pts.shape).astype(np.float32)
                rows.append(hand_to_row(label, f"{label}-{take}", "test", hand_from_points(pts)))
    return rows


def test_csv_roundtrip_and_undo(tmp_path, rng):
    path = tmp_path / "data.csv"
    pts = place(SHAPES["B"], (500, 300), 120, 10)
    hand = hand_from_points(pts)
    append_rows(path, [hand_to_row("B", "take-1", "test", hand)] * 3)
    append_rows(path, [hand_to_row("D", "take-2", "test", hand)] * 2)
    ds = load_dataset(path)
    assert len(ds) == 5
    assert np.allclose(ds.points[0], pts, atol=1e-3)
    assert list(ds.labels) == ["B", "B", "B", "D", "D"]
    assert label_counts(path) == {"B": 3, "D": 2}
    assert remove_take(path, "take-1") == 3
    assert label_counts(path) == {"D": 2}


def test_group_split_keeps_takes_together(rng):
    labels = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 20)
    groups = np.array([f"A{i // 10}" for i in range(50)] + [f"B{i // 25}" for i in range(50)]
                      + ["C0"] * 20)
    val, frame_level = train_script.group_split(labels, groups, 0.2, rng)
    for g in np.unique(groups[labels != "C"]):
        in_val = val[groups == g]
        assert in_val.all() or not in_val.any(), f"group {g} was split"
    for c in "AB":
        assert val[labels == c].any() and not val[labels == c].all()
    assert frame_level == ["C"] and val[labels == "C"].sum() == 4


def test_model_save_load_roundtrip(tmp_path):
    model = LandmarkMLP(FEATURE_DIM, 4, (32, 16), 0.1)
    model.set_feature_stats(np.ones(FEATURE_DIM), np.full(FEATURE_DIM, 2.0))
    path = tmp_path / "m.pt"
    save_classifier(path, model, ["A", "B", "C", "D"], (32, 16), 0.1, {"note": "test"})
    loaded, labels, meta = load_classifier(path)
    x = torch.randn(5, FEATURE_DIM)
    model.eval()
    assert torch.allclose(model(x), loaded(x))
    assert labels == ["A", "B", "C", "D"] and meta["note"] == "test"


def test_training_end_to_end(tmp_path, rng):
    data = tmp_path / "train.csv"
    append_rows(data, synthetic_rows(rng))
    test = tmp_path / "test.csv"
    append_rows(test, synthetic_rows(np.random.default_rng(99), takes_per_label=2, per_take=10))
    out = tmp_path / "model.pt"
    args = train_script.parse_args(["--data", str(data), "--test-data", str(test), "--out", str(out),
                                    "--epochs", "40", "--patience", "40", "--no-plots"])
    results = train_script.train(args)
    assert results["validation"]["accuracy"] > 0.95
    assert results["test"]["accuracy"] > 0.95

    clf = LetterClassifier(out)
    assert clf.labels == ["B", "D", "I", "V"]
    for label, canonical in SHAPES.items():
        probs = clf.predict_proba(place(canonical, (640, 360), 110, 5))
        assert clf.labels[int(np.argmax(probs))] == label
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)
    # Mirrored (left) hands work too, thanks to mirror augmentation.
    probs = clf.predict_proba(place(SHAPES["V"], (640, 360), 110, mirror=True))
    assert clf.labels[int(np.argmax(probs))] == "V"


def test_feature_version_mismatch_is_reported(tmp_path):
    model = LandmarkMLP(FEATURE_DIM, 2, (8,), 0.0)
    path = tmp_path / "old.pt"
    torch.save({"state_dict": model.state_dict(), "labels": ["A", "B"], "in_dim": FEATURE_DIM,
                "hidden": [8], "dropout": 0.0, "feature_version": -1, "meta": {}}, path)
    with pytest.raises(ValueError, match="feature version"):
        load_classifier(path)
