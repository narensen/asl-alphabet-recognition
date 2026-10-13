"""PyTorch classifier over landmark features, with save/load and inference helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from .features import FEATURE_DIM, FEATURE_VERSION, features, normalize


class LandmarkMLP(nn.Module):
    """Small MLP: standardise features -> [Linear, LayerNorm, GELU, Dropout] x k -> logits.

    Feature mean/std live in buffers, so they are saved with the weights and
    inference cannot forget to apply them.
    """

    def __init__(self, in_dim: int = FEATURE_DIM, num_classes: int = 24,
                 hidden: Sequence[int] = (256, 128), dropout: float = 0.3):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))
        layers: List[nn.Module] = []
        width = in_dim
        for h in hidden:
            layers += [nn.Linear(width, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, num_classes))
        self.net = nn.Sequential(*layers)

    def set_feature_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.as_tensor(np.maximum(std, 1e-4), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.feat_mean) / self.feat_std)


def save_classifier(path: Union[str, Path], model: LandmarkMLP, labels: Sequence[str],
                    hidden: Sequence[int], dropout: float, meta: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "labels": list(labels),
        "in_dim": FEATURE_DIM,
        "hidden": list(hidden),
        "dropout": float(dropout),
        "feature_version": FEATURE_VERSION,
        "meta": meta,
    }, path)


def load_classifier(path: Union[str, Path],
                    device: str = "cpu") -> Tuple[LandmarkMLP, List[str], Dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Collect data and run train.py first.")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only argument
        ckpt = torch.load(path, map_location=device)
    if ckpt.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"{path} was trained with feature version {ckpt.get('feature_version')}, "
            f"but this code uses version {FEATURE_VERSION}. Re-run train.py.")
    model = LandmarkMLP(ckpt["in_dim"], len(ckpt["labels"]), ckpt["hidden"], ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, list(ckpt["labels"]), dict(ckpt.get("meta", {}))


class LetterClassifier:
    """Pixel landmarks of one hand -> class probabilities."""

    def __init__(self, path: Union[str, Path], device: str = "cpu"):
        self.model, self.labels, self.meta = load_classifier(path, device)
        self.device = device

    @torch.no_grad()
    def predict_proba(self, points: np.ndarray) -> np.ndarray:
        """points: (21, 3) or (N, 21, 3) pixel landmarks. Returns (C,) or (N, C)."""
        pts = np.asarray(points, dtype=np.float32)
        single = pts.ndim == 2
        feats = features(normalize(pts[None] if single else pts))
        logits = self.model(torch.from_numpy(feats).to(self.device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs[0] if single else probs
