# =========================
# problem2.py
# =========================
"""
HW4 - Problem 2: CNN on CIFAR-10 (PyTorch)

You will implement a compact CNN, a deterministic corruption function,
and a comparison routine that trains MLP vs CNN under the same budget.

Autograder imports these functions/classes directly. Keep names/signatures unchanged.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# You may import from problem1 for convenience
from problem1 import set_seed, train_one_epoch, evaluate, MLP


class SmallCNN(nn.Module):
    """
    CNN classifier for CIFAR-10.

    Requirements:
    - At least 3 conv layers
    - BatchNorm in at least 2 places
    - At least one pooling op
    - Global pooling / adaptive pooling before classifier
    - Output logits (B, 10)
    """

    def __init__(self, width: int = 32, dropout_p: float = 0.0):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(3, w, 3, padding=1),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(w, w * 2, 3, padding=1),
            nn.BatchNorm2d(w * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(w * 2, w * 4, 3, padding=1),
            nn.BatchNorm2d(w * 4),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(w * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def apply_corruption(
    x: torch.Tensor,
    kind: str,
    severity: float,
    seed: int,
) -> torch.Tensor:
    """
    Apply a deterministic corruption to images in [0,1].

    Parameters
    ----------
    x : torch.Tensor, shape (B,3,32,32), values in [0,1]
    kind : one of {"gaussian_noise", "channel_drop", "cutout"}
    severity : controls strength/area (interpretation up to you, but must be consistent)
    seed : int for determinism

    Returns
    -------
    x_corr : torch.Tensor, shape (B,3,32,32), values clipped to [0,1]
    """
    g = torch.Generator()
    g.manual_seed(seed)
    x = x.clone()
    if kind == "gaussian_noise":
        noise = torch.randn_like(x, generator=g if x.device.type == "cpu" else None) * severity
        x = x + noise
    elif kind == "channel_drop":
        B = x.shape[0]
        mask = torch.bernoulli(torch.full((B, 3, 1, 1), 1.0 - severity), generator=g)
        x = x * mask.to(x.device)
    elif kind == "cutout":
        B, C, H, W = x.shape
        cut_h = int(H * severity)
        cut_w = int(W * severity)
        if cut_h > 0 and cut_w > 0:
            cy = torch.randint(0, H, (B,), generator=g)
            cx = torch.randint(0, W, (B,), generator=g)
            for i in range(B):
                y1 = max(0, cy[i] - cut_h // 2)
                y2 = min(H, cy[i] + cut_h // 2)
                x1 = max(0, cx[i] - cut_w // 2)
                x2 = min(W, cx[i] + cut_w // 2)
                x[i, :, y1:y2, x1:x2] = 0.0
    else:
        raise ValueError(f"Unknown corruption kind: {kind}")
    return x.clamp(0.0, 1.0)


def compare_mlp_cnn(
    mlp: nn.Module,
    cnn: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    seed: int,
) -> Dict[str, float]:
    """
    Train both models for the same number of epochs and return:
      {"mlp_test_acc": ..., "cnn_test_acc": ..., "delta": ...}
    where delta = cnn_test_acc - mlp_test_acc.

    You must use the same seed for determinism.
    """
    set_seed(seed)
    mlp_opt = torch.optim.SGD(mlp.parameters(), lr=0.1, momentum=0.9)
    cnn_opt = torch.optim.SGD(cnn.parameters(), lr=0.1, momentum=0.9)
    for epoch in range(epochs):
        mlp_tr = train_one_epoch(mlp, train_loader, mlp_opt, device)
        cnn_tr = train_one_epoch(cnn, train_loader, cnn_opt, device)
    mlp_te = evaluate(mlp, test_loader, device)
    cnn_te = evaluate(cnn, test_loader, device)
    return {
        "mlp_test_acc": mlp_te["acc"],
        "cnn_test_acc": cnn_te["acc"],
        "delta": cnn_te["acc"] - mlp_te["acc"],
    }


if __name__ == "__main__":
    # Quick sanity run (not graded)
    from problem1 import get_cifar10_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        set_seed(0)
        train_loader, test_loader = get_cifar10_loaders(batch_size=64, seed=0, limit_train=512, limit_test=256)
        mlp = MLP(hidden_sizes=(512, 256), dropout_p=0.2).to(device)
        cnn = SmallCNN(width=32, dropout_p=0.1).to(device)
        out = compare_mlp_cnn(mlp, cnn, train_loader, test_loader, device, epochs=1, seed=0)
        print(out)
    except NotImplementedError:
        print("Implement TODOs first.")
