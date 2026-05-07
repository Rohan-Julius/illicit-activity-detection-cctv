from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class DCENet(nn.Module):
    """
    Zero-DCE/Zero-DCE++ style enhancement network.

    This implementation matches the common open-source DCENet topology:
    a small conv stack producing 24 channels (8 curves * RGB), applied iteratively.
    """

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        # 8 iterative curve adjustments
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (x * x - x)
        x = x + r2 * (x * x - x)
        x = x + r3 * (x * x - x)
        x = x + r4 * (x * x - x)
        x = x + r5 * (x * x - x)
        x = x + r6 * (x * x - x)
        x = x + r7 * (x * x - x)
        x = x + r8 * (x * x - x)
        return torch.clamp(x, 0.0, 1.0)


@dataclass(frozen=True)
class ZeroDCEConfig:
    weights_path: str
    device: str = "cpu"


def load_zerodce(cfg: ZeroDCEConfig) -> DCENet:
    model = DCENet()
    state = torch.load(cfg.weights_path, map_location=cfg.device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    # Some trainers prefix keys with "module."
    if isinstance(state, dict):
        cleaned = {}
        for k, v in state.items():
            ck = k.replace("module.", "")
            cleaned[ck] = v
        state = cleaned
    model.load_state_dict(state, strict=False)
    model.to(cfg.device)
    model.eval()
    return model

