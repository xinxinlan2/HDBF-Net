"""Continuous contextual height with bounded local detail; no GT in forward."""

from pathlib import Path as _HDBFPath
import sys as _hdbf_sys

_HDBF_ROOT = _HDBFPath(__file__).resolve().parents[1]
if str(_HDBF_ROOT) not in _hdbf_sys.path:
    _hdbf_sys.path.insert(0, str(_HDBF_ROOT))
import torch
from torch import nn
import torch.nn.functional as F


def box_average(value, kernel):
    value = F.avg_pool2d(value, (1, kernel), stride=1, padding=(0, kernel // 2))
    return F.avg_pool2d(value, (kernel, 1), stride=1, padding=(kernel // 2, 0))


class ContextDetailHeightHead(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        detail_limit=3.0,
        building_affinity=False,
    ):
        super().__init__()
        if hidden_channels < 8 or hidden_channels % 8 or detail_limit <= 0:
            raise ValueError(
                "Hidden channels must be divisible by 8; detail limit must be positive"
            )
        self.detail_limit = float(detail_limit)
        self.building_affinity = bool(building_affinity)
        if self.building_affinity:
            from hdbf.dfc_building_affinity import BuildingAffinityContext

            self.affinity = BuildingAffinityContext(hidden_channels)
        self.direct = nn.Conv2d(in_channels, 1, 1)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(4 * hidden_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        self.context_out = nn.Conv2d(hidden_channels, 1, 1)
        self.detail = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        self.reset_output()

    def reset_output(self):
        nn.init.normal_(self.direct.weight, std=0.01)
        nn.init.constant_(self.direct.bias, 1.0)
        for layer in [self.context_out, self.detail[-1]]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, feature, footprint_logits):
        if footprint_logits.shape != (feature.shape[0], 1, *feature.shape[-2:]):
            raise ValueError(
                "Footprint guidance and height features must share a spatial grid"
            )
        local = self.project(feature)
        weight = 0.1 + 0.9 * footprint_logits.detach().float().sigmoid()
        contexts = [local.float()]
        if self.building_affinity:
            (neighbours, affinity_logits) = self.affinity(local, weight)
            contexts.extend(neighbours)
        else:
            for kernel in (5, 17, 33):
                denominator = box_average(weight, kernel).clamp_min(1e-06)
                contexts.append(
                    box_average(local.float() * weight, kernel) / denominator
                )
        context = self.context(torch.cat(contexts, dim=1).to(local.dtype))
        base = F.softplus(
            self.direct(feature).float() + self.context_out(context).float()
        )
        highpass = local.float() - box_average(local.float(), 5)
        detail = self.detail_limit * torch.tanh(
            self.detail(highpass.to(local.dtype)).float()
        )
        result = (F.relu(base + detail), base, detail)
        return (*result, affinity_logits) if self.building_affinity else result
