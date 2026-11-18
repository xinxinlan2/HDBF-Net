"""Continuous height prediction through supervised range probabilities and experts.

Fixed log1p-spaced ranges do not depend on validation/test labels. The last
expert is open-ended; 128 m is a range boundary, never an output clipping cap.
"""

from pathlib import Path as _HDBFPath
import sys as _hdbf_sys

_HDBF_ROOT = _HDBFPath(__file__).resolve().parents[1]
if str(_HDBF_ROOT) not in _hdbf_sys.path:
    _hdbf_sys.path.insert(0, str(_HDBF_ROOT))

import math

import torch
from torch import nn
import torch.nn.functional as F


class DistributionHeightHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        edges = torch.expm1(torch.linspace(0.0, math.log1p(128.0), 17))
        edges[0], edges[-1] = 0.0, 128.0
        self.register_buffer("edges", edges)
        self.context = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(hidden_channels, 17, 1)
        self.experts = nn.Conv2d(hidden_channels, 17, 1)
        self.reset_output()

    def reset_output(self):
        for layer in (self.classifier, self.experts):
            nn.init.normal_(layer.weight, std=0.001)
            nn.init.zeros_(layer.bias)

    def forward(self, feature):
        context = self.context(feature)
        logits = self.classifier(context).float()
        raw = self.experts(context).float()
        left = self.edges[:-1].view(1, -1, 1, 1)
        width = self.edges.diff().view(1, -1, 1, 1)
        bounded = left + width * torch.sigmoid(raw[:, :-1])
        tail = self.edges[-1] + 64.0 * F.softplus(raw[:, -1:])
        candidates = torch.cat((bounded, tail), dim=1)
        height = (logits.softmax(dim=1) * candidates).sum(dim=1, keepdim=True)
        return height, logits, candidates


def distribution_losses(output, target, valid):
    logits = output["height_range_logits"].float()
    candidates = output["height_range_candidates"].float()
    edges = output["height_range_edges"].float()
    if target.shape != valid.shape or target.ndim != 4 or target.shape[1] != 1:
        raise ValueError("Target and validity must share Bx1xHxW shape")
    if logits.shape != candidates.shape or logits.shape[1] != edges.numel():
        raise ValueError("Height range output channels do not match edges")
    if not valid.any():
        zero = logits.reshape(-1)[:0].sum() + candidates.reshape(-1)[:0].sum()
        return zero, zero, zero.detach()
    selection = valid[:, 0]
    gt = target[:, 0][selection].float()
    if not torch.isfinite(gt).all() or (gt < 0).any():
        raise ValueError("Range supervision requires finite nonnegative valid heights")
    # Upsample predictions, not labels: mixed roofs retain their original heights.
    logits = F.interpolate(
        logits, target.shape[-2:], mode="bilinear", align_corners=False
    )
    candidates = F.interpolate(
        candidates, target.shape[-2:], mode="bilinear", align_corners=False
    )
    scores = logits.permute(0, 2, 3, 1)[selection]
    values = candidates.permute(0, 2, 3, 1)[selection]
    if not torch.isfinite(scores).all() or not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite distribution-head outputs at valid pixels")
    labels = torch.bucketize(gt.contiguous(), edges[1:], right=True)
    classification = F.cross_entropy(scores, labels)
    selected_value = values.gather(1, labels[:, None])[:, 0]
    regression = F.smooth_l1_loss(selected_value, gt, beta=1.0)
    accuracy = (scores.detach().argmax(1) == labels).float().mean()
    return classification, regression, accuracy


@torch.no_grad()
def range_confusion(output, target, valid):
    edges = output["height_range_edges"].float()
    count = edges.numel()
    scores = F.interpolate(
        output["height_range_logits"].float(),
        target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    selected = valid[:, 0]
    gt = torch.bucketize(
        target[:, 0][selected].float().contiguous(), edges[1:], right=True
    )
    pred = scores.argmax(1)[selected]
    return torch.bincount(gt * count + pred, minlength=count * count).reshape(
        count, count
    )
