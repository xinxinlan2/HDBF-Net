"""Task-specific RGB/SAR fusion with shared local correspondence, no GT inputs."""

import torch
from torch import nn
from torch.nn import functional as F


def project(cin, cout, kernel=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
        nn.GroupNorm(8, cout),
        nn.GELU(),
    )


class TaskRouter(nn.Module):
    def __init__(self, channels, context=False):
        super().__init__()
        self.context = context
        self.rgb = project(channels, channels)
        self.sar = project(channels, channels)
        self.selector = nn.Sequential(
            project(channels * 3, channels), nn.Conv2d(channels, channels * 2, 1)
        )
        self.interaction = project(channels * 2, channels, 3)
        self.output = project(channels * 2, channels)

    def forward(self, rgb, sar):
        r, s = self.rgb(rgb), self.sar(sar)
        evidence = torch.cat((r, s, (r - s).abs()), 1)
        if self.context:
            evidence = F.avg_pool2d(
                evidence, 5, stride=1, padding=2, count_include_pad=False
            )
        b, c, h, w = r.shape
        weights = (
            self.selector(evidence)
            .reshape(b, 2, c, h, w)
            .float()
            .softmax(1)
            .to(r.dtype)
        )
        selected = weights[:, 0] * r + weights[:, 1] * s
        interaction = self.interaction(torch.cat((r * s, r - s), 1))
        return self.output(torch.cat((selected, interaction), 1))


class TaskAwareFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        key_channels = max(16, channels // 8)
        self.query = nn.Conv2d(channels, key_channels, 1, bias=False)
        self.key = nn.Conv2d(channels, key_channels, 1, bias=False)
        self.structure = TaskRouter(channels)
        self.height = TaskRouter(channels, context=True)

    def align_sar(self, rgb, sar):
        # Nine valid candidates; FP32 cosine softmax avoids AMP overflow.
        q = F.normalize(self.query(rgb).float(), dim=1, eps=1e-6)
        k = F.normalize(self.key(sar).float(), dim=1, eps=1e-6)
        h, w = rgb.shape[-2:]
        padded_key = F.pad(k, (1, 1, 1, 1))
        valid = F.pad(torch.ones_like(k[:, :1]), (1, 1, 1, 1))
        logits = []
        for y in range(3):
            for x in range(3):
                score = (q * padded_key[:, :, y : y + h, x : x + w]).sum(
                    1, keepdim=True
                ) * 4.0
                logits.append(
                    score.masked_fill(valid[:, :, y : y + h, x : x + w] == 0, -1e4)
                )
        weights = torch.cat(logits, 1).softmax(1).to(sar.dtype)
        padded_sar = F.pad(sar, (1, 1, 1, 1))
        aligned = torch.zeros_like(sar)
        for y in range(3):
            for x in range(3):
                aligned = (
                    aligned
                    + weights[:, y * 3 + x : y * 3 + x + 1]
                    * padded_sar[:, :, y : y + h, x : x + w]
                )
        return aligned

    def forward(self, rgb, sar):
        if rgb.shape != sar.shape:
            raise ValueError(
                "Task fusion requires matched RGB/SAR feature grids and channels"
            )
        aligned = self.align_sar(rgb, sar)
        structure = (
            self.structure(rgb.detach(), aligned.detach())
            if getattr(self, "isolate_structure", False)
            else self.structure(rgb, aligned)
        )
        return structure, self.height(rgb, aligned)
