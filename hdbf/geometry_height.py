"""Predicted multiscale geometry informs height before its first prediction."""

import torch
from torch import nn
from torch.nn import functional as F


class ChannelNorm2d(nn.Module):
    """Normalize channels independently at each spatial position."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, value):
        value = value.permute(0, 2, 3, 1)
        value = F.layer_norm(
            value, (self.weight.numel(),), self.weight, self.bias, self.eps
        )
        return value.permute(0, 3, 1, 2).contiguous()


def projection(cin, cout, local=False):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 1, bias=False),
        ChannelNorm2d(cout) if local else nn.GroupNorm(8, cout),
        nn.GELU(),
    )


class GeometryHeightFusion(nn.Module):
    def __init__(
        self, height_channels=512, geometry_channels=192, width=64, mode="joint-global"
    ):
        super().__init__()
        if mode not in ("joint-global", "asymmetric-local"):
            raise ValueError(f"Unknown geometry-height conditioning mode: {mode}")
        self.mode = mode
        local = mode == "asymmetric-local"
        if local:
            # Prevent a missing/wrong metadata mode from silently loading the
            # same-shaped old GN affine weights into the new normalization.
            self.register_buffer("local_conditioning", torch.ones((), dtype=torch.long))
        norm = lambda channels: (
            ChannelNorm2d(channels) if local else nn.GroupNorm(8, channels)
        )
        self.height_projection = projection(height_channels, width, local)
        self.geometry_projections = nn.ModuleList(
            projection(geometry_channels, width, local) for _ in range(4)
        )
        self.ade_projection = projection(geometry_channels, width, local)
        self.scale_score = nn.Sequential(
            nn.Conv2d(width * 2 + 4, width // 2, 1),
            nn.GELU(),
            nn.Conv2d(width // 2, 1, 1),
        )
        self.reliability = nn.Sequential(
            nn.Conv2d(width * 2 + 4, width, 3, padding=1),
            norm(width),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )
        self.context = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1, bias=False),
            norm(width),
            nn.GELU(),
        )
        self.output = nn.Conv2d(width, height_channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(
        self, height_feature, geometry_pyramid, cnn_context, semantic_mask, cnn_logits
    ):
        if len(geometry_pyramid) != 4 or cnn_logits.shape[1] != 2:
            raise ValueError(
                "Expected four geometry scales and predicted CNN mask/contour logits"
            )
        if self.mode == "asymmetric-local":
            # Only this new height-conditioning route stops at the reference
            # features. The original tensors still train the geometry decoder.
            geometry_pyramid = [feature.detach() for feature in geometry_pyramid]
            cnn_context = cnn_context.detach()
        # FP32 avoids BF16 bilinear limitations and low-precision gate/softmax
        # arithmetic. Class probabilities are references in both modes, not a
        # route to repurpose binary logits as heights.
        with torch.autocast(device_type=height_feature.device.type, enabled=False):
            size = height_feature.shape[-2:]
            resize = lambda x: F.interpolate(
                x.float(), size, mode="bilinear", align_corners=False
            )
            h = self.height_projection(height_feature.float())
            cnn = resize(cnn_logits.detach()).sigmoid()
            semantic = resize(semantic_mask.detach()).sigmoid()
            priors = torch.cat((semantic, cnn, (semantic - cnn[:, :1]).abs()), 1)
            geometries = [
                resize(layer(feature.float()))
                for layer, feature in zip(self.geometry_projections, geometry_pyramid)
            ]
            scores = torch.cat(
                [self.scale_score(torch.cat((h, g, priors), 1)) for g in geometries], 1
            )
            weights = scores.softmax(1)
            geometry = sum(g * weights[:, i : i + 1] for i, g in enumerate(geometries))
            gate = self.reliability(torch.cat((h, geometry, priors), 1)).sigmoid()
            context = self.context(
                torch.cat(
                    (geometry, resize(self.ade_projection(cnn_context.float()))), 1
                )
            )
            delta = self.output(context * gate)
            adapted = height_feature + delta.to(height_feature.dtype)
            stats = dict(
                height_geometry_scale_weights=weights.detach(),
                height_geometry_gate_mean=gate.detach().mean(),
                height_geometry_relative_residual=delta.detach().square().mean().sqrt()
                / height_feature.detach()
                .float()
                .square()
                .mean()
                .sqrt()
                .clamp_min(1e-6),
            )
        return adapted, stats
