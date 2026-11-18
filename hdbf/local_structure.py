"""Dense semantic reconstruction with task-specific local affinity refinement.

No GT enters forward. Neighbor supervision is derived from the existing binary
footprint/contour labels, not new instance boundaries. PyTorch PixelShuffle and
Unfold provide upsampling and neighborhood extraction.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.GELU()
    )


def learned_up(cin, cout):
    projection = nn.Conv2d(cin, cout * 4, 1)
    # Identical subpixel kernels avoid injecting an initial checkerboard pattern.
    with torch.no_grad():
        kernel = torch.empty(cout, cin, 1, 1)
        nn.init.kaiming_normal_(kernel)
        projection.weight.copy_(kernel.repeat_interleave(4, dim=0))
        projection.bias.zero_()
    return nn.Sequential(projection, nn.PixelShuffle(2))


def up2(value, size):
    value = F.interpolate(
        value.float(), scale_factor=2, mode="bilinear", align_corners=False
    )
    if value.shape[-2] < size[0] or value.shape[-1] < size[1]:
        raise ValueError("Native pyramid must cover the destination grid")
    return value[..., : size[0], : size[1]]


def crop(value, size):
    return value[..., : size[0], : size[1]]


def neighborhoods(value):
    b, c, h, w = value.shape
    return F.unfold(value.float(), 3, padding=1).reshape(b, c, 9, h, w)


def height_features(height, native):
    # Resize height to its original tile before pooling: output coordinates stay
    # aligned with the RGB grid even for odd image dimensions.
    value = F.interpolate(
        height.detach().float(), native, mode="bilinear", align_corners=False
    )
    value = torch.log1p(value.clamp_min(0)) / math.log(101)
    relief = value - F.avg_pool2d(F.pad(value, (2, 2, 2, 2), mode="replicate"), 5, 1)
    dx = F.pad(value[..., 1:] - value[..., :-1], (0, 1, 0, 0))
    dy = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
    return torch.cat((value, relief, dx, dy), 1)


class DilatedResidual(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, value):
        return value + self.context(value)


class LocalStructureDecoder(nn.Module):
    def __init__(self, width=192):
        super().__init__()
        self.register_buffer("training_epoch", torch.zeros((), dtype=torch.long))
        self.mask_seed = nn.Sequential(block(width, 64), nn.Conv2d(64, 1, 1))
        self.contour_seed = nn.Sequential(
            block(width + 32 + 2, 64), nn.Conv2d(64, 1, 1)
        )
        self.semantic_up = learned_up(width, 64)
        self.edge_up = learned_up(32, 32)
        self.half_context = nn.Sequential(
            block(64 + 32 + 16 + 2 + 4, 64),
            DilatedResidual(64, 1),
            DilatedResidual(64, 2),
        )
        self.half_mask = nn.Conv2d(64, 1, 1)
        self.half_contour = nn.Conv2d(64, 1, 1)
        self.native_up = learned_up(64, 32)
        self.native_edge_up = learned_up(32, 16)
        self.native_context = nn.Sequential(
            block(32 + 16 + 16 + 2 + 4, 32), DilatedResidual(32, 1)
        )
        self.native_mask = nn.Sequential(block(32, 16), nn.Conv2d(16, 1, 1))
        self.native_contour = nn.Sequential(block(32, 16), nn.Conv2d(16, 1, 1))
        self.height_gate = nn.Conv2d(16 + 4, 4, 1)
        nn.init.zeros_(self.height_gate.weight)
        nn.init.constant_(self.height_gate.bias, -2.0)
        self.affinity = nn.Conv2d(32 + 2, 18, 3, padding=1)
        self.propagation_gate = nn.Conv2d(32 + 2, 2, 1)
        nn.init.zeros_(self.affinity.weight)
        nn.init.zeros_(self.affinity.bias)
        with torch.no_grad():
            self.affinity.bias.view(2, 9)[:, 4] = 4.0
        nn.init.zeros_(self.propagation_gate.weight)
        nn.init.constant_(self.propagation_gate.bias, -2.0)
        for head in (
            self.half_mask,
            self.half_contour,
            self.native_mask[-1],
            self.native_contour[-1],
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def set_epoch(self, epoch):
        self.training_epoch.fill_(int(epoch))

    def guidance_strength(self):
        return ((self.training_epoch.float() - 20) / 20).clamp(0, 1)

    def forward(self, joint, shallow, detail, seeds, height):
        native = detail.shape[-2:]
        half_size = tuple((v + 1) // 2 for v in native)
        coarse = torch.cat(
            (
                seeds[:, :1] + self.mask_seed(joint),
                seeds[:, 1:] + self.contour_seed(torch.cat((joint, shallow, seeds), 1)),
            ),
            1,
        ).float()
        description = height_features(height, native)
        gate = (
            self.height_gate(torch.cat((detail.float(), description), 1))
            .float()
            .sigmoid()
        )
        description = description * gate * self.guidance_strength()
        edge_half = crop(self.edge_up(shallow), half_size)
        half_seed = up2(coarse, half_size)
        half_feature = self.half_context(
            torch.cat(
                (
                    crop(self.semantic_up(joint), half_size),
                    edge_half,
                    F.avg_pool2d(detail.float(), 2, ceil_mode=True),
                    half_seed,
                    F.avg_pool2d(description, 2, ceil_mode=True),
                ),
                1,
            )
        )
        half_logits = (
            half_seed
            + torch.cat(
                (self.half_mask(half_feature), self.half_contour(half_feature)), 1
            ).float()
        )
        native_seed = up2(half_logits, native)
        feature = self.native_context(
            torch.cat(
                (
                    crop(self.native_up(half_feature), native),
                    crop(self.native_edge_up(edge_half), native),
                    detail,
                    native_seed,
                    description,
                ),
                1,
            )
        )
        logits = (
            native_seed
            + torch.cat(
                (self.native_mask(feature), self.native_contour(feature)), 1
            ).float()
        )
        preprop = logits
        # Every pixel is updated. The two tasks have separate neighbor weights;
        # contour is never replaced with a morphological edge of the mask.
        with torch.autocast(device_type=joint.device.type, enabled=False):
            state = torch.cat((feature.float(), logits.sigmoid()), 1)
            affinity = self.affinity(state).reshape(logits.shape[0], 2, 9, *native)
            valid = neighborhoods(torch.ones_like(logits[:, :1])) > 0.5
            weights = affinity.masked_fill(~valid, -torch.inf).softmax(2)
            strength = self.propagation_gate(state).sigmoid()
            for _ in range(2):
                neighbors = neighborhoods(logits)
                propagated = (weights * neighbors).sum(2)
                logits = logits + strength * (propagated - logits)
        return {
            "mask": logits[:, :1],
            "contour": logits[:, 1:],
            "local_coarse_logits": coarse,
            "local_half_logits": half_logits,
            "local_preprop_logits": preprop,
            "local_affinity_logits": affinity,
        }


def _bce_dice(logits, target):
    probability = logits.sigmoid()
    axes = (-2, -1)
    dice = 1 - (2 * (probability * target).sum(axes) + 1e-6) / (
        probability.sum(axes) + target.sum(axes) + 1e-6
    )
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(
        axes
    )
    return 0.5 * (bce + dice)


def local_structure_loss(output, mask, contour, mask_weight=1.0, contour_weight=0.4):
    """Deep semantic supervision plus local same-label neighbor distributions."""
    with torch.autocast(device_type=mask.device.type, enabled=False):
        target = torch.cat((mask, contour), 1).float()
        task_weights = target.new_tensor([mask_weight, contour_weight])
        deep = []
        for key, stride in (("local_coarse_logits", 4), ("local_half_logits", 2)):
            logits = output[key].float()
            truth = F.avg_pool2d(
                target, stride, stride, ceil_mode=True, count_include_pad=False
            )
            if truth.shape != logits.shape:
                raise ValueError(
                    "Auxiliary targets must share the decoder's native stride/crop grid"
                )
            deep.append((_bce_dice(logits, truth) * task_weights).sum(1).mean())
        affinity = output["local_affinity_logits"].float()
        neighbor_target = neighborhoods(target)
        valid = neighborhoods(torch.ones_like(target[:, :1])) > 0.5
        same = (neighbor_target == target.unsqueeze(2)) & valid
        distribution = same.float() / same.sum(2, keepdim=True).clamp_min(1)
        log_prob = affinity.masked_fill(~valid, -1e4).log_softmax(2)
        error = -(distribution * log_prob).sum(2)
        boundary = ((neighbor_target != target.unsqueeze(2)) & valid).any(2)
        # Balance boundary/interior pixels per task, including empty/full tiles.
        group_losses = []
        for group in (boundary, ~boundary):
            count = group.sum((-2, -1))
            value = (error * group).sum((-2, -1)) / count.clamp_min(1)
            group_losses.append((value, count > 0))
        affinity_loss = sum(v * present for v, present in group_losses) / sum(
            present.float() for _, present in group_losses
        ).clamp_min(1)
        affinity_loss = (affinity_loss * task_weights).sum(1).mean()
        return torch.stack(deep).mean() + 0.25 * affinity_loss
