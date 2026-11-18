"""Independent structure/height pyramids with detached structure guidance."""

from torch import nn
import torch
from torch.nn import functional as F


def conv_norm(cin, cout, kernel=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
        nn.GroupNorm(8, cout),
        nn.GELU(),
    )


def resize_like(value, reference):
    return F.interpolate(
        value, size=reference.shape[-2:], mode="bilinear", align_corners=False
    )


class PyramidReadout(nn.Module):
    def __init__(self, width, output_channels):
        super().__init__()
        self.fuse = nn.Sequential(
            conv_norm(width * 4, output_channels), nn.Dropout2d(0.1)
        )

    def forward(self, pyramid):
        fine = pyramid[0]
        return self.fuse(
            torch.cat(
                [
                    resize_like(x, fine) if x.shape[-2:] != fine.shape[-2:] else x
                    for x in pyramid[::-1]
                ],
                dim=1,
            )
        )


class StructurePyramid(nn.Module):
    def __init__(self, channels, width, output_channels):
        super().__init__()
        self.lateral = nn.ModuleList([conv_norm(c, width) for c in channels])
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    conv_norm(width * (2 if i < 3 else 1), width),
                    conv_norm(width, width, 3),
                )
                for i in range(4)
            ]
        )
        self.readout = PyramidReadout(width, output_channels)

    def forward(self, features):
        lateral = [layer(x) for layer, x in zip(self.lateral, features)]
        pyramid = [None] * 4
        for i in range(3, -1, -1):
            inputs = (
                lateral[i]
                if i == 3
                else torch.cat((lateral[i], resize_like(pyramid[i + 1], lateral[i])), 1)
            )
            pyramid[i] = self.refine[i](inputs)
        return self.readout(pyramid), pyramid


class GuidedHeightPyramid(nn.Module):
    def __init__(self, channels, width, output_channels):
        super().__init__()
        self.lateral = nn.ModuleList([conv_norm(c, width) for c in channels])
        self.guidance = nn.ModuleList([conv_norm(width, width) for _ in channels])
        self.gates = nn.ModuleList([nn.Conv2d(width * 2, width, 1) for _ in channels])
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    conv_norm(width * (3 if i < 3 else 2), width),
                    conv_norm(width, width, 3),
                )
                for i in range(4)
            ]
        )
        self.readout = PyramidReadout(width, output_channels)

    def forward(self, features, structure):
        lateral = [layer(x) for layer, x in zip(self.lateral, features)]
        pyramid = [None] * 4
        for i in range(3, -1, -1):
            # Height trains the guidance adapters, not the source structure pyramid.
            guide = self.guidance[i](structure[i].detach())
            gate = self.gates[i](torch.cat((lateral[i], guide), 1)).sigmoid()
            inputs = [lateral[i], gate * guide]
            if i < 3:
                inputs.append(resize_like(pyramid[i + 1], lateral[i]))
            pyramid[i] = self.refine[i](torch.cat(inputs, 1))
        return self.readout(pyramid)


class TaskSplitPyramidDecoder(nn.Module):
    def __init__(self, channels, output_channels=512, width=192):
        super().__init__()
        if (
            len(channels) != 4
            or width < 8
            or width % 8
            or output_channels < 8
            or output_channels % 8
        ):
            raise ValueError(
                "Four scales and positive channels divisible by eight are required"
            )
        self.channels = tuple(channels)
        self.structure = StructurePyramid(channels, width, output_channels)
        self.height = GuidedHeightPyramid(channels, width, output_channels)

    def forward(self, features):
        height_features = features["height"] if isinstance(features, dict) else features
        features = features["structure"] if isinstance(features, dict) else features
        if len(features) != 4:
            raise ValueError("Task-split decoder requires four feature scales")
        for i, (x, channels) in enumerate(zip(features, self.channels)):
            if (
                x.ndim != 4
                or x.shape[1] != channels
                or x.shape[0] != features[0].shape[0]
            ):
                raise ValueError(f"Invalid feature shape at stage {i}")
        structure, pyramid = self.structure(features)
        if len(height_features) != 4 or any(
            a.shape != b.shape for a, b in zip(features, height_features)
        ):
            raise ValueError("Structure and height pyramids must have matching shapes")
        height = self.height(height_features, pyramid)
        return structure, height
