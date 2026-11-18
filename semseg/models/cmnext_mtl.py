import math
import torch
from torch import nn
import torch.nn.functional as F
from semseg.models.backbones.cmnext import CMNeXt as CMNeXtBackbone


class MLPProjector(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.proj(x.flatten(2).transpose(1, 2))


class SegFormerFeatureDecoder(nn.Module):

    def __init__(self, channels, embed_dim=512):
        super().__init__()
        self.projections = nn.ModuleList(
            [MLPProjector(channel, embed_dim) for channel in channels]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, features):
        (batch_size, _, height, width) = features[0].shape
        projected = []
        for projector, feature in zip(self.projections, features):
            feature = (
                projector(feature)
                .transpose(1, 2)
                .reshape(batch_size, -1, feature.shape[-2], feature.shape[-1])
            )
            if feature.shape[-2:] != (height, width):
                feature = F.interpolate(
                    feature, size=(height, width), mode="bilinear", align_corners=False
                )
            projected.append(feature)
        return self.fuse(torch.cat(projected[::-1], dim=1))


class RoleDecoupledStructureFusion(nn.Module):
    """Select structural RGB-SAR residuals without exposing height to raw SAR."""

    def __init__(self, channels, initial_strength=0.1):
        super().__init__()
        self.residual = nn.Sequential(
            ConvBNAct(channels, channels, groups=channels),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.spatial_gate = nn.Sequential(
            ConvBNAct(channels * 3, channels // 2, kernel_size=1),
            nn.Conv2d(channels // 2, 1, 1),
        )
        hidden = max(channels // 16, 16)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        initial_strength = min(max(float(initial_strength), 0.0001), 1.0 - 0.0001)
        self.strength_logit = nn.Parameter(
            torch.tensor(math.log(initial_strength / (1.0 - initial_strength)))
        )

    def reset_identity(self):
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.zeros_(self.spatial_gate[-1].bias)
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)

    def forward(self, rgb_feature, fused_feature):
        difference = fused_feature - rgb_feature
        spatial = torch.sigmoid(
            self.spatial_gate(
                torch.cat([rgb_feature, fused_feature, difference.abs()], dim=1)
            )
        )
        channel = torch.sigmoid(
            self.channel_gate(torch.cat([rgb_feature, fused_feature], dim=1))
        )
        strength = torch.sigmoid(self.strength_logit)
        return rgb_feature + strength * spatial * channel * self.residual(difference)


class PredictionHead(nn.Module):

    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, x):
        return self.layers(x)


class TaskFeatureAdapter(nn.Module):
    """Create a compact task-specific feature before cross-task interaction."""

    def __init__(self, in_channels, task_channels):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNAct(in_channels, task_channels, kernel_size=1),
            ConvBNAct(task_channels, task_channels, groups=task_channels),
            ConvBNAct(task_channels, task_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.layers(x)


class CrossTaskResidual(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNAct(channels * 3, channels, kernel_size=1),
            ConvBNAct(channels, channels, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )

    def reset_identity(self):
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, joint):
        return self.layers(joint)


class TriTaskCollaborativeInteraction(nn.Module):
    """Exchange task features and predict gated residual output corrections."""

    def __init__(
        self, channels, prediction_residual=True, directional_gradient_isolation=False
    ):
        super().__init__()
        self.prediction_residual = prediction_residual
        self.directional_gradient_isolation = bool(directional_gradient_isolation)
        self.mask_update = CrossTaskResidual(channels)
        self.contour_update = CrossTaskResidual(channels)
        self.height_update = CrossTaskResidual(channels)
        self.strength_logits = nn.Parameter(torch.tensor([-1.386, -1.386, -2.944]))
        if self.prediction_residual:
            self.mask_correction = nn.Conv2d(channels, 1, 1)
            self.contour_correction = nn.Conv2d(channels, 1, 1)
            self.height_correction = nn.Conv2d(channels, 1, 1)
            self.correction_logits = nn.Parameter(
                torch.tensor([-1.386, -1.386, -2.944])
            )
        self.reset_identity()

    def reset_identity(self):
        self.mask_update.reset_identity()
        self.contour_update.reset_identity()
        self.height_update.reset_identity()
        if self.prediction_residual:
            for head in (
                self.mask_correction,
                self.contour_correction,
                self.height_correction,
            ):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, mask_feature, contour_feature, height_feature):
        if self.directional_gradient_isolation:
            mask_joint = torch.cat(
                [mask_feature, contour_feature.detach(), height_feature.detach()], dim=1
            )
            contour_joint = torch.cat(
                [mask_feature.detach(), contour_feature, height_feature.detach()], dim=1
            )
            height_joint = torch.cat(
                [mask_feature.detach(), contour_feature.detach(), height_feature], dim=1
            )
        else:
            joint = torch.cat([mask_feature, contour_feature, height_feature], dim=1)
            mask_joint = contour_joint = height_joint = joint
        strengths = torch.sigmoid(self.strength_logits)
        mask_feature = mask_feature + strengths[0] * torch.tanh(
            self.mask_update(mask_joint)
        )
        contour_feature = contour_feature + strengths[1] * torch.tanh(
            self.contour_update(contour_joint)
        )
        height_feature = height_feature + strengths[2] * torch.tanh(
            self.height_update(height_joint)
        )
        return (mask_feature, contour_feature, height_feature)

    def prediction_corrections(self, mask_feature, contour_feature, height_feature):
        if not self.prediction_residual:
            raise RuntimeError("Prediction residuals are disabled for legacy TTCI.")
        strengths = torch.sigmoid(self.correction_logits)
        mask_delta = strengths[0] * self.mask_correction(mask_feature)
        contour_delta = strengths[1] * self.contour_correction(contour_feature)
        height_delta = (
            5.0 * strengths[2] * torch.tanh(self.height_correction(height_feature))
        )
        return (mask_delta, contour_delta, height_delta)


class ConvBNAct(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class RoofContextResidualRefiner(nn.Module):
    """Correct height with bounded multi-scale RGB roof-context residuals."""

    def __init__(self, decoder_dim, channels=64, max_residual=3.0):
        super().__init__()
        self.max_residual = float(max_residual)
        self.input_projection = ConvBNAct(decoder_dim + 1, channels, kernel_size=1)
        self.context_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.fuse = nn.Sequential(
            ConvBNAct(channels * 3, channels, kernel_size=1),
            ConvBNAct(channels, channels, groups=channels),
            nn.Conv2d(channels, 1, 1),
        )

    def reset_identity(self):
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, decoded, height, mask_logits, contour_logits):
        normalized_height = torch.log1p(height.clamp_min(0.0)) / math.log(201.0)
        feature = self.input_projection(torch.cat([decoded, normalized_height], dim=1))
        context = torch.cat(
            [branch(feature) for branch in self.context_branches], dim=1
        )
        residual = self.max_residual * torch.tanh(self.fuse(context))
        confidence = torch.sigmoid(mask_logits) * (
            1.0 - 0.5 * torch.sigmoid(contour_logits)
        )
        return F.relu(height + confidence * residual)


class RGBHeightDetailResidualRefiner(nn.Module):
    """Refine height at half resolution with an isolated RGB detail branch."""

    def __init__(
        self, decoder_dim, channels=48, max_residual=4.0, initial_strength=0.02
    ):
        super().__init__()
        channels = int(channels)
        if channels < 8:
            raise ValueError("RGB height-detail channels must be at least 8")
        self.max_residual = float(max_residual)
        self.rgb_stem = nn.Sequential(
            ConvBNAct(3, channels, stride=2),
            ConvBNAct(channels, channels, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1),
        )
        self.semantic_projection = ConvBNAct(decoder_dim, channels, kernel_size=1)
        self.fusion = nn.Sequential(
            ConvBNAct(channels * 2 + 1, channels, kernel_size=1),
            ConvBNAct(channels, channels, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1),
        )
        self.residual_head = nn.Conv2d(channels, 1, 3, padding=1)
        self.gate_head = nn.Conv2d(channels, 1, 1)
        initial_strength = min(max(float(initial_strength), 0.0001), 1.0 - 0.0001)
        self.initial_gate_logit = math.log(initial_strength / (1.0 - initial_strength))

    def reset_identity(self):
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, self.initial_gate_logit)

    def forward(self, rgb, decoded, height):
        rgb_detail = self.rgb_stem(rgb.detach())
        semantic = self.semantic_projection(decoded.detach())
        semantic = F.interpolate(
            semantic, size=rgb_detail.shape[-2:], mode="bilinear", align_corners=False
        )
        height_context = F.interpolate(
            height.detach(),
            size=rgb_detail.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        normalized_height = torch.log1p(height_context.clamp_min(0.0)) / math.log(201.0)
        feature = self.fusion(
            torch.cat([rgb_detail, semantic, normalized_height], dim=1)
        )
        residual = self.max_residual * torch.tanh(self.residual_head(feature))
        gate = torch.sigmoid(self.gate_head(feature))
        correction = F.interpolate(
            gate * residual, size=rgb.shape[-2:], mode="bilinear", align_corners=False
        )
        base_height = F.interpolate(
            height, size=rgb.shape[-2:], mode="bilinear", align_corners=False
        )
        return (F.relu(base_height + correction), gate)


class HighResolutionBoundaryRefiner(nn.Module):
    """Fuse half-resolution RGB-SAR details with quarter-resolution semantics."""

    def __init__(
        self,
        decoder_dim,
        detail_channels=48,
        refine_channels=64,
        sar_in_channels=1,
        rgb_anchored=False,
        sar_initial_strength=0.02,
        rgb_only=False,
    ):
        super().__init__()
        self.rgb_anchored = bool(rgb_anchored)
        self.rgb_only = bool(rgb_only)
        rgb_channels = (
            detail_channels
            if self.rgb_anchored or self.rgb_only
            else detail_channels * 2 // 3
        )
        sar_channels = (
            detail_channels if self.rgb_anchored else detail_channels - rgb_channels
        )
        self.rgb_stem = nn.Sequential(
            ConvBNAct(3, rgb_channels, stride=2),
            ConvBNAct(rgb_channels, rgb_channels, groups=rgb_channels),
        )
        self.sar_stem = (
            None
            if self.rgb_only
            else nn.Sequential(
                ConvBNAct(sar_in_channels, sar_channels, stride=2),
                ConvBNAct(sar_channels, sar_channels, groups=sar_channels),
            )
        )
        if self.rgb_only:
            self.detail_fuse = ConvBNAct(
                detail_channels, detail_channels, kernel_size=1
            )
        elif self.rgb_anchored:
            self.sar_reliability = nn.Conv2d(rgb_channels + sar_channels, 1, 1)
            sar_initial_strength = min(
                max(float(sar_initial_strength), 0.0001), 1.0 - 0.0001
            )
            self.sar_strength_logit = nn.Parameter(
                torch.tensor(
                    math.log(sar_initial_strength / (1.0 - sar_initial_strength))
                )
            )
            self.detail_fuse = nn.Identity()
        else:
            self.detail_fuse = ConvBNAct(
                detail_channels, detail_channels, kernel_size=1
            )
        self.semantic_proj = ConvBNAct(decoder_dim, refine_channels, kernel_size=1)
        fused_channels = detail_channels + refine_channels
        self.refine = nn.Sequential(
            ConvBNAct(fused_channels, refine_channels, kernel_size=1),
            ConvBNAct(refine_channels, refine_channels, groups=refine_channels),
            ConvBNAct(refine_channels, refine_channels, kernel_size=1),
        )

    def forward(self, rgb, sar, decoded):
        rgb_detail = self.rgb_stem(rgb)
        if self.rgb_only:
            detail = self.detail_fuse(rgb_detail)
        elif self.rgb_anchored:
            sar_detail = self.sar_stem(sar)
            reliability = torch.sigmoid(
                self.sar_reliability(torch.cat([rgb_detail, sar_detail], dim=1))
            )
            detail = (
                rgb_detail
                + torch.sigmoid(self.sar_strength_logit) * reliability * sar_detail
            )
        else:
            sar_detail = self.sar_stem(sar)
            detail = self.detail_fuse(torch.cat([rgb_detail, sar_detail], dim=1))
        semantic = self.semantic_proj(decoded)
        semantic = F.interpolate(
            semantic, size=detail.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.refine(torch.cat([semantic, detail], dim=1))


class BoundaryResidualHead(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNAct(channels, channels, groups=channels), nn.Conv2d(channels, 1, 1)
        )

    def forward(self, x):
        return self.layers(x)


class HeightConditionalResidualHead(nn.Module):
    """Soft height-range experts that correct the coarse height prediction."""

    def __init__(
        self, decoder_dim, context_channels=128, residual_scales=(6.0, 3.0, 6.0)
    ):
        super().__init__()
        residual_scales = tuple((float(value) for value in residual_scales))
        if not residual_scales:
            raise ValueError("residual_scales must contain at least one expert")
        self.num_experts = len(residual_scales)
        self.context = nn.Sequential(
            ConvBNAct(decoder_dim, context_channels, kernel_size=1),
            ConvBNAct(context_channels, context_channels),
        )
        self.gate = nn.Sequential(
            ConvBNAct(context_channels + 1, context_channels, kernel_size=1),
            nn.Conv2d(context_channels, self.num_experts, 1),
        )
        self.residual_heads = nn.ModuleList(
            [nn.Conv2d(context_channels, 1, 1) for _ in range(self.num_experts)]
        )
        self.register_buffer(
            "residual_scales",
            torch.tensor(residual_scales).view(1, self.num_experts, 1, 1),
        )
        self.strength_logit = nn.Parameter(torch.tensor(-1.386))
        for head in self.residual_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, decoded, coarse_height, mask_logits):
        context = self.context(decoded)
        gate_input = torch.cat([context, torch.log1p(coarse_height)], dim=1)
        gate_logits = self.gate(gate_input)
        gate_probability = torch.softmax(gate_logits, dim=1)
        residuals = torch.cat([head(context) for head in self.residual_heads], dim=1)
        residuals = torch.tanh(residuals) * self.residual_scales
        residuals = torch.sigmoid(self.strength_logit) * residuals
        delta = (gate_probability * residuals).sum(dim=1, keepdim=True)
        delta = delta * torch.sigmoid(mask_logits.detach())
        corrected_height = F.relu(coarse_height + delta)
        return (corrected_height, gate_logits)


class StructureGuidedLocalMonotonicCalibration(nn.Module):
    """Locally calibrate height ranges while preserving their ordering."""

    def __init__(
        self,
        decoder_dim,
        context_channels=128,
        knots=(0.0, 3.0, 6.0, 10.0, 20.0, 40.0, 100.0),
        max_log_scale=0.7,
    ):
        super().__init__()
        knot_tensor = torch.tensor(knots, dtype=torch.float32)
        self.register_buffer("left_knots", knot_tensor[:-1].view(1, -1, 1, 1))
        self.register_buffer(
            "interval_widths", (knot_tensor[1:] - knot_tensor[:-1]).view(1, -1, 1, 1)
        )
        self.max_height = float(knot_tensor[-1])
        self.max_log_scale = float(max_log_scale)
        self.interval_head = nn.Sequential(
            ConvBNAct(decoder_dim + 3, context_channels, kernel_size=1),
            ConvBNAct(context_channels, context_channels),
            nn.Conv2d(context_channels, len(knots) - 1, 1),
        )

    def reset_identity(self):
        nn.init.zeros_(self.interval_head[-1].weight)
        nn.init.zeros_(self.interval_head[-1].bias)

    def forward(self, decoded, height, mask_logits, contour_logits):
        mask_probability = torch.sigmoid(mask_logits.detach())
        contour_probability = torch.sigmoid(contour_logits.detach())
        normalized_height = torch.log1p(height).div_(math.log1p(self.max_height))
        calibration_input = torch.cat(
            [decoded, normalized_height, mask_probability, contour_probability], dim=1
        )
        log_scales = self.max_log_scale * torch.tanh(
            self.interval_head(calibration_input)
        )
        scales = torch.exp(log_scales)
        covered = (height - self.left_knots).clamp_min(0.0)
        covered = torch.minimum(covered, self.interval_widths)
        calibrated = (covered * scales).sum(dim=1, keepdim=True)
        calibrated = (
            calibrated + (height - self.max_height).clamp_min(0.0) * scales[:, -1:]
        )
        structure_confidence = mask_probability * (1.0 - 0.5 * contour_probability)
        corrected = height + structure_confidence * (calibrated - height)
        return (F.relu(corrected), log_scales)


class CMNeXtMTL(nn.Module):
    """Official CMNeXt-B4 RGB-SAR backbone with three DFC2023 output heads."""

    def __init__(
        self,
        variant="B4",
        decoder_dim=512,
        use_hrbr=False,
        use_hcr=False,
        use_slmhc=False,
        use_ttci=False,
        task_channels=256,
        ttci_legacy=False,
        sar_channels=1,
        rgb_anchored_fusion=False,
        auxiliary_initial_strength=0.02,
        rgb_anchored_hrbr=False,
        hcr_residual_scales=(6.0, 3.0, 6.0),
        slmhc_knots=(0.0, 3.0, 6.0, 10.0, 20.0, 40.0, 100.0),
        structure_only_sar=False,
        structure_fusion_initial_strength=0.1,
        directional_ttci=False,
        use_roof_context_refiner=False,
        height_guidance_stop_gradient=False,
        use_rgb_height_detail_refiner=False,
        rgb_height_detail_channels=48,
        rgb_height_detail_initial_strength=0.02,
        rgb_height_detail_max_residual=4.0,
        use_contour_structure_gate=False,
        contour_gate_kernel=9,
        contour_gate_floor=0.15,
        contour_gate_initial_strength=0.2,
        use_multiscale_height_head=False,
        multiscale_height_head_kind="lightweight",
        height_priority=False,
        distribution_height=False,
        context_detail_height=False,
        building_affinity=False,
        rgb_only=False,
    ):
        super().__init__()
        self.variant = variant
        self.rgb_only = bool(rgb_only)
        if self.rgb_only and any(
            (rgb_anchored_fusion, rgb_anchored_hrbr, structure_only_sar)
        ):
            raise ValueError("RGB-only cannot enable SAR-specific fusion paths")
        self.use_hrbr = use_hrbr
        self.use_hcr = use_hcr
        self.use_slmhc = use_slmhc
        self.use_ttci = use_ttci
        self.height_priority = bool(height_priority)
        self.distribution_height = bool(distribution_height)
        self.context_detail_height = bool(context_detail_height)
        self.building_affinity = bool(building_affinity)
        if self.building_affinity and (not self.context_detail_height):
            raise ValueError("Building affinity requires ContextDetail height")
        if self.context_detail_height and any(
            (
                distribution_height,
                use_multiscale_height_head,
                height_priority,
                ttci_legacy,
                structure_only_sar,
            )
        ):
            raise ValueError(
                "Context-detail height requires standard shared decoding and non-legacy TTCI"
            )
        if self.distribution_height and (
            use_multiscale_height_head or height_priority or ttci_legacy
        ):
            raise ValueError(
                "Distribution height requires the standard shared decoder and non-legacy TTCI"
            )
        if self.height_priority and any(
            (
                use_ttci,
                use_hcr,
                use_slmhc,
                structure_only_sar,
                use_multiscale_height_head,
                use_roof_context_refiner,
                use_rgb_height_detail_refiner,
                use_contour_structure_gate,
            )
        ):
            raise ValueError(
                "Height priority requires direct height without task interaction or refiners"
            )
        self.use_multiscale_height_head = bool(use_multiscale_height_head)
        if self.use_multiscale_height_head and (ttci_legacy or structure_only_sar):
            raise ValueError(
                "Multiscale height head requires non-legacy standard fusion"
            )
        self.ttci_legacy = ttci_legacy
        self.sar_channels = int(sar_channels)
        self.rgb_anchored_fusion = bool(rgb_anchored_fusion)
        self.structure_only_sar = bool(structure_only_sar)
        self.directional_ttci = bool(directional_ttci)
        self.use_roof_context_refiner = bool(use_roof_context_refiner)
        self.height_guidance_stop_gradient = bool(height_guidance_stop_gradient)
        self.use_rgb_height_detail_refiner = bool(use_rgb_height_detail_refiner)
        self.use_contour_structure_gate = bool(use_contour_structure_gate)
        self.contour_gate_kernel = int(contour_gate_kernel)
        self.contour_gate_floor = float(contour_gate_floor)
        if self.contour_gate_kernel < 1 or self.contour_gate_kernel % 2 == 0:
            raise ValueError("contour_gate_kernel must be a positive odd integer")
        if not 0.0 < self.contour_gate_floor <= 1.0:
            raise ValueError("contour_gate_floor must be in (0, 1]")
        if not 0.0 < float(contour_gate_initial_strength) < 1.0:
            raise ValueError("contour_gate_initial_strength must be in (0, 1)")
        if self.use_contour_structure_gate:
            initial_logit = math.log(
                float(contour_gate_initial_strength)
                / (1.0 - float(contour_gate_initial_strength))
            )
            self.contour_gate_logit = nn.Parameter(torch.tensor(initial_logit))
        if self.sar_channels < 1:
            raise ValueError("sar_channels must be positive")
        self.backbone = CMNeXtBackbone(
            model_name=variant,
            modals=["rgb"] if self.rgb_only else ["rgb", "sar"],
            extra_in_channels=self.sar_channels,
            rgb_anchored_fusion=self.rgb_anchored_fusion,
            auxiliary_initial_strength=auxiliary_initial_strength,
        )
        self.decoder = SegFormerFeatureDecoder(self.backbone.channels, decoder_dim)
        if self.height_priority:
            self.height_decoder = SegFormerFeatureDecoder(
                self.backbone.channels, decoder_dim
            )
        if self.structure_only_sar:
            self.height_decoder = SegFormerFeatureDecoder(
                self.backbone.channels, decoder_dim
            )
            self.structure_fusion = RoleDecoupledStructureFusion(
                decoder_dim, structure_fusion_initial_strength
            )
        if self.use_ttci:
            self.mask_adapter = TaskFeatureAdapter(decoder_dim, task_channels)
            self.contour_adapter = TaskFeatureAdapter(decoder_dim, task_channels)
            self.height_adapter = TaskFeatureAdapter(decoder_dim, task_channels)
            self.task_interaction = TriTaskCollaborativeInteraction(
                task_channels,
                prediction_residual=not self.ttci_legacy,
                directional_gradient_isolation=self.directional_ttci,
            )
        prediction_channels = (
            task_channels if self.use_ttci and self.ttci_legacy else decoder_dim
        )
        self.mask_head = PredictionHead(prediction_channels)
        self.contour_head = PredictionHead(prediction_channels)
        if self.context_detail_height:
            from hdbf.dfc_context_detail_height import ContextDetailHeightHead

            self.height_head = ContextDetailHeightHead(
                prediction_channels, building_affinity=self.building_affinity
            )
        elif self.distribution_height:
            from hdbf.dfc_height_distribution import DistributionHeightHead

            self.height_head = DistributionHeightHead(prediction_channels)
        elif self.use_multiscale_height_head:
            raise ValueError("Historical multiscale height head is not included")
        else:
            self.height_head = PredictionHead(prediction_channels)
        if self.use_hcr:
            self.height_conditional_residual = HeightConditionalResidualHead(
                decoder_dim, residual_scales=hcr_residual_scales
            )
        if self.use_slmhc:
            self.height_calibrator = StructureGuidedLocalMonotonicCalibration(
                decoder_dim, knots=slmhc_knots
            )
        if self.use_hrbr:
            self.boundary_refiner = HighResolutionBoundaryRefiner(
                decoder_dim,
                sar_in_channels=self.sar_channels,
                rgb_anchored=rgb_anchored_hrbr,
                sar_initial_strength=auxiliary_initial_strength,
                rgb_only=self.rgb_only,
            )
            self.mask_boundary_head = BoundaryResidualHead(64)
            self.contour_boundary_head = BoundaryResidualHead(64)
            self.mask_refine_logit = nn.Parameter(torch.tensor(-1.386))
            self.contour_refine_logit = nn.Parameter(torch.tensor(-1.386))
        if self.use_roof_context_refiner:
            self.roof_context_refiner = RoofContextResidualRefiner(decoder_dim)
        if self.use_rgb_height_detail_refiner:
            self.rgb_height_detail_refiner = RGBHeightDetailResidualRefiner(
                decoder_dim,
                channels=rgb_height_detail_channels,
                max_residual=rgb_height_detail_max_residual,
                initial_strength=rgb_height_detail_initial_strength,
            )
        self.apply(self._init_weights)
        if self.distribution_height or self.context_detail_height:
            self.height_head.reset_output()
        if self.height_priority:
            self.height_decoder.load_state_dict(self.decoder.state_dict())
        if self.use_multiscale_height_head:
            self.height_head.reset_local_gates()
        if self.structure_only_sar:
            self.height_decoder.load_state_dict(self.decoder.state_dict())
            self.structure_fusion.reset_identity()
        if self.rgb_anchored_fusion:
            for fusion in self.backbone.rgb_anchored_fusions:
                fusion.reset_parameters()
        if self.use_hrbr and self.boundary_refiner.rgb_anchored:
            nn.init.zeros_(self.boundary_refiner.sar_reliability.weight)
            nn.init.zeros_(self.boundary_refiner.sar_reliability.bias)
        if self.use_hcr:
            for head in self.height_conditional_residual.residual_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        if self.use_slmhc:
            self.height_calibrator.reset_identity()
        if self.use_ttci:
            self.task_interaction.reset_identity()
        if self.use_roof_context_refiner:
            self.roof_context_refiner.reset_identity()
        if self.use_rgb_height_detail_refiner:
            self.rgb_height_detail_refiner.reset_identity()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = (
                module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            )
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _unwrap_checkpoint(checkpoint):
        for key in ("state_dict", "model"):
            if isinstance(checkpoint, dict) and key in checkpoint:
                checkpoint = checkpoint[key]
        cleaned = {}
        for key, value in checkpoint.items():
            if key.startswith("module."):
                key = key[7:]
            if key.startswith("backbone."):
                key = key[9:]
            cleaned[key] = value
        return cleaned

    @staticmethod
    def _convert_huggingface_mit(source):
        """Convert NVIDIA Hugging Face SegFormer encoder keys to official MiT keys."""
        hf_prefix = None
        for prefix in ("segformer.encoder.", "encoder."):
            if any((key.startswith(prefix + "patch_embeddings.") for key in source)):
                hf_prefix = prefix
                break
        if hf_prefix is None:
            return (source, False)
        converted = {}
        for stage in range(4):
            hf_patch = f"{hf_prefix}patch_embeddings.{stage}"
            mit_patch = f"patch_embed{stage + 1}"
            for hf_name, mit_name in (
                ("proj.weight", "proj.weight"),
                ("proj.bias", "proj.bias"),
                ("layer_norm.weight", "norm.weight"),
                ("layer_norm.bias", "norm.bias"),
            ):
                value = source.get(f"{hf_patch}.{hf_name}")
                if value is not None:
                    converted[f"{mit_patch}.{mit_name}"] = value
            hf_norm = f"{hf_prefix}layer_norm.{stage}"
            for parameter in ("weight", "bias"):
                value = source.get(f"{hf_norm}.{parameter}")
                if value is not None:
                    converted[f"norm{stage + 1}.{parameter}"] = value
            block_prefix = f"{hf_prefix}block.{stage}."
            block_ids = sorted(
                {
                    int(key[len(block_prefix) :].split(".", 1)[0])
                    for key in source
                    if key.startswith(block_prefix)
                }
            )
            for block_id in block_ids:
                hf_block = f"{block_prefix}{block_id}"
                mit_block = f"block{stage + 1}.{block_id}"
                direct_names = (
                    ("layer_norm_1", "norm1"),
                    ("attention.self.query", "attn.q"),
                    ("attention.output.dense", "attn.proj"),
                    ("attention.self.sr", "attn.sr"),
                    ("attention.self.layer_norm", "attn.norm"),
                    ("layer_norm_2", "norm2"),
                    ("mlp.dense1", "mlp.fc1"),
                    ("mlp.dwconv.dwconv", "mlp.dwconv.dwconv"),
                    ("mlp.dense2", "mlp.fc2"),
                )
                for hf_name, mit_name in direct_names:
                    for parameter in ("weight", "bias"):
                        value = source.get(f"{hf_block}.{hf_name}.{parameter}")
                        if value is not None:
                            converted[f"{mit_block}.{mit_name}.{parameter}"] = value
                for parameter in ("weight", "bias"):
                    key = source.get(f"{hf_block}.attention.self.key.{parameter}")
                    value = source.get(f"{hf_block}.attention.self.value.{parameter}")
                    if key is not None and value is not None:
                        converted[f"{mit_block}.attn.kv.{parameter}"] = torch.cat(
                            [key, value], dim=0
                        )
        return (converted, True)

    def init_pretrained(self, checkpoint_path, require_complete=False):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        source = self._unwrap_checkpoint(checkpoint)
        (source, is_huggingface) = self._convert_huggingface_mit(source)
        target = self.backbone.state_dict()
        compatible = {
            key: value
            for (key, value) in source.items()
            if key in target and target[key].shape == value.shape
        }
        if require_complete:
            rgb_prefixes = tuple(
                (
                    f"{name}{stage}."
                    for name in ("patch_embed", "block", "norm")
                    for stage in range(1, 5)
                )
            )
            expected = {key for key in target if key.startswith(rgb_prefixes)}
            missing = sorted(expected - compatible.keys())
            extra = sorted(
                (
                    key
                    for key in source
                    if key.startswith(rgb_prefixes) and key not in expected
                )
            )
            if missing or extra or (not expected):
                raise ValueError(
                    f"Incomplete or wrong-variant MiT-{self.variant} pretraining: {len(expected - set(missing))}/{len(expected)} matching RGB tensors; missing/mismatched={missing[:8]}, extra RGB keys={extra[:8]}. Use the matching backbone checkpoint; training has not started."
                )
            print(
                f"Complete MiT-{self.variant} RGB pretraining verified: {len(expected)}/{len(expected)} tensors"
            )
        message = self.backbone.load_state_dict(compatible, strict=False)
        sar_state = {}
        for stage in range(1, 5):
            src_prefix = f"patch_embed{stage}"
            dst_prefix = f"extra_downsample_layers.{stage - 1}"
            conv_weight = source.get(f"{src_prefix}.proj.weight")
            if conv_weight is not None:
                if stage == 1:
                    conv_weight = conv_weight.mean(dim=1, keepdim=True)
                    if self.sar_channels > 1:
                        conv_weight = conv_weight.repeat(
                            1, self.sar_channels, 1, 1
                        ).div_(self.sar_channels)
                sar_state[f"{dst_prefix}.proj.module.weight"] = conv_weight
            conv_bias = source.get(f"{src_prefix}.proj.bias")
            if conv_bias is not None:
                sar_state[f"{dst_prefix}.proj.module.bias"] = conv_bias
            for parameter in ("weight", "bias"):
                value = source.get(f"{src_prefix}.norm.{parameter}")
                if value is not None:
                    sar_state[f"{dst_prefix}.norm.ln_0.{parameter}"] = value
        sar_state = {
            key: value
            for (key, value) in sar_state.items()
            if key in target and target[key].shape == value.shape
        }
        self.backbone.load_state_dict(sar_state, strict=False)
        print(
            f"Loaded MiT-{self.variant} pretrained tensors ({('Hugging Face' if is_huggingface else 'official')} format): {len(compatible)} RGB, {len(sar_state)} SAR patch tensors"
        )
        if message.unexpected_keys:
            print(f"Unexpected pretrained keys: {message.unexpected_keys}")

    def forward(self, rgb, sar=None, return_intermediates=False, sar_scale=1.0):
        if hasattr(self, "geometry_interaction"):
            raise ValueError(
                "Historical architecture is not included in the final release"
            )
        output_size = rgb.shape[-2:]
        if self.structure_only_sar:
            (features, rgb_features) = self.backbone(
                [rgb, sar], return_rgb_features=True, auxiliary_scale=sar_scale
            )
            fused_decoded = self.decoder(features)
            height_decoded = self.height_decoder(rgb_features)
            decoded = self.structure_fusion(height_decoded, fused_decoded)
        else:
            features = self.backbone([rgb] if self.rgb_only else [rgb, sar])
            if getattr(self, "task_split_decoder", False):
                (decoded, height_decoded) = self.decoder(features)
            elif self.height_priority:
                decoded = self.decoder([feature.detach() for feature in features])
                height_decoded = self.height_decoder(features)
            else:
                decoded = self.decoder(features)
                height_decoded = decoded
        if self.use_ttci and self.ttci_legacy:
            mask_feature = self.mask_adapter(decoded)
            contour_feature = self.contour_adapter(decoded)
            height_feature = self.height_adapter(height_decoded)
            (mask_feature, contour_feature, height_feature) = self.task_interaction(
                mask_feature, contour_feature, height_feature
            )
        else:
            mask_feature = decoded
            contour_feature = decoded
            height_feature = height_decoded
        mask = self.mask_head(mask_feature)
        contour = self.contour_head(contour_feature)
        encoded_geometry = None
        height_geometry_stats = {}
        if hasattr(self, "height_geometry_fusion"):
            if (
                not hasattr(self, "unirep_structure")
                or not self.unirep_structure.local_structure
                or (not getattr(self, "task_split_decoder", False))
                or self.ttci_legacy
                or (not self.context_detail_height)
                or (not isinstance(features, dict))
            ):
                raise ValueError(
                    "Geometry-height fusion requires Local UniRep, TaskSplit and ContextDetail"
                )
            encoded_geometry = self.unirep_structure.encode_geometry(
                rgb, features["structure"], decoded
            )
            (height_decoded, height_geometry_stats) = self.height_geometry_fusion(
                height_decoded,
                encoded_geometry["fused"],
                encoded_geometry["cnn_decoded"],
                mask,
                encoded_geometry["cnn_logits"],
            )
            height_feature = height_decoded
        if self.context_detail_height:
            head_output = self.height_head(height_feature, mask)
            (coarse_height, context_base, local_detail) = head_output[:3]
        elif self.distribution_height:
            (coarse_height, height_range_logits, height_range_candidates) = (
                self.height_head(height_feature)
            )
        else:
            coarse_height = F.softplus(
                self.height_head(features, rgb=rgb)
                if self.use_multiscale_height_head
                else self.height_head(height_feature)
            )
        head_height = coarse_height if return_intermediates else None
        if self.use_ttci and (not self.ttci_legacy):
            mask_task = self.mask_adapter(decoded)
            contour_task = self.contour_adapter(decoded)
            height_task = self.height_adapter(height_decoded)
            (mask_task, contour_task, height_task) = self.task_interaction(
                mask_task, contour_task, height_task
            )
            (mask_delta, contour_delta, height_delta) = (
                self.task_interaction.prediction_corrections(
                    mask_task, contour_task, height_task
                )
            )
            mask = mask + mask_delta
            contour = contour + contour_delta
            coarse_height = F.relu(coarse_height + height_delta)
        ttci_coarse_height = coarse_height if return_intermediates else None
        ttci_mask = mask
        ttci_contour = contour
        height = coarse_height
        isolate_height_guidance = (
            self.structure_only_sar or self.height_guidance_stop_gradient
        )
        height_mask_guidance = mask.detach() if isolate_height_guidance else mask
        height_contour_guidance = (
            contour.detach() if isolate_height_guidance else contour
        )
        hcr_gate = None
        if self.use_hcr:
            (height, hcr_gate) = self.height_conditional_residual(
                height_decoded, coarse_height, height_mask_guidance
            )
        height_after_hcr = height if return_intermediates else None
        if self.use_slmhc:
            (height, _) = self.height_calibrator(
                height_decoded, height, height_mask_guidance, height_contour_guidance
            )
        height_after_slmhc = height if return_intermediates else None
        if self.use_roof_context_refiner:
            height = self.roof_context_refiner(
                height_decoded, height, height_mask_guidance, height_contour_guidance
            )
        rgb_height_detail_gate = None
        if self.use_rgb_height_detail_refiner:
            (height, rgb_height_detail_gate) = self.rgb_height_detail_refiner(
                rgb, height_decoded, height
            )
        structure_aux = {}
        if hasattr(self, "unirep_structure"):
            needs_height = self.unirep_structure.query_structure or getattr(
                self.unirep_structure, "local_structure", False
            )
            geometry_kwargs = {"height": height} if needs_height else {}
            if encoded_geometry is None:
                geometry = self.unirep_structure(
                    rgb,
                    features["structure"],
                    decoded,
                    mask,
                    contour,
                    **geometry_kwargs,
                )
            else:
                geometry = self.unirep_structure.decode_geometry(
                    encoded_geometry, mask, contour, **geometry_kwargs
                )
            (mask, contour) = (geometry["mask"], geometry["contour"])
            structure_aux = {
                k: v for (k, v) in geometry.items() if k not in ("mask", "contour")
            }
        elif self.use_hrbr:
            boundary_feature = self.boundary_refiner(
                rgb, None if self.rgb_only else sar * float(sar_scale), decoded
            )
            half_size = boundary_feature.shape[-2:]
            mask = F.interpolate(mask, half_size, mode="bilinear", align_corners=False)
            contour = F.interpolate(
                contour, half_size, mode="bilinear", align_corners=False
            )
            mask = mask + torch.sigmoid(
                self.mask_refine_logit
            ) * self.mask_boundary_head(boundary_feature)
            contour = contour + torch.sigmoid(
                self.contour_refine_logit
            ) * self.contour_boundary_head(boundary_feature)
        mask = F.interpolate(mask, output_size, mode="bilinear", align_corners=False)
        contour = F.interpolate(
            contour, output_size, mode="bilinear", align_corners=False
        )
        contour_support = None
        if self.use_contour_structure_gate:
            contour_dtype = contour.dtype
            mask_probability = torch.sigmoid(mask.detach().float())
            contour_support = F.max_pool2d(
                mask_probability,
                kernel_size=self.contour_gate_kernel,
                stride=1,
                padding=self.contour_gate_kernel // 2,
            )
            contour_support = (
                self.contour_gate_floor
                + (1.0 - self.contour_gate_floor) * contour_support
            )
            gate_strength = torch.sigmoid(self.contour_gate_logit.float())
            contour_gate = 1.0 - gate_strength * (1.0 - contour_support)
            contour_probability = torch.sigmoid(contour.float()) * contour_gate
            contour = torch.logit(
                contour_probability.clamp(min=0.0001, max=1.0 - 0.0001)
            ).to(contour_dtype)
        height = F.interpolate(
            height, output_size, mode="bilinear", align_corners=False
        )
        (mask_before_refinement, contour_before_refinement) = (mask, contour)
        if hasattr(self, "full_resolution_structure"):
            (mask, contour) = self.full_resolution_structure(
                rgb, sar, boundary_feature, mask, contour
            )
        output = {"mask": mask, "contour": contour, "height": height}
        if self.distribution_height:
            output.update(
                height_range_logits=height_range_logits,
                height_range_candidates=height_range_candidates,
                height_range_edges=self.height_head.edges,
            )
        if return_intermediates:
            output.update(structure_aux)
            output.update(height_geometry_stats)
            if hasattr(self, "full_resolution_structure"):
                output["mask_before_refinement"] = mask_before_refinement
                output["contour_before_refinement"] = contour_before_refinement
            if self.building_affinity:
                output["building_affinity_logits"] = head_output[3]
            if self.context_detail_height:
                output["height_context_base"] = F.interpolate(
                    context_base, output_size, mode="bilinear", align_corners=False
                )
                output["height_local_detail"] = F.interpolate(
                    local_detail, output_size, mode="bilinear", align_corners=False
                )
            output["height_head"] = F.interpolate(
                head_height, output_size, mode="bilinear", align_corners=False
            )
            output["height_coarse"] = F.interpolate(
                ttci_coarse_height, output_size, mode="bilinear", align_corners=False
            )
            output["height_after_hcr"] = F.interpolate(
                height_after_hcr, output_size, mode="bilinear", align_corners=False
            )
            output["height_after_slmhc"] = F.interpolate(
                height_after_slmhc, output_size, mode="bilinear", align_corners=False
            )
            output["ttci_mask"] = F.interpolate(
                ttci_mask, output_size, mode="bilinear", align_corners=False
            )
            output["ttci_contour"] = F.interpolate(
                ttci_contour, output_size, mode="bilinear", align_corners=False
            )
            if contour_support is not None:
                output["contour_support"] = contour_support
                output["contour_gate_strength"] = torch.sigmoid(self.contour_gate_logit)
        if hcr_gate is not None:
            output["hcr_gate"] = F.interpolate(
                hcr_gate, output_size, mode="bilinear", align_corners=False
            )
        if return_intermediates and rgb_height_detail_gate is not None:
            output["rgb_height_detail_gate"] = F.interpolate(
                rgb_height_detail_gate,
                output_size,
                mode="bilinear",
                align_corners=False,
            )
        return output
