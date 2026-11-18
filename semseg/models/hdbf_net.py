"""DINOv2 RGB features with the existing B4 SAR and HDBF task pathways."""

import bootstrap
import hashlib
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from dinov2.models.vision_transformer import vit_base
from semseg.models.cmnext_mtl import CMNeXtMTL
from hdbf.task_split_decoder import TaskSplitPyramidDecoder
from hdbf.task_aware_fusion import TaskAwareFusion

ARCHITECTURE = "hdbf_dinov2_vitb14_sar_b4_v1"
LOCAL_ARCHITECTURE = "hdbf_dinov2_local_sar_v1"
STRUCTURE_ARCHITECTURE = "hdbf_dinov2_structure_local_v1"
TASK_SPLIT_ARCHITECTURE = "hdbf_dinov2_task_split_pyramid_v1"
TASK_FUSION_ARCHITECTURE = "hdbf_dinov2_task_aware_fusion_v1"
FUSION_ARCHITECTURES = {
    "cmnext": ARCHITECTURE,
    "local-sar": LOCAL_ARCHITECTURE,
    "structure-local": STRUCTURE_ARCHITECTURE,
}
SOURCE_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def checkpoint_fusion_type(model_config, train_args):
    architecture = model_config.get("architecture")
    inverse = {value: key for (key, value) in FUSION_ARCHITECTURES.items()}
    inverse[TASK_SPLIT_ARCHITECTURE] = "cmnext"
    inverse[TASK_FUSION_ARCHITECTURE] = "task-aware"
    if architecture not in inverse:
        raise ValueError(
            "Unsupported DINOv2 architecture; use the legacy evaluator for MiT models"
        )
    expected = inverse[architecture]
    fusion = model_config.get("fusion_type", train_args.get("fusion_type", expected))
    if fusion != expected:
        raise ValueError("Checkpoint architecture and fusion_type disagree")
    return fusion


def checkpoint_task_split(model_config, train_args):
    expected = model_config.get("architecture") in (
        TASK_SPLIT_ARCHITECTURE,
        TASK_FUSION_ARCHITECTURE,
    )
    enabled = model_config.get(
        "task_split_decoder", train_args.get("task_split_decoder", expected)
    )
    if not isinstance(enabled, bool) or enabled != expected:
        raise ValueError("Checkpoint architecture and task_split_decoder disagree")
    return enabled


def native_grid(feature, native_size, padded_size, stride):
    """Sample padded patch features at native pixel-grid cell centers."""
    (h, w) = native_size
    (ph, pw) = padded_size
    (oh, ow) = (math.ceil(h / stride), math.ceil(w / stride))
    y = (torch.arange(oh, device=feature.device, dtype=torch.float32) + 0.5) * stride
    x = (torch.arange(ow, device=feature.device, dtype=torch.float32) + 0.5) * stride
    (yy, xx) = torch.meshgrid(y * (2.0 / ph) - 1.0, x * (2.0 / pw) - 1.0, indexing="ij")
    grid = torch.stack((xx, yy), -1).unsqueeze(0).expand(feature.shape[0], -1, -1, -1)
    return F.grid_sample(
        feature.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


class DinoSARBackbone(nn.Module):
    channels = [64, 128, 320, 512]
    tap_layers = (2, 5, 8, 11)

    def __init__(self, sar_template, gradient_checkpointing=True, fusion_type="cmnext"):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise RuntimeError("DINOv2 branch requires PyTorch >=2.0 with SDPA")
        self.gradient_checkpointing = gradient_checkpointing
        self.extra_downsample_layers = sar_template.extra_downsample_layers
        self.fusion_type = fusion_type
        if fusion_type in ("cmnext", "structure-local"):
            self.FRMs = sar_template.FRMs
            self.FFMs = sar_template.FFMs
        elif fusion_type == "task-aware":
            pass
        elif fusion_type == "local-sar":
            raise ValueError(
                "Historical architecture is not included in the final release"
            )
        else:
            raise ValueError(f"Unknown fusion: {fusion_type}")
        for stage in range(1, 5):
            setattr(
                self,
                f"extra_block{stage}",
                getattr(sar_template, f"extra_block{stage}"),
            )
            setattr(
                self, f"extra_norm{stage}", getattr(sar_template, f"extra_norm{stage}")
            )
        self.dino = vit_base(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            block_chunks=0,
            num_register_tokens=0,
        )
        self.rgb_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(768, c, 1, bias=False), nn.GroupNorm(8, c), nn.GELU()
                )
                for c in self.channels
            ]
        )
        if fusion_type == "task-aware":
            with torch.random.fork_rng(devices=[]):
                self.task_fusions = nn.ModuleList(
                    [TaskAwareFusion(c) for c in self.channels]
                )
        if fusion_type == "structure-local":
            raise ValueError(
                "Historical architecture is not included in the final release"
            )

    def rgb_pyramid(self, rgb):
        (h, w) = rgb.shape[-2:]
        (ph, pw) = (math.ceil(h / 14) * 14, math.ceil(w / 14) * 14)
        image = F.pad(rgb, (0, pw - w, 0, ph - h), mode="replicate")
        tokens = self.dino.prepare_tokens_with_masks(image)
        features = []
        for index, block in enumerate(self.dino.blocks):
            if (
                self.training
                and self.gradient_checkpointing
                and torch.is_grad_enabled()
            ):
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if index in self.tap_layers:
                patch = self.dino.norm(tokens)[:, 1:]
                feature = patch.transpose(1, 2).reshape(
                    rgb.shape[0], 768, ph // 14, pw // 14
                )
                stage = len(features)
                feature = self.rgb_projections[stage](feature)
                features.append(native_grid(feature, (h, w), (ph, pw), 4 * 2**stage))
        return features

    def forward(self, inputs):
        (rgb, sar) = inputs
        if rgb.shape[-2:] != sar.shape[-2:]:
            raise ValueError("RGB and SAR must share the native spatial grid")
        rgb_features = self.rgb_pyramid(rgb)
        geometry = (
            self.geometry_pyramid(rgb)
            if self.fusion_type == "structure-local"
            else None
        )
        (extra, outputs) = ([sar], [])
        height_outputs = []
        for i, rgb_feature in enumerate(rgb_features):
            (extra, _, _) = self.extra_downsample_layers[i](extra)
            sar_feature = extra[0]
            for block in getattr(self, f"extra_block{i + 1}"):
                sar_feature = block(sar_feature)
            sar_feature = getattr(self, f"extra_norm{i + 1}")(sar_feature)
            if rgb_feature.shape[-2:] != sar_feature.shape[-2:]:
                raise RuntimeError("DINO projection and SAR feature grids differ")
            if self.fusion_type in ("cmnext", "structure-local"):
                (rectified_rgb, rectified_sar) = self.FRMs[i](rgb_feature, sar_feature)
                fused = self.FFMs[i](rectified_rgb, rectified_sar)
                if geometry is not None and i < 2:
                    fused = self.structure_supplements[i](
                        rgb_feature, sar_feature, fused, geometry[i]
                    )
                outputs.append(fused)
                extra = [rectified_sar]
            elif self.fusion_type == "task-aware":
                (structure, height) = self.task_fusions[i](rgb_feature, sar_feature)
                outputs.append(structure)
                height_outputs.append(height)
                extra = [sar_feature]
            else:
                outputs.append(self.local_fusions[i](rgb_feature, sar_feature))
                extra = [sar_feature]
        if self.fusion_type == "task-aware":
            if hasattr(self, "structure_detail"):
                local = self.structure_detail(torch.cat((rgb, sar), dim=1))
                if local.shape[-2:] != outputs[0].shape[-2:]:
                    local = F.interpolate(
                        local,
                        outputs[0].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                outputs[0] = self.structure_detail_fuse(
                    torch.cat((outputs[0], local), dim=1)
                )
            return {"structure": outputs, "height": height_outputs}
        return outputs


class DinoHDBF(CMNeXtMTL):

    def __init__(
        self,
        gradient_checkpointing=True,
        fusion_type="cmnext",
        segmentation_refinement=False,
        task_split_decoder=False,
        structure_isolation=False,
        geometry_interaction=False,
        **kwargs,
    ):
        if structure_isolation:
            if fusion_type != "task-aware" or not task_split_decoder:
                raise ValueError(
                    "Structure isolation requires TaskFusion and TaskSplit"
                )
            if not kwargs.get("directional_ttci") or not kwargs.get(
                "height_guidance_stop_gradient"
            ):
                raise ValueError(
                    "Structure isolation requires directional TTCI and detached height guidance"
                )
        if task_split_decoder and (
            fusion_type not in ("cmnext", "task-aware") or segmentation_refinement
        ):
            raise ValueError(
                "Task-split requires cmnext/task-aware fusion without final segmentation refinement"
            )
        if fusion_type == "task-aware" and (not task_split_decoder):
            raise ValueError("Task-aware fusion requires task-split decoder")
        if kwargs.get("variant", "B4") != "B4":
            raise ValueError(
                "variant=B4 identifies the fixed SAR branch, not the RGB encoder"
            )
        if any(
            (
                kwargs.get(k, False)
                for k in (
                    "rgb_only",
                    "height_priority",
                    "distribution_height",
                    "structure_only_sar",
                    "rgb_anchored_fusion",
                    "ttci_legacy",
                )
            )
        ):
            raise ValueError(
                "Unsupported legacy mode in the main DINOv2 HDBF configuration"
            )
        super().__init__(**kwargs)
        if geometry_interaction:
            raise ValueError(
                "Historical architecture is not included in the final release"
            )
        self.backbone = DinoSARBackbone(
            self.backbone, gradient_checkpointing, fusion_type
        )
        self.structure_isolation = bool(structure_isolation)
        if self.structure_isolation:
            for fusion in self.backbone.task_fusions:
                fusion.isolate_structure = True

            def detail_block(cin, cout, stride=1):
                return nn.Sequential(
                    nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                    nn.GroupNorm(8, cout),
                    nn.GELU(),
                )

            with torch.random.fork_rng(devices=[]):
                self.backbone.structure_detail = nn.Sequential(
                    detail_block(3 + self.sar_channels, 32, 2),
                    detail_block(32, 32),
                    detail_block(32, 64, 2),
                    detail_block(64, 64),
                )
                self.backbone.structure_detail_fuse = nn.Sequential(
                    detail_block(128, 64), detail_block(64, 64)
                )
        self.architecture = (
            TASK_FUSION_ARCHITECTURE
            if fusion_type == "task-aware"
            else FUSION_ARCHITECTURES[fusion_type]
        )
        self.task_split_decoder = bool(task_split_decoder)
        if self.task_split_decoder:
            with torch.random.fork_rng(devices=[]):
                self.decoder = TaskSplitPyramidDecoder(
                    self.backbone.channels,
                    output_channels=kwargs.get("decoder_dim", 512),
                )
            self.architecture = (
                TASK_FUSION_ARCHITECTURE
                if fusion_type == "task-aware"
                else TASK_SPLIT_ARCHITECTURE
            )
        if segmentation_refinement:
            raise ValueError(
                "Historical architecture is not included in the final release"
            )

    def init_pretrained(self, checkpoint_path, require_complete=True):
        source = torch.load(checkpoint_path, map_location="cpu")
        self.backbone.dino.load_state_dict(source, strict=True)
        with open(checkpoint_path, "rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.pretrained_sha256 = digest.hexdigest()
        print(
            f"DINOv2 ViT-B/14 strict load: {len(source)} tensors; SHA256={self.pretrained_sha256}"
        )

    def init_sar_pretrained(self, path):
        source = self._unwrap_checkpoint(torch.load(path, map_location="cpu"))
        (source, _) = self._convert_huggingface_mit(source)
        (target, state) = (self.backbone.state_dict(), {})
        for i in range(1, 5):
            (src, dst) = (f"patch_embed{i}", f"extra_downsample_layers.{i - 1}")
            weight = source[f"{src}.proj.weight"]
            if i == 1:
                weight = weight.mean(1, keepdim=True)
                channels = target[f"{dst}.proj.module.weight"].shape[1]
                weight = weight.repeat(1, channels, 1, 1) / channels
            state[f"{dst}.proj.module.weight"] = weight
            state[f"{dst}.proj.module.bias"] = source[f"{src}.proj.bias"]
            for suffix in ("weight", "bias"):
                state[f"{dst}.norm.ln_0.{suffix}"] = source[f"{src}.norm.{suffix}"]
        if len(state) != 16 or any(
            (k not in target or target[k].shape != v.shape for (k, v) in state.items())
        ):
            raise ValueError(
                "SAR initialization must match the reference B4 patch pyramid"
            )
        self.backbone.load_state_dict(state, strict=False)
        print("Reference B4 SAR patch initialization: 16/16 tensors")
