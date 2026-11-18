import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from hdbf.dataset_dfc2023 import DFC2023Dataset
from hdbf.dfc_height_distribution import distribution_losses, range_confusion
from semseg.models.hdbf_net import ARCHITECTURE, SOURCE_COMMIT
from ablation_dfc2023.model import AblationDinoHDBF, apply_case
from ablation_dfc2023.protocol import CASES, verify_reference


def parse_args():
    parser = argparse.ArgumentParser("Train DINOv2 HDBF-Net on DFC2023")
    parser.add_argument("--ablation-case", required=True, choices=CASES)
    parser.add_argument("--reference-config", required=True)
    parser.set_defaults(baseline=False)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument(
        "--require-sensor-coverage",
        action="store_true",
        help="Reject evaluation sensors absent from training; opt in for in-domain comparisons.",
    )
    parser.add_argument(
        "--require-sensor-stratification",
        action="store_true",
        help="Require the same sensor groups in train, val, and test; prevents validation missing an evaluated sensor.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints_dfc2023/hdbf_dinov2_affinity_no_slmhc100_best_mae",
    )
    parser.add_argument(
        "--pretrained",
        default=str(
            bootstrap.ROOT / "checkpoints/pretrained/dinov2/dinov2_vitb14_pretrain.pth"
        ),
    )
    parser.add_argument("--variant", default="B4", choices=["B2", "B4", "B5"])
    parser.add_argument(
        "--sar-pretrained",
        default="checkpoints/pretrained/segformer/segformer_b4_ade20k.bin",
    )
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-complete-pretrained",
        action="store_true",
        help="Reject wrong-variant or incomplete RGB encoder pretraining",
    )
    parser.add_argument("--decoder-dim", type=int, default=512)
    parser.add_argument("--hrbr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hcr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable height-range conditional residual correction.",
    )
    parser.add_argument(
        "--slmhc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable structure-guided local monotonic height calibration.",
    )
    parser.add_argument(
        "--bihm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable equal-weight building-instance height modeling.",
    )
    parser.add_argument(
        "--ttci",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gated residual tri-task collaborative interaction.",
    )
    parser.add_argument("--task-channels", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use page-locked DataLoader memory. Keep disabled if pinning is unstable.",
    )
    parser.add_argument("--lr", type=float, default=6e-05)
    parser.add_argument(
        "--rgb-backbone-lr-scale",
        type=float,
        default=0.1,
        help="LR multiplier for the pretrained RGB encoder only; fusion and task layers keep --lr.",
    )
    parser.add_argument("--rgb-jitter-probability", type=float, default=0.0)
    parser.add_argument("--rgb-jitter-strength", type=float, default=0.1)
    parser.add_argument(
        "--rgb-normalization",
        choices=["none", "imagenet"],
        default="imagenet",
        help="RGB input normalization; stored in the checkpoint for evaluation.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--poly-power", type=float, default=0.35)
    parser.add_argument("--lr-schedule", choices=["cosine", "poly"], default="cosine")
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.05,
        help="Final LR divided by the base LR for cosine and polynomial schedules.",
    )
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--ema-bn-policy",
        choices=["average", "copy"],
        default="average",
        help="Copy BN running statistics without a second EMA, or retain legacy averaging.",
    )
    parser.add_argument("--mask-weight", type=float, default=1.0)
    parser.add_argument("--contour-weight", type=float, default=0.4)
    parser.add_argument("--boundary-band-width", type=int, default=3)
    parser.add_argument("--height-weight", type=float, default=1.2)
    parser.add_argument(
        "--final-height-loss",
        choices=["plain", "convnext-weighted"],
        default="plain",
        help="Final height supervision only; auxiliary losses and evaluation stay unchanged.",
    )
    parser.add_argument(
        "--height-head-aux-weight",
        type=float,
        default=0.2,
        help="Smooth L1 weight for full-resolution initial height H0 on the same valid building pixels.",
    )
    parser.add_argument("--hcr-range-weight", type=float, default=0.05)
    parser.add_argument("--slmhc-tail-weight", type=float, default=0.05)
    parser.add_argument("--slmhc-warmup-epochs", type=int, default=50)
    parser.add_argument("--bihm-weight", type=float, default=0.05)
    parser.add_argument("--bihm-warmup-epochs", type=int, default=50)
    parser.add_argument("--bihm-min-pixels", type=int, default=16)
    parser.add_argument("--bihm-min-valid-ratio", type=float, default=0.5)
    parser.set_defaults(bihm_height_balance=False)
    parser.add_argument("--bg-height-weight", type=float, default=0.02)
    parser.add_argument("--min-height", type=float, default=0.1)
    parser.add_argument("--sar-clip", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.set_defaults(spatial_group_sampling=False)
    parser.set_defaults(height_priority=False)
    parser.set_defaults(distribution_height=False)
    parser.add_argument(
        "--context-detail-height", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--building-affinity", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="RGB encoder and RGB-only HRBR; no SAR encoder or fusion modules",
    )
    parser.add_argument("--affinity-loss-weight", type=float, default=0.05)
    parser.set_defaults(height_teacher=None)
    parser.add_argument("--height-kd-weight", type=float, default=0.2)
    parser.add_argument(
        "--height-kd-temperature",
        type=float,
        default=3.0,
        help="Training GT error scale in meters for teacher reliability",
    )
    parser.add_argument("--height-kd-warmup-epochs", type=int, default=10)
    parser.add_argument("--teacher-microbatch", type=int, default=1)
    parser.add_argument("--height-range-cls-weight", type=float, default=0.2)
    parser.add_argument("--height-range-expert-weight", type=float, default=0.2)
    parser.add_argument("--spatial-grid-m", type=float, default=768.0)
    parser.add_argument("--spatial-sampling-fraction", type=float, default=0.5)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="HDBF-Net-DINOv2-Affinity-NoSLMHC")
    parser.add_argument(
        "--checkpoint-mode",
        choices=["full", "inference"],
        default="inference",
        help="Save resumable training checkpoints or inference-only weight packages. Inference packages retain epoch/metrics/args but omit optimizer, scheduler and scaler.",
    )
    parser.add_argument(
        "--save-best-mae-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ordinary mode saves only best MAE; UniRep mode also retains task-best and final weights.",
    )
    parser.set_defaults(
        slmhc=True,
        lr=3e-05,
        model_name="HDBF-Net-DINOv2-Affinity-SLMHC-LR3e5",
        checkpoint_dir="checkpoints_dfc2023/hdbf_dinov2_affinity_slmhc_lr3e5_100_best_mae",
    )
    parser.add_argument(
        "--fusion-type",
        choices=("cmnext", "local-sar", "structure-local", "task-aware"),
        default="cmnext",
    )
    parser.add_argument(
        "--segmentation-refinement",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--task-split-decoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--unirep-pretrained", default=None)
    parser.add_argument("--unirep-lr-scale", type=float, default=0.1)
    parser.add_argument("--structure-branch-weight", type=float, default=0.2)
    parser.add_argument(
        "--geometry-interaction", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--query-structure", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--query-structure-weight", type=float, default=0.2)
    parser.add_argument(
        "--local-structure", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--local-structure-weight", type=float, default=0.2)
    parser.add_argument(
        "--geometry-height-fusion", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--geometry-height-mode",
        choices=("joint-global", "asymmetric-local"),
        default="joint-global",
    )
    parser.add_argument("--log-interval", type=int, default=25)
    args = parser.parse_args()
    if args.epochs != 100 or args.checkpoint_mode != "inference":
        parser.error("DFC2023 ablation requires 100 epochs and inference-only weights")
    if args.geometry_height_fusion and (
        not args.local_structure or not args.context_detail_height
    ):
        parser.error("--geometry-height-fusion requires Local UniRep and ContextDetail")
    if args.geometry_height_mode != "joint-global" and (
        not args.geometry_height_fusion
    ):
        parser.error("--geometry-height-mode requires --geometry-height-fusion")
    if args.local_structure and (
        not args.unirep_pretrained or args.query_structure or args.geometry_interaction
    ):
        parser.error(
            "--local-structure requires UniRep without query/dual-scale geometry interaction"
        )
    if args.local_structure_weight <= 0 or not math.isfinite(
        args.local_structure_weight
    ):
        parser.error("--local-structure-weight must be finite and positive")
    if args.query_structure and (
        not args.unirep_pretrained or args.geometry_interaction
    ):
        parser.error("--query-structure requires UniRep without --geometry-interaction")
    if args.query_structure_weight <= 0 or not math.isfinite(
        args.query_structure_weight
    ):
        parser.error("--query-structure-weight must be finite and positive")
    if args.geometry_interaction and (
        not args.unirep_pretrained
        or not args.context_detail_height
        or (not args.ttci)
        or (not args.hcr)
        or (not args.slmhc)
    ):
        parser.error(
            "--geometry-interaction requires UniRep, ContextDetail and TTCI/HCR/SLMHC"
        )
    if args.unirep_pretrained:
        if args.baseline:
            parser.error(
                "UniRep experiment cannot be combined with the baseline preset"
            )
        if (
            args.fusion_type != "task-aware"
            or not args.task_split_decoder
            or args.hrbr
            or args.segmentation_refinement
        ):
            parser.error(
                "UniRep structure requires task-aware/TaskSplit and --no-hrbr/--no-segmentation-refinement"
            )
        if args.rgb_normalization != "imagenet" or args.height_head_aux_weight <= 0:
            parser.error(
                "UniRep structure requires ImageNet input and height intermediates"
            )
        if not 0 < args.unirep_lr_scale <= 1 or args.structure_branch_weight <= 0:
            parser.error("Invalid UniRep LR or branch supervision weight")
    if args.log_interval < 1:
        parser.error("--log-interval must be positive")
    if args.task_split_decoder and (
        args.fusion_type not in ("cmnext", "task-aware") or args.segmentation_refinement
    ):
        parser.error(
            "Task-split decoder requires cmnext/task-aware fusion without segmentation refinement"
        )
    if args.fusion_type == "task-aware" and (not args.task_split_decoder):
        parser.error("Task-aware fusion requires --task-split-decoder")
    if args.baseline or args.rgb_only or args.height_teacher or (args.variant != "B4"):
        parser.error(
            "DINOv2 branch supports RGB+SAR with B4 SAR depth, without legacy baseline/KD modes"
        )
    if (
        not args.pretrained
        or not args.sar_pretrained
        or args.rgb_normalization != "imagenet"
    ):
        parser.error(
            "DINOv2 and B4 SAR pretrained files plus ImageNet normalization are required"
        )
    if args.baseline:
        raise ValueError("This historical experiment is not part of the final release")
    if (
        not math.isfinite(args.height_head_aux_weight)
        or args.height_head_aux_weight < 0.0
    ):
        parser.error("--height-head-aux-weight must be finite and nonnegative.")
    if args.distribution_height and args.height_priority:
        parser.error("--distribution-height cannot be combined with --height-priority")
    if args.context_detail_height and (
        args.distribution_height or args.height_priority
    ):
        parser.error(
            "--context-detail-height cannot be combined with distribution/priority height"
        )
    if args.context_detail_height and args.height_head_aux_weight <= 0:
        parser.error(
            "--context-detail-height requires positive --height-head-aux-weight"
        )
    if args.building_affinity and (not args.context_detail_height):
        parser.error("--building-affinity requires --context-detail-height")
    if not math.isfinite(args.affinity_loss_weight) or args.affinity_loss_weight < 0:
        parser.error("--affinity-loss-weight must be finite and nonnegative")
    if args.height_teacher:
        if (
            args.baseline
            or not args.context_detail_height
            or args.height_head_aux_weight <= 0
        ):
            parser.error(
                "Height KD requires the ContextDetail H0 auxiliary path, not baseline mode"
            )
        if args.rgb_normalization != "imagenet":
            parser.error(
                "Height KD requires ImageNet RGB normalization for both models"
            )
        if not math.isfinite(args.height_kd_weight) or args.height_kd_weight <= 0:
            parser.error("--height-kd-weight must be positive and finite")
        if (
            not math.isfinite(args.height_kd_temperature)
            or args.height_kd_temperature <= 0
        ):
            parser.error("--height-kd-temperature must be positive and finite")
        if args.height_kd_warmup_epochs < 1 or args.teacher_microbatch < 1:
            parser.error("KD warmup and teacher microbatch must be positive integers")
    for weight in (args.height_range_cls_weight, args.height_range_expert_weight):
        if not math.isfinite(weight) or weight < 0:
            parser.error(
                "Height range supervision weights must be finite and nonnegative"
            )
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def collect_split_metadata(split_dir):
    (counts, hashes, members) = ({}, {}, {})
    for split in ("train", "val", "test"):
        with open(os.path.join(split_dir, f"{split}.txt"), "rb") as stream:
            payload = stream.read()
        files = [
            line.strip()
            for line in payload.decode("utf-8").splitlines()
            if line.strip()
        ]
        if not files or len(files) != len(set(files)):
            raise ValueError(f"{split} split is empty or contains duplicate filenames.")
        members[split] = set(files)
        counts[split] = len(files)
        hashes[split] = hashlib.sha256(payload).hexdigest()
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if members[left] & members[right]:
            raise ValueError(f"{left} and {right} share source tiles.")
    return (counts, hashes)


def dice_loss_with_logits(logits, target, eps=1e-06):
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    union = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()


def collect_sensor_metadata(
    split_dir, require_coverage=False, require_stratification=False
):
    counts = {}
    for split in ("train", "val", "test"):
        with open(os.path.join(split_dir, f"{split}.txt"), encoding="utf-8") as stream:
            names = [line.strip() for line in stream if line.strip()]
        counts[split] = dict(
            Counter((os.path.basename(name).split("_", 1)[0] for name in names))
        )
    unseen = {
        split: sorted(set(counts[split]) - set(counts["train"]))
        for split in ("val", "test")
    }
    if require_coverage and any(unseen.values()):
        raise ValueError(
            f"Evaluation sensors absent from training: {unseen}. Check the split protocol."
        )
    if require_stratification and any(
        (set(counts[split]) != set(counts["train"]) for split in ("val", "test"))
    ):
        raise ValueError(
            f"Sensor-stratified training requires the same sensors in train/val/test: {counts}."
        )
    return (counts, unseen)


def bce_dice_loss(logits, target):
    (logits, target) = (logits.float(), target.float())
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return 0.5 * bce + 0.5 * dice_loss_with_logits(logits, target)


def dilate_binary(target, width):
    kernel_size = 2 * width + 1
    return (
        F.max_pool2d(target.float(), kernel_size=kernel_size, stride=1, padding=width)
        > 0.5
    )


def masked_height_loss(prediction, target, valid):
    if not valid.any():
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(prediction[valid], target[valid], beta=1.0)


def final_height_loss(prediction, target, valid, kind="plain"):
    if kind == "plain":
        return masked_height_loss(prediction, target, valid)
    if kind != "convnext-weighted":
        raise ValueError(f"Unknown final height loss: {kind}")
    if not valid.any():
        return prediction[valid].sum() * 0.0
    (prediction, target) = (prediction[valid].float(), target[valid].float())
    weights = 1.0 + (target / 20.0).clamp(0.0, 4.0).sqrt()
    error = F.smooth_l1_loss(prediction, target, beta=1.0, reduction="none")
    return (weights * error).sum() / weights.sum().clamp_min(1.0)


def height_head_auxiliary_loss(output, target, valid):
    if "height_head" not in output:
        raise KeyError(
            "Initial height supervision requires return_intermediates=True and height_head."
        )
    prediction = output["height_head"].float()
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError(
            "Initial height, reference height, and validity mask must share full-resolution shape."
        )
    if not valid.any():
        return prediction[valid].sum() * 0.0
    if not torch.isfinite(prediction[valid]).all():
        raise FloatingPointError(
            "Nonfinite initial height prediction at valid building pixels."
        )
    return masked_height_loss(prediction, target.float(), valid)


def height_range_loss(gate_logits, target, valid):
    if valid.dim() == 4 and valid.size(1) == 1:
        valid = valid.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    if not valid.any():
        return gate_logits.float().sum() * 0.0
    labels = torch.zeros_like(target, dtype=torch.long)
    labels[target >= 3.0] = 1
    labels[target >= 20.0] = 2
    class_weight = gate_logits.new_tensor([1.9, 0.47, 0.64], dtype=torch.float32)
    return F.cross_entropy(
        gate_logits.float().permute(0, 2, 3, 1)[valid],
        labels[valid],
        weight=class_weight,
    )


def directional_tail_loss(prediction, target, valid):
    low = valid & (target < 3.0)
    high = valid & (target >= 20.0)
    losses = []
    if low.any():
        low_overestimate = F.relu(prediction[low] - target[low])
        losses.append(
            F.smooth_l1_loss(
                low_overestimate, torch.zeros_like(low_overestimate), beta=1.0
            )
        )
    if high.any():
        high_underestimate = F.relu(target[high] - prediction[high])
        losses.append(
            F.smooth_l1_loss(
                high_underestimate, torch.zeros_like(high_underestimate), beta=1.0
            )
        )
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def building_instance_height_loss(
    prediction,
    target,
    instance_map,
    valid,
    min_pixels=16,
    min_valid_ratio=0.5,
    height_bin_weights=None,
):
    """Give each reliable building equal weight in height supervision."""
    prediction = prediction.float()
    target = target.float()
    if prediction.dim() == 4 and prediction.size(1) == 1:
        prediction = prediction.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    if instance_map.dim() == 4 and instance_map.size(1) == 1:
        instance_map = instance_map.squeeze(1)
    if valid.dim() == 4 and valid.size(1) == 1:
        valid = valid.squeeze(1)
    instance_losses = []
    for batch_index in range(prediction.size(0)):
        ids = instance_map[batch_index].long()
        positive = ids > 0
        if not positive.any():
            continue
        max_id = int(ids.max().item())
        all_counts = torch.bincount(ids[positive], minlength=max_id + 1).to(
            prediction.dtype
        )
        reliable = positive & valid[batch_index]
        if not reliable.any():
            continue
        reliable_ids = ids[reliable]
        reliable_counts = torch.bincount(reliable_ids, minlength=max_id + 1).to(
            prediction.dtype
        )
        pred_sums = prediction.new_zeros(max_id + 1)
        target_sums = prediction.new_zeros(max_id + 1)
        pred_sums.scatter_add_(0, reliable_ids, prediction[batch_index][reliable])
        target_sums.scatter_add_(0, reliable_ids, target[batch_index][reliable])
        keep = reliable_counts >= float(min_pixels)
        keep &= reliable_counts / all_counts.clamp_min(1.0) >= min_valid_ratio
        keep[0] = False
        if not keep.any():
            continue
        pred_mean = pred_sums[keep] / reliable_counts[keep]
        target_mean = target_sums[keep] / reliable_counts[keep]
        absolute = F.smooth_l1_loss(pred_mean, target_mean, beta=1.0, reduction="none")
        relative = (pred_mean - target_mean).abs() / target_mean.clamp_min(1.0)
        per_instance = absolute + 0.25 * relative
        if height_bin_weights is None:
            instance_losses.append(per_instance.mean())
        else:
            raise ValueError("Historical height-bin reweighting is not included")
    if not instance_losses:
        return prediction.sum() * 0.0
    return torch.stack(instance_losses).mean()


def make_optimizer(
    model, lr, weight_decay, rgb_backbone_lr_scale=1.0, unirep_lr_scale=0.1
):
    if (
        not math.isfinite(rgb_backbone_lr_scale)
        or not 0.0 < rgb_backbone_lr_scale <= 1.0
    ):
        raise ValueError("--rgb-backbone-lr-scale must be in (0, 1].")
    parameters = [
        (name, value)
        for (name, value) in model.named_parameters()
        if value.requires_grad
    ]
    if rgb_backbone_lr_scale == 1.0 and (not hasattr(model, "unirep_structure")):
        return torch.optim.AdamW(
            [value for (_, value) in parameters], lr=lr, weight_decay=weight_decay
        )
    (rgb_parameters, cnn_parameters, task_parameters) = ([], [], [])
    for name, value in parameters:
        is_rgb_encoder = name.startswith("backbone.dino.")
        is_unirep = name.startswith(
            ("unirep_structure.cnn.", "unirep_structure.ade_decoder.")
        )
        (
            rgb_parameters
            if is_rgb_encoder
            else cnn_parameters if is_unirep else task_parameters
        ).append(value)
    if not rgb_parameters or not task_parameters:
        raise ValueError(
            "Grouped LR requires both RGB encoder and fusion/task parameters."
        )
    groups = [
        {
            "params": rgb_parameters,
            "lr": lr * rgb_backbone_lr_scale,
            "name": "rgb_backbone",
        },
        {"params": task_parameters, "lr": lr, "name": "fusion_and_tasks"},
    ]
    if cnn_parameters:
        groups.append(
            {
                "params": cnn_parameters,
                "lr": lr * unirep_lr_scale,
                "name": "unirep_pretrained",
            }
        )
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def make_poly_scheduler(
    optimizer, steps_per_epoch, epochs, warmup_epochs, power, min_lr_ratio
):
    total_steps = max(steps_per_epoch * epochs, 1)
    warmup_steps = max(steps_per_epoch * warmup_epochs, 0)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return 0.1 + 0.9 * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        polynomial = max(1.0 - progress, 0.0) ** power
        return min_lr_ratio + (1.0 - min_lr_ratio) * polynomial

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_cosine_scheduler(
    optimizer, steps_per_epoch, epochs, warmup_epochs, min_lr_ratio
):
    total_steps = max(steps_per_epoch * epochs, 1)
    warmup_steps = max(steps_per_epoch * warmup_epochs, 0)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return 0.1 + 0.9 * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ModelEMA:
    """Maintain an exponential moving average for stable validation weights."""

    def __init__(self, model, decay, bn_policy="average"):
        if bn_policy not in ("average", "copy"):
            raise ValueError("Unknown EMA BN policy")
        self.decay = decay
        self.bn_policy = bn_policy
        self.bn_buffers = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                for buffer_name, _ in module.named_buffers(recurse=False):
                    self.bn_buffers.add((name + "." if name else "") + buffer_name)
        self.shadow = {
            key: value.detach().clone() for (key, value) in model.state_dict().items()
        }
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        current = model.state_dict()
        for key, averaged in self.shadow.items():
            if key == "height_head.edges" or (
                self.bn_policy == "copy" and key in self.bn_buffers
            ):
                averaged.copy_(current[key].detach())
            elif torch.is_floating_point(averaged):
                averaged.mul_(self.decay).add_(
                    current[key].detach(), alpha=1.0 - self.decay
                )
            else:
                averaged.copy_(current[key].detach())

    @torch.no_grad()
    def copy_to(self, model):
        self.backup = {
            key: value.detach().clone() for (key, value) in model.state_dict().items()
        }
        state = model.state_dict()
        for key, averaged in self.shadow.items():
            state[key].copy_(averaged)

    @torch.no_grad()
    def restore(self, model):
        if self.backup is None:
            return
        state = model.state_dict()
        for key, original in self.backup.items():
            state[key].copy_(original)
        self.backup = None

    def model_state_dict(self):
        return {
            key: value.detach().cpu().clone() for (key, value) in self.shadow.items()
        }

    def state_dict(self):
        return {
            "decay": self.decay,
            "bn_policy": self.bn_policy,
            "shadow": self.model_state_dict(),
        }


@torch.no_grad()
def evaluate(model, loader, device, min_height, boundary_band_width, amp_enabled):
    model.eval()
    tp = fp = fn = 0.0
    contour_tp = contour_fp = contour_fn = 0.0
    boundary_tp = boundary_fp = boundary_fn = 0.0
    absolute_error = squared_error = pixels = 0.0
    range_correct = range_pixels = 0
    use_unirep = hasattr(model, "unirep_structure")
    branch_counts = {
        f"{b}_{t}": [0, 0, 0] for b in ("cnn", "semantic") for t in ("mask", "contour")
    }
    for batch in tqdm(
        loader, desc="val", ascii=True, leave=False, disable=not sys.stderr.isatty()
    ):
        rgb = batch["rgb"].to(device, non_blocking=True)
        sar = batch["sar"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        contour = batch["contour"].to(device, non_blocking=True)
        height = batch["height"].to(device, non_blocking=True)
        label_valid = batch["valid_height"].to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            output = model(rgb, sar, return_intermediates=use_unirep)
        pred_mask = torch.sigmoid(output["mask"]) > 0.5
        pred_contour = torch.sigmoid(output["contour"]) > 0.5
        gt_mask = mask > 0.5
        gt_contour = contour > 0.5
        if use_unirep:
            for key, counts in branch_counts.items():
                p = output["unirep_" + key] > 0
                target = gt_mask if key.endswith("_mask") else gt_contour
                for i, v in enumerate(
                    ((p & target).sum(), (p & ~target).sum(), (~p & target).sum())
                ):
                    counts[i] += v.item()
        valid_height = (
            gt_mask & label_valid & torch.isfinite(height) & (height > min_height)
        )
        if "height_range_logits" in output:
            confusion = range_confusion(output, height, valid_height)
            range_correct += confusion.diag().sum().item()
            range_pixels += confusion.sum().item()
        if valid_height.any():
            prediction = output["height"][valid_height].float()
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Non-finite height predictions in validation.")
            error = prediction.clamp_min(1e-06) - height[valid_height].float()
            absolute_error += error.abs().sum().item()
            squared_error += error.square().sum().item()
            pixels += valid_height.sum().item()
        tp += (pred_mask & gt_mask).sum().item()
        fp += (pred_mask & ~gt_mask).sum().item()
        fn += (~pred_mask & gt_mask).sum().item()
        contour_tp += (pred_contour & gt_contour).sum().item()
        contour_fp += (pred_contour & ~gt_contour).sum().item()
        contour_fn += (~pred_contour & gt_contour).sum().item()
        boundary_band = dilate_binary(gt_contour, boundary_band_width)
        boundary_tp += (pred_mask & gt_mask & boundary_band).sum().item()
        boundary_fp += (pred_mask & ~gt_mask & boundary_band).sum().item()
        boundary_fn += (~pred_mask & gt_mask & boundary_band).sum().item()
    precision = tp / max(tp + fp, 1.0)
    if pixels == 0:
        raise ValueError("Validation contains no valid building-height pixels.")
    recall = tp / max(tp + fn, 1.0)
    boundary_precision = boundary_tp / max(boundary_tp + boundary_fp, 1.0)
    boundary_recall = boundary_tp / max(boundary_tp + boundary_fn, 1.0)
    result = {
        "mae": absolute_error / max(pixels, 1.0),
        "height_range_accuracy": range_correct / range_pixels if range_pixels else None,
        "rmse": math.sqrt(squared_error / max(pixels, 1.0)),
        "f1": 2 * precision * recall / max(precision + recall, 1e-08),
        "precision": precision,
        "recall": recall,
        "contour_f1": 2
        * contour_tp
        / max(2 * contour_tp + contour_fp + contour_fn, 1.0),
        "boundary_f1": 2
        * boundary_precision
        * boundary_recall
        / max(boundary_precision + boundary_recall, 1e-08),
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
    }
    if use_unirep:
        result.update(
            {
                f"{k}_f1": 2 * v[0] / max(2 * v[0] + v[1] + v[2], 1)
                for (k, v) in branch_counts.items()
            }
        )
    return result


def save_checkpoint(
    path, model, ema, optimizer, scheduler, scaler, epoch, metrics, args
):
    if args.checkpoint_mode == "inference":
        torch.save(
            {
                "format": "hdbfnet_inference_v1",
                "model_name": args.model_name,
                "model_config": {
                    "ablation_config": args.ablation_config,
                    "sar_channels": 1,
                    "geometry_height_fusion": getattr(
                        args, "geometry_height_fusion", False
                    ),
                    "geometry_height_mode": getattr(
                        args, "geometry_height_mode", "joint-global"
                    ),
                    "local_structure": args.local_structure,
                    "query_structure": False,
                    "geometry_interaction": False,
                    "unirep_structure": bool(args.unirep_pretrained),
                    "unirep_provenance": getattr(args, "unirep_provenance", None),
                    "architecture": model.architecture,
                    "fusion_type": args.fusion_type,
                    "segmentation_refinement": False,
                    "task_split_decoder": args.task_split_decoder,
                    "dinov2_source_commit": SOURCE_COMMIT,
                    "dinov2_pretrained_sha256": getattr(
                        model, "pretrained_sha256", None
                    ),
                    "gradient_checkpointing": bool(args.gradient_checkpointing),
                    "variant": args.variant,
                    "decoder_dim": args.decoder_dim,
                    "use_hrbr": bool(args.hrbr),
                    "use_hcr": bool(args.hcr),
                    "use_slmhc": bool(args.slmhc),
                    "use_ttci": bool(args.ttci),
                    "use_bihm": bool(args.bihm),
                    "task_channels": args.task_channels,
                    "height_priority": bool(getattr(args, "height_priority", False)),
                    "distribution_height": bool(
                        getattr(args, "distribution_height", False)
                    ),
                    "context_detail_height": bool(
                        getattr(args, "context_detail_height", False)
                    ),
                    "building_affinity": bool(
                        getattr(args, "building_affinity", False)
                    ),
                    "rgb_only": bool(getattr(args, "rgb_only", False)),
                },
                "data_config": {
                    "rgb_normalization": getattr(args, "rgb_normalization", "none"),
                    "sar_clip": args.sar_clip,
                    "min_height": args.min_height,
                    "split_counts": args.split_counts,
                    "split_hashes": args.split_hashes,
                    "sensor_counts": getattr(args, "sensor_counts", {}),
                },
                "evaluation_weights": "EMA" if args.ema_decay > 0 else "raw",
                "epoch": epoch,
                "metrics": metrics,
                "args": vars(args),
                "model_state_dict": ema.model_state_dict(),
            },
            path,
        )
        return
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": ema.model_state_dict(),
            "raw_model_state_dict": {
                key: value.detach().cpu().clone()
                for (key, value) in model.state_dict().items()
            },
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": metrics,
            "args": vars(args),
            "model_config": {
                "architecture": model.architecture,
                "fusion_type": args.fusion_type,
                "segmentation_refinement": False,
                "task_split_decoder": args.task_split_decoder,
                "dinov2_source_commit": SOURCE_COMMIT,
                "geometry_height_fusion": getattr(
                    args, "geometry_height_fusion", False
                ),
                "geometry_height_mode": getattr(
                    args, "geometry_height_mode", "joint-global"
                ),
            },
        },
        path,
    )


def save_epoch_checkpoints(
    run_dir,
    model,
    ema,
    optimizer,
    scheduler,
    scaler,
    epoch,
    metrics,
    args,
    best_metrics,
):
    if metrics["mae"] < best_metrics["mae"]:
        save_checkpoint(
            os.path.join(run_dir, "checkpoint_best_mae.pth"),
            model,
            ema,
            optimizer,
            scheduler,
            scaler,
            epoch,
            metrics,
            args,
        )
        best_metrics["mae"] = metrics["mae"]


def main():
    args = parse_args()
    if args.data_seed is None:
        args.data_seed = args.seed
    seed_everything(args.seed)
    base_dir = args.base_dir or os.environ.get("DFC23_BASE", "D:\\znx\\data\\dfc23")
    split_dir = args.split_dir or os.path.join(base_dir, "dfc23_newsplit")
    (args.base_dir, args.split_dir) = (base_dir, split_dir)
    (args.split_counts, args.split_hashes) = collect_split_metadata(split_dir)
    (args.sensor_counts, args.unseen_evaluation_sensors) = collect_sensor_metadata(
        split_dir, args.require_sensor_coverage, args.require_sensor_stratification
    )
    verify_reference(args)
    apply_case(args)
    train_split = os.path.join(split_dir, "train.txt")
    val_split = os.path.join(split_dir, "val.txt")
    run_dir = os.path.join(
        args.checkpoint_dir, datetime.now().strftime("%y%m%d_%H%M%S")
    )
    os.makedirs(run_dir, exist_ok=True)
    with open(
        os.path.join(run_dir, "effective_config.json"), "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "args": vars(args),
                "height_validity": "footprint & source validity & finite height & height > min_height",
                "evaluation_weights": "EMA" if args.ema_decay > 0 else "raw",
            },
            stream,
            indent=2,
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and device.type == "cuda"
    model = AblationDinoHDBF(
        ablation_case=args.ablation_case,
        sar_channels=1,
        fusion_type=args.fusion_type,
        segmentation_refinement=False,
        task_split_decoder=args.task_split_decoder,
        gradient_checkpointing=args.gradient_checkpointing,
        geometry_interaction=False,
        variant=args.variant,
        decoder_dim=args.decoder_dim,
        use_hrbr=args.hrbr,
        use_hcr=args.hcr,
        use_slmhc=args.slmhc,
        use_ttci=args.ttci,
        task_channels=args.task_channels,
        height_priority=False,
        distribution_height=False,
        context_detail_height=args.context_detail_height,
        building_affinity=args.building_affinity,
    )
    print(f"Fusion: {args.fusion_type}; architecture: {model.architecture}")
    print(f"Full-resolution segmentation refinement: {False}")
    print(
        f"Decoder: {('task-split-pyramid' if args.task_split_decoder else 'shared-segformer')}"
    )
    if args.pretrained:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(
                f"DINOv2 checkpoint not found: {args.pretrained}. Upload the official dinov2_vitb14_pretrain.pth file."
            )
        model.init_pretrained(
            args.pretrained, require_complete=args.require_complete_pretrained
        )
    if args.ablation_config["sar"]:
        model.init_sar_pretrained(args.sar_pretrained)
    if args.unirep_pretrained:
        from hdbf.unirep_structure import UniRepStructure

        with torch.random.fork_rng(devices=[]):
            model.unirep_structure = UniRepStructure(
                model.backbone.channels,
                args.decoder_dim,
                query_structure=False,
                local_structure=args.local_structure,
            )
        args.unirep_provenance = model.unirep_structure.load_pretrained(
            args.unirep_pretrained
        )
        with open(
            os.path.join(run_dir, "unirep_pretrained_manifest.json"),
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(args.unirep_provenance, stream, indent=2)
    if args.geometry_height_fusion:
        from hdbf.geometry_height import GeometryHeightFusion

        with torch.random.fork_rng(devices=[]):
            model.height_geometry_fusion = GeometryHeightFusion(
                args.decoder_dim, mode=args.geometry_height_mode
            )
    model.to(device)
    teacher = None
    train_set = DFC2023Dataset(
        base_dir,
        train_split,
        train=True,
        sar_clip=args.sar_clip,
        rgb_jitter_probability=args.rgb_jitter_probability,
        rgb_jitter_strength=args.rgb_jitter_strength,
        rgb_normalization=args.rgb_normalization,
    )
    val_set = DFC2023Dataset(
        base_dir,
        val_split,
        train=False,
        sar_clip=args.sar_clip,
        rgb_normalization=args.rgb_normalization,
    )
    bihm_bin_weights = None
    train_sampler = None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.data_seed),
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.data_seed + 1),
        worker_init_fn=seed_worker,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = make_optimizer(
        model,
        args.lr,
        args.weight_decay,
        args.rgb_backbone_lr_scale,
        args.unirep_lr_scale,
    )
    if args.lr_schedule == "cosine":
        scheduler = make_cosine_scheduler(
            optimizer,
            len(train_loader),
            args.epochs,
            args.warmup_epochs,
            args.min_lr_ratio,
        )
    else:
        scheduler = make_poly_scheduler(
            optimizer,
            len(train_loader),
            args.epochs,
            args.warmup_epochs,
            args.poly_power,
            args.min_lr_ratio,
        )
    scaler = GradScaler(enabled=amp_enabled)
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in [0, 1).")
    if not 0.0 <= args.min_lr_ratio < 1.0:
        raise ValueError("--min-lr-ratio must be in [0, 1).")
    ema = ModelEMA(model, args.ema_decay, args.ema_bn_policy)
    print(f"Run dir: {run_dir}")
    print(f"Effective configuration: {os.path.join(run_dir, 'effective_config.json')}")
    model_suffix = "-HRBR" if args.hrbr else ""
    if args.hcr:
        model_suffix += "-HCR"
    if args.slmhc:
        model_suffix += "-SLMHC"
    if args.bihm:
        model_suffix += "-BIHM"
    if args.ttci:
        model_suffix += "-TTCI"
    print(
        f"Model: DINOv2-ViT-B14 + SAR-{args.variant}{model_suffix} | Input: RGB 3ch + SAR 1ch"
    )
    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")
    print(
        f"Batch size: {args.batch_size} | LR: {args.lr} | AMP: {amp_enabled} | Seed: {args.seed} | Data seed: {args.data_seed}"
    )
    print(f"Pretrained DINOv2: {args.pretrained}; source={SOURCE_COMMIT}")
    print(
        "Model input: "
        + (
            "RGB only; SAR encoder/fusion/HRBR stem absent"
            if args.rgb_only
            else "RGB+SAR"
        )
    )
    print(f"RGB normalization: {args.rgb_normalization}")
    print(f"Split sensor counts: {args.sensor_counts}")
    if any(args.unseen_evaluation_sensors.values()):
        print(
            f"WARNING: evaluation includes sensors absent from training: {args.unseen_evaluation_sensors}"
        )
    for group in optimizer.param_groups:
        print(
            f"LR group: {group.get('name', 'all')} | Base LR: {group.get('initial_lr', group['lr']):.6g} | Parameters: {sum((value.numel() for value in group['params'])) / 1000000.0:.3f}M"
        )
    print(
        f"Train-only RGB brightness/contrast jitter: probability={args.rgb_jitter_probability} | strength={args.rgb_jitter_strength}; SAR and labels unchanged"
    )
    print(f"Boundary validation band width: {args.boundary_band_width}px")
    if args.lr_schedule == "cosine":
        print(
            f"LR schedule: cosine | min_ratio={args.min_lr_ratio} | warmup={args.warmup_epochs} epochs"
        )
    else:
        print(
            f"LR schedule: floor slow polynomial | power={args.poly_power} | min_ratio={args.min_lr_ratio} | warmup={args.warmup_epochs} epochs"
        )
    print(f"EMA: enabled | decay={args.ema_decay}")
    print(f"EMA BN policy: {args.ema_bn_policy}; no post-training BN calibration")
    print(f"Checkpoint mode: {args.checkpoint_mode}")
    if args.local_structure:
        print(
            "Structure: 192-channel fusion, native CNN bypass, quarter/half/native dense reconstruction, two task-specific local affinity steps; no query matching or sparse point updates."
        )
        print(
            f"Local auxiliary weight={args.local_structure_weight}: averaged coarse/half BCE+Dice + 0.25 boundary-balanced affinity CE; final task losses and LR unchanged. Detached height descriptors with learned relevance gate; ramp epochs20..40."
        )
    if args.geometry_height_fusion:
        print(
            f"Geometry-height mode: {args.geometry_height_mode}; new-route CNN gradients={('stopped' if args.geometry_height_mode == 'asymmetric-local' else 'joint')}; geometry losses and original task interactions remain trainable"
        )
        print(
            "DFC23 supervision: fixed HCR class weights=(1.90,0.47,0.64), ranges=(3m,20m); FP32 range CE; original valid pixels, SmoothL1 and directional tail/BIHM losses"
        )
    print(
        f"Ablation: {args.ablation_case}; full reference: {args.ablation_config['reference_full']}"
    )
    print("Checkpoint selection: best validation MAE, inference weights only")
    print(f"Split dir: {split_dir} | Split counts: {args.split_counts}")
    print(
        "Height validity: footprint & source validity & finite height & height > min_height"
    )
    print(
        f"Final height loss: {args.final_height_loss} | global weight={args.height_weight}; auxiliary losses unchanged; checkpoint selection uses unweighted validation MAE"
    )
    if args.final_height_loss == "convnext-weighted":
        print(
            "Final height pixel weight: 1 + sqrt(clamp(GT_height / 20m, 0, 4)); normalized by the sum of valid pixel weights"
        )
    print(
        f"Initial height auxiliary supervision: weight={args.height_head_aux_weight:g} | full-resolution H0 | same valid building pixels"
    )
    print(
        f"Model params: {sum((p.numel() for p in model.parameters())) / 1000000.0:.1f}M"
    )
    if args.building_affinity:
        print(
            f"Building affinity: predicted same-instance aggregation; 25 samples per scale; feature spans=5/17/33; class-balanced supervision={args.affinity_loss_weight}; pure valid instance cells only; no GT in forward"
        )
    if args.context_detail_height:
        print(
            "Context-detail height: continuous base + bounded +/-3m local detail; context spans=5/17/33; no height-range classifier"
        )
    print(
        f"Trainable params: {sum((p.numel() for p in trainable_parameters)) / 1000000.0:.3f}M"
    )
    best_metrics = {
        "mae": float("inf"),
        "f1": 0.0,
        "contour_f1": 0.0,
        "boundary_f1": 0.0,
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.local_structure:
            model.unirep_structure.local_decoder.set_epoch(epoch)
        kd_weight = (
            args.height_kd_weight * min(epoch / args.height_kd_warmup_epochs, 1.0)
            if teacher is not None
            else 0.0
        )
        kd_sums = dict(
            pixels=0, active=0, weight=0.0, teacher_ae=0.0, head_ae=0.0, loss=0.0
        )
        sums = {
            "local_structure": 0.0,
            "query_structure": 0.0,
            "structure_branches": 0.0,
            "loss": 0.0,
            "mask": 0.0,
            "contour": 0.0,
            "height": 0.0,
            "height_head_aux": 0.0,
            "affinity": 0.0,
            "affinity_pos_pairs": 0.0,
            "affinity_neg_pairs": 0.0,
            "range_cls": 0.0,
            "range_expert": 0.0,
            "range_accuracy": 0.0,
            "bg": 0.0,
            "hcr_range": 0.0,
            "tail": 0.0,
            "bihm": 0.0,
        }
        if args.geometry_height_fusion:
            sums.update(height_geometry_residual=0.0, height_geometry_gate=0.0)
            sums.update({f"height_geometry_scale_{i}": 0.0 for i in range(4)})
        tail_weight = args.slmhc_tail_weight * min(
            max((epoch - 1) / max(args.slmhc_warmup_epochs, 1), 0.0), 1.0
        )
        bihm_weight = args.bihm_weight * min(
            max((epoch - 1) / max(args.bihm_warmup_epochs, 1), 0.0), 1.0
        )
        batches = 0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            ascii=True,
            disable=not sys.stderr.isatty(),
        )
        for batch in progress:
            rgb = batch["rgb"].to(device, non_blocking=True)
            sar = batch["sar"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            contour = batch["contour"].to(device, non_blocking=True)
            height = batch["height"].to(device, non_blocking=True)
            label_valid = batch["valid_height"].to(device, non_blocking=True)
            instance = batch["instance"].to(device, non_blocking=True)
            teacher_height = (
                predict_teacher(teacher, rgb, args.teacher_microbatch)
                if teacher is not None
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                output = (
                    model(rgb, sar, return_intermediates=True)
                    if args.height_head_aux_weight > 0.0
                    else model(rgb, sar)
                )
                valid_height = (
                    (mask > 0.5)
                    & label_valid
                    & torch.isfinite(height)
                    & (height > args.min_height)
                )
                loss_height = final_height_loss(
                    output["height"], height, valid_height, args.final_height_loss
                )
                loss_range_cls = loss_height.new_zeros(())
                loss_affinity = loss_height.new_zeros(())
                affinity_pos_pairs = affinity_neg_pairs = 0
                if args.building_affinity:
                    from hdbf.dfc_building_affinity import affinity_supervision

                    (loss_affinity, affinity_pos_pairs, affinity_neg_pairs, _, _) = (
                        affinity_supervision(
                            output["building_affinity_logits"], instance, valid_height
                        )
                    )
                loss_range_expert = loss_height.new_zeros(())
                range_accuracy = loss_height.new_zeros(())
                loss_height_head_aux = (
                    height_head_auxiliary_loss(output, height, valid_height)
                    if args.height_head_aux_weight > 0.0
                    else loss_height.new_zeros(())
                )
                loss_hcr_range = (
                    height_range_loss(output["hcr_gate"], height, valid_height)
                    if args.hcr
                    else output["height"].sum() * 0.0
                )
                loss_tail = (
                    directional_tail_loss(output["height"], height, valid_height)
                    if args.slmhc
                    else output["height"].sum() * 0.0
                )
                loss_bihm = (
                    building_instance_height_loss(
                        output["height"],
                        height,
                        instance,
                        valid_height,
                        args.bihm_min_pixels,
                        args.bihm_min_valid_ratio,
                        height_bin_weights=bihm_bin_weights,
                    )
                    if args.bihm
                    else output["height"].sum() * 0.0
                )
                loss_mask = bce_dice_loss(output["mask"], mask)
                loss_contour = bce_dice_loss(output["contour"], contour)
                loss_local = loss_mask.new_zeros(())
                if args.local_structure:
                    from hdbf.local_structure import local_structure_loss

                    loss_local = local_structure_loss(
                        output, mask, contour, args.mask_weight, args.contour_weight
                    )
                loss_query = loss_mask.new_zeros(())
                loss_branches = loss_mask.new_zeros(())
                if args.unirep_pretrained:
                    loss_branches = (
                        sum(
                            (
                                args.mask_weight
                                * bce_dice_loss(output[f"unirep_{branch}_mask"], mask)
                                + args.contour_weight
                                * bce_dice_loss(
                                    output[f"unirep_{branch}_contour"], contour
                                )
                                for branch in ("cnn", "semantic")
                            )
                        )
                        / 2
                    )
                background = mask <= 0.5
                loss_bg = (
                    output["height"][background].abs().mean()
                    if background.any()
                    else output["height"].sum() * 0.0
                )
                loss = (
                    args.mask_weight * loss_mask
                    + args.contour_weight * loss_contour
                    + args.height_weight * loss_height
                    + args.hcr_range_weight * loss_hcr_range
                    + tail_weight * loss_tail
                    + bihm_weight * loss_bihm
                    + args.bg_height_weight * loss_bg
                )
                if args.height_head_aux_weight > 0.0:
                    loss = loss + args.height_head_aux_weight * loss_height_head_aux
                loss = loss + args.affinity_loss_weight * loss_affinity
                loss = loss + args.structure_branch_weight * loss_branches
                loss = loss + args.query_structure_weight * loss_query
                loss = loss + args.local_structure_weight * loss_local
                if teacher_height is not None:
                    (loss_kd, kd_stats) = height_distillation_loss(
                        output["height_head"],
                        teacher_height,
                        height,
                        valid_height,
                        args.height_kd_temperature,
                    )
                    loss = loss + kd_weight * loss_kd
                    kd_sums["loss"] += loss_kd.detach().item()
                    for key, value in kd_stats.items():
                        kd_sums[key] += value
            if not torch.isfinite(loss.detach()):
                bad = {
                    k: int((~torch.isfinite(v.detach())).sum())
                    for (k, v) in output.items()
                    if isinstance(v, torch.Tensor)
                }
                raise FloatingPointError(
                    f"Nonfinite loss: epoch={epoch}, batch={batches + 1}, output counts={bad}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
                ema.update(model)
            batches += 1
            values = {
                "local_structure": loss_local.item(),
                "structure_branches": loss_branches.item(),
                "query_structure": loss_query.item(),
                "loss": loss.item(),
                "mask": loss_mask.item(),
                "contour": loss_contour.item(),
                "height": loss_height.item(),
                "height_head_aux": loss_height_head_aux.item(),
                "affinity": loss_affinity.item(),
                "affinity_pos_pairs": affinity_pos_pairs,
                "affinity_neg_pairs": affinity_neg_pairs,
                "range_cls": loss_range_cls.item(),
                "range_expert": loss_range_expert.item(),
                "range_accuracy": range_accuracy.item(),
                "bg": loss_bg.item(),
                "hcr_range": loss_hcr_range.item(),
                "tail": loss_tail.item(),
                "bihm": loss_bihm.item(),
            }
            if args.geometry_height_fusion:
                values["height_geometry_residual"] = output[
                    "height_geometry_relative_residual"
                ].item()
                values["height_geometry_gate"] = output[
                    "height_geometry_gate_mean"
                ].item()
                scale_weights = (
                    output["height_geometry_scale_weights"].mean((0, 2, 3)).tolist()
                )
                values.update(
                    {
                        f"height_geometry_scale_{i}": value
                        for (i, value) in enumerate(scale_weights)
                    }
                )
            for key, value in values.items():
                sums[key] += value
            progress.set_postfix(
                loss=f"{loss.item():.3f}",
                h=f"{loss_height.item():.3f}",
                m=f"{loss_mask.item():.3f}",
            )
            if not sys.stderr.isatty() and (
                batches == 1 or batches % args.log_interval == 0
            ):
                print(
                    f"Epoch {epoch}/{args.epochs} batch {batches}/{len(train_loader)} loss={loss.item():.4f} mask={loss_mask.item():.4f} contour={loss_contour.item():.4f} query={loss_query.item():.4f} local={loss_local.item():.4f}",
                    flush=True,
                )
        if args.building_affinity and (
            sums["affinity_pos_pairs"] == 0 or sums["affinity_neg_pairs"] == 0
        ):
            raise RuntimeError(
                "Affinity supervision lacks same/different-building pairs; inspect instance targets before continuing"
            )
        if teacher is not None:
            n = max(kd_sums["pixels"], 1)
            print(
                f"Height KD train: loss={kd_sums['loss'] / max(batches, 1):.4f} weight={kd_weight:.4f} teacher_mae={kd_sums['teacher_ae'] / n:.4f} head_mae={kd_sums['head_ae'] / n:.4f} active_fraction={kd_sums['active'] / n:.4f} mean_reliability={kd_sums['weight'] / n:.4f} pixels={kd_sums['pixels']}"
            )
        ema.copy_to(model)
        try:
            metrics = evaluate(
                model,
                val_loader,
                device,
                args.min_height,
                args.boundary_band_width,
                amp_enabled,
            )
        finally:
            ema.restore(model)
        metrics.update({key: value / max(batches, 1) for (key, value) in sums.items()})
        metrics["lr"] = max((group["lr"] for group in optimizer.param_groups))
        metrics["rgb_lr"] = next(
            (
                group["lr"]
                for group in optimizer.param_groups
                if group.get("name") == "rgb_backbone"
            ),
            metrics["lr"],
        )
        if args.geometry_height_fusion:
            print(
                f"Geometry-height train: relative_feature_residual={metrics['height_geometry_residual']:.4f}; gate={metrics['height_geometry_gate']:.4f}; scale_weights={[round(metrics[f'height_geometry_scale_{i}'], 4) for i in range(4)]}"
            )
        print(
            f"Epoch {epoch}: loss={metrics['loss']:.4f} mask_loss={metrics['mask']:.4f} contour_loss={metrics['contour']:.4f} height_loss={metrics['height']:.4f} height_head_aux_loss={metrics['height_head_aux']:.4f} height_head_aux_weight={args.height_head_aux_weight:.4f} affinity_loss={metrics['affinity']:.4f} affinity_pos_pairs={metrics['affinity_pos_pairs']:.0f} affinity_neg_pairs={metrics['affinity_neg_pairs']:.0f} range_cls={metrics['range_cls']:.4f} range_expert={metrics['range_expert']:.4f} range_accuracy_batch_mean={metrics['range_accuracy']:.4f} hcr_range_loss={metrics['hcr_range']:.4f} tail_loss={metrics['tail']:.4f} tail_weight={tail_weight:.4f} bihm_loss={metrics['bihm']:.4f} bihm_weight={bihm_weight:.4f} bg_loss={metrics['bg']:.4f} | val_f1={metrics['f1']:.4f} val_contour_f1={metrics['contour_f1']:.4f} val_boundary_f1={metrics['boundary_f1']:.4f} val_boundary_recall={metrics['boundary_recall']:.4f} val_mae={metrics['mae']:.4f} val_rmse={metrics['rmse']:.4f} lr={metrics['lr']:.6g} rgb_lr={metrics['rgb_lr']:.6g}"
        )
        if args.unirep_pretrained:
            print(
                f"Joint-model paths (not retrained ablations): CNN mask/contour={metrics['cnn_mask_f1']:.4f}/{metrics['cnn_contour_f1']:.4f}; semantic mask/contour={metrics['semantic_mask_f1']:.4f}/{metrics['semantic_contour_f1']:.4f}"
            )
        save_epoch_checkpoints(
            run_dir,
            model,
            ema,
            optimizer,
            scheduler,
            scaler,
            epoch,
            metrics,
            args,
            best_metrics,
        )
        with open(
            os.path.join(run_dir, "training_metrics.jsonl"), "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps({"epoch": epoch, **metrics}) + "\n")


if __name__ == "__main__":
    main()
