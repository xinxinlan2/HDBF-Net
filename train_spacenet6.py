import argparse
import json
import math
import os
import random
import sys
import bootstrap
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from hdbf.dataset_spacenet6 import SpaceNet6HeightDataset, split_metadata
from semseg.models.cmnext_mtl import CMNeXtMTL


def parse_args():
    parser = argparse.ArgumentParser(
        "Train the standard HDBF-Net on SpaceNet6 height labels"
    )
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--encoder", choices=["mit", "dinov2"], default="mit")
    parser.add_argument(
        "--fusion-type", choices=["cmnext", "task-aware"], default="cmnext"
    )
    parser.add_argument("--task-split-decoder", action="store_true")
    parser.set_defaults(structure_isolation=False)
    parser.add_argument("--building-affinity", action="store_true")
    parser.add_argument("--affinity-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--sar-pretrained", default=None)
    parser.add_argument("--unirep-pretrained", default=None)
    parser.add_argument("--unirep-lr-scale", type=float, default=0.1)
    parser.add_argument("--structure-branch-weight", type=float, default=0.2)
    parser.add_argument(
        "--local-structure", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--local-structure-weight", type=float, default=0.2)
    parser.add_argument(
        "--geometry-height-fusion",
        action="store_true",
        help="Use predicted multiscale CNN geometry before the initial height head",
    )
    parser.add_argument(
        "--geometry-height-mode",
        choices=["joint-global", "asymmetric-local"],
        default="joint-global",
        help="Legacy GN/joint gradients or local norm/reference geometry",
    )
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--context-detail-height", action="store_true")
    parser.add_argument("--save-best-mae-only", action="store_true")
    parser.add_argument(
        "--save-task-best",
        action="store_true",
        help="Also retain best mask and contour checkpoints in best-MAE-only runs",
    )
    parser.add_argument(
        "--contour-objective",
        choices=["legacy", "balanced", "bce-dice"],
        default="legacy",
    )
    parser.add_argument(
        "--normalize-rgb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply ImageNet RGB standardization after scaling to [0, 1].",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="SpaceNet6 source root containing AOI_11_Rotterdam. Defaults to the parent of base-dir.",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints_spacenet6/hdbfnet_b4_standard"
    )
    parser.add_argument(
        "--pretrained", default="checkpoints/pretrained/segformer/pytorch_model.bin"
    )
    parser.set_defaults(finetune_checkpoint=None)
    parser.set_defaults(height_rescue=False)
    parser.set_defaults(roof_context_refiner=False)
    parser.add_argument(
        "--height-guidance-stop-gradient",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use footprint and contour predictions as height guidance without sending height-loss gradients into their prediction heads.",
    )
    parser.add_argument("--variant", default="B4", choices=["B2", "B4", "B5"])
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
        default=True,
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
    parser.set_defaults(rgb_anchored_fusion=False)
    parser.set_defaults(rgb_anchored_hrbr=False)
    parser.add_argument("--auxiliary-initial-strength", type=float, default=0.02)
    parser.set_defaults(structure_only_sar=False)
    parser.set_defaults(directional_ttci=False)
    parser.set_defaults(rgb_height_detail_refiner=False)
    parser.add_argument("--rgb-height-detail-channels", type=int, default=48)
    parser.add_argument(
        "--rgb-height-detail-initial-strength", type=float, default=0.02
    )
    parser.add_argument("--rgb-height-detail-max-residual", type=float, default=4.0)
    parser.add_argument("--structure-fusion-initial-strength", type=float, default=0.1)
    parser.add_argument(
        "--height-range-boundaries", type=float_tuple, default=(3.0, 20.0)
    )
    parser.add_argument(
        "--hcr-residual-scales", type=float_tuple, default=(6.0, 3.0, 6.0)
    )
    parser.add_argument(
        "--slmhc-knots",
        type=float_tuple,
        default=(0.0, 3.0, 6.0, 10.0, 20.0, 40.0, 100.0),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument(
        "--val-inference", choices=["whole", "sliding"], default="whole"
    )
    parser.add_argument("--val-window", type=int, default=768)
    parser.add_argument("--val-stride", type=int, default=384)
    parser.set_defaults(experiment="hdbf")
    parser.add_argument("--hdbf-reference-checkpoint", default=None)
    parser.set_defaults(building_mix_probability=0.0)
    parser.set_defaults(standard_tune100=False)
    parser.set_defaults(standard_scale_augmentation=False)
    parser.add_argument(
        "--height-head-kind",
        choices=[
            "lightweight",
            "standard",
            "standard_concat",
            "standard_local",
            "standard_dual",
        ],
        default="standard",
    )
    parser.add_argument(
        "--height-local-pretrained",
        default=None,
        help="Official torchvision ResNet50 state dict; required for standard_local/standard_dual fresh training",
    )
    parser.add_argument("--stop-after-epoch", type=int, default=None)
    parser.set_defaults(resume=None)
    parser.set_defaults(use_multiscale_height_head=False)
    parser.set_defaults(use_height_multiscale_adapter=False)
    parser.add_argument("--ttci-reference-checkpoint", default=None)
    parser.set_defaults(ttci_bypass=False)
    parser.add_argument("--geometry-checkpoint", default=None)
    parser.add_argument("--paired-control-checkpoint", default=None)
    parser.add_argument("--paired-hdbf-checkpoint", default=None)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--crops-per-image", type=int, default=1)
    parser.add_argument("--min-building-pixels", type=int, default=64)
    parser.add_argument(
        "--positive-crop-probability",
        type=float,
        default=0.7,
        help="Probability of retrying a crop until it contains enough building pixels.",
    )
    parser.add_argument(
        "--targeted-crop-probability",
        type=float,
        default=0.0,
        help="Probability of centering a crop on a large building in the selected height interval. Zero preserves the original sampler.",
    )
    parser.add_argument("--targeted-height-min", type=float, default=8.0)
    parser.add_argument("--targeted-height-max", type=float, default=20.0)
    parser.add_argument("--targeted-area-min-m2", type=float, default=500.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use page-locked DataLoader memory. Keep disabled if pinning is unstable.",
    )
    parser.add_argument("--lr", type=float, default=6e-05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--rgb-backbone-lr-scale",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for the pretrained RGB encoder.",
    )
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--poly-power", type=float, default=0.35)
    parser.add_argument("--lr-schedule", choices=["cosine", "poly"], default="poly")
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--ema-bn-policy",
        choices=["average", "copy"],
        default="average",
        help="Copy BN running statistics without a second EMA, or retain legacy averaging.",
    )
    parser.add_argument("--mask-weight", type=float, default=1.0)
    parser.add_argument("--contour-weight", type=float, default=0.4)
    parser.add_argument(
        "--contour-structure-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress contour responses outside a soft dilated footprint support.",
    )
    parser.add_argument("--contour-gate-kernel", type=int, default=9)
    parser.add_argument("--contour-gate-floor", type=float, default=0.15)
    parser.add_argument("--contour-gate-initial-strength", type=float, default=0.2)
    parser.add_argument("--contour-bce-weight", type=float, default=0.4)
    parser.add_argument("--contour-tversky-weight", type=float, default=0.4)
    parser.add_argument("--contour-far-weight", type=float, default=0.2)
    parser.add_argument("--contour-consistency-weight", type=float, default=0.1)
    parser.add_argument("--contour-tversky-fp-weight", type=float, default=0.7)
    parser.add_argument("--contour-tversky-fn-weight", type=float, default=0.3)
    parser.add_argument("--contour-far-band-width", type=int, default=3)
    parser.add_argument("--contour-hard-negative-fraction", type=float, default=0.1)
    parser.add_argument("--contour-consistency-kernel", type=int, default=5)
    parser.add_argument(
        "--checkpoint-min-mask-f1",
        type=float,
        default=0.7,
        help="Minimum validation mask F1 for the balanced-MAE checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-min-contour-f1",
        type=float,
        default=0.3,
        help="Minimum validation contour F1 for the balanced-MAE checkpoint.",
    )
    parser.add_argument("--boundary-band-width", type=int, default=3)
    parser.add_argument("--height-weight", type=float, default=1.0)
    parser.add_argument("--height-head-weight", type=float, default=0.0)
    parser.add_argument("--height-correction-guard-weight", type=float, default=0.0)
    parser.add_argument("--height-correction-guard-warmup-epochs", type=int, default=10)
    parser.add_argument("--hcr-range-weight", type=float, default=0.05)
    parser.add_argument("--slmhc-tail-weight", type=float, default=0.05)
    parser.add_argument("--slmhc-warmup-epochs", type=int, default=50)
    parser.add_argument("--bihm-weight", type=float, default=0.05)
    parser.add_argument("--bihm-warmup-epochs", type=int, default=50)
    parser.add_argument("--bihm-min-pixels", type=int, default=16)
    parser.add_argument("--bihm-min-valid-ratio", type=float, default=0.5)
    parser.add_argument("--bg-height-weight", type=float, default=0.02)
    parser.add_argument("--roof-shape-weight", type=float, default=0.0)
    parser.add_argument("--roof-shape-warmup-epochs", type=int, default=15)
    parser.set_defaults(sar_dropout_probability=0.0)
    parser.set_defaults(sar_hold_epochs=0)
    parser.set_defaults(sar_ramp_epochs=0)
    parser.add_argument("--tail-low-threshold", type=float, default=3.0)
    parser.add_argument("--tail-high-threshold", type=float, default=20.0)
    parser.add_argument("--min-height", type=float, default=0.1)
    parser.add_argument("--sar-clip", type=float, default=60.0)
    parser.add_argument(
        "--sar-normalization",
        choices=["linear", "log"],
        default="linear",
        help="SpaceNet6 SAR scaling. Linear is appropriate for the provided dB rasters; log is legacy.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="HDBF-Net")
    parser.add_argument(
        "--checkpoint-mode",
        choices=["full", "inference"],
        default="full",
        help="Save resumable training checkpoints or inference-only weight packages. Inference packages contain no epoch, metrics, optimizer, scheduler, scaler, or training arguments.",
    )
    args = parser.parse_args()
    if args.geometry_height_fusion and (not args.local_structure):
        parser.error("Geometry-height fusion requires --local-structure")
    if args.geometry_height_mode != "joint-global" and (
        not args.geometry_height_fusion
    ):
        parser.error("Geometry-height mode requires --geometry-height-fusion")
    if args.local_structure or args.unirep_pretrained:
        if (
            not args.local_structure
            or not args.unirep_pretrained
            or args.encoder != "dinov2"
            or (args.fusion_type != "task-aware")
            or (not args.task_split_decoder)
            or args.hrbr
            or args.structure_isolation
            or args.directional_ttci
            or args.height_guidance_stop_gradient
            or (not args.context_detail_height)
            or (args.height_head_weight <= 0)
            or (not args.normalize_rgb)
            or (args.contour_objective != "bce-dice")
            or args.contour_structure_gate
            or (not all((args.ttci, args.hcr, args.slmhc)))
        ):
            parser.error(
                "Local structure requires UniRep, DINO TaskFusion/TaskSplit, ContextDetail, TTCI/HCR/SLMHC, ImageNet RGB, BCE+Dice and --no-hrbr, without gradient isolation or contour gate"
            )
        if (
            not 0 < args.unirep_lr_scale <= 1
            or not math.isfinite(args.unirep_lr_scale)
            or (not math.isfinite(args.structure_branch_weight))
            or (args.structure_branch_weight <= 0)
            or (not math.isfinite(args.local_structure_weight))
            or (args.local_structure_weight <= 0)
        ):
            parser.error("Invalid UniRep LR or auxiliary loss weights")
    return args


def float_tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple((float(item) for item in value))
    try:
        parsed = tuple((float(item.strip()) for item in str(value).split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return parsed


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset_metadata(base_dir):
    summary_path = os.path.join(base_dir, "summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"SpaceNet6 dataset summary not found: {summary_path}. Use the processed_height_dense directory produced by the dense-height preprocessing script."
        )
    with open(summary_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    height_definition = str(metadata.get("height_definition", ""))
    if "pixel-level" not in height_definition.lower():
        raise ValueError(
            f"The selected SpaceNet6 labels are not the dense pixel-level height target: {height_definition or 'height definition missing'}"
        )
    return metadata


def dice_loss_with_logits(logits, target, eps=1e-06):
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    union = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()


def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return 0.5 * bce + 0.5 * dice_loss_with_logits(logits, target)


def tversky_loss_with_logits(logits, target, fp_weight=0.7, fn_weight=0.3, eps=1.0):
    probability = torch.sigmoid(logits.float())
    target = target.float()
    dims = (1, 2, 3)
    true_positive = (probability * target).sum(dim=dims)
    false_positive = (probability * (1.0 - target)).sum(dim=dims)
    false_negative = ((1.0 - probability) * target).sum(dim=dims)
    score = (true_positive + eps) / (
        true_positive + fp_weight * false_positive + fn_weight * false_negative + eps
    )
    return (1.0 - score).mean()


def balanced_contour_loss(logits, target):
    """Equal positive/negative class means per image, plus symmetric Dice."""
    (logits, target) = (logits.float(), target.float())
    positive = target > 0.5
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    losses = []
    for values, selected in zip(bce, positive):
        terms = []
        if selected.any():
            terms.append(values[selected].mean())
        if (~selected).any():
            terms.append(values[~selected].mean())
        losses.append(torch.stack(terms).mean())
    return 0.5 * torch.stack(losses).mean() + 0.5 * dice_loss_with_logits(
        logits, target
    )


def contour_far_negative_loss(logits, target, band_width=3, fraction=0.1):
    support = dilate_binary(target > 0.5, band_width)
    negative_bce = F.softplus(logits.float())
    image_losses = []
    for batch_index in range(logits.shape[0]):
        candidates = negative_bce[batch_index][~support[batch_index]]
        if candidates.numel() == 0:
            continue
        count = max(1, int(math.ceil(candidates.numel() * fraction)))
        image_losses.append(torch.topk(candidates, count, sorted=False).values.mean())
    if not image_losses:
        return logits.sum() * 0.0
    return torch.stack(image_losses).mean()


def contour_mask_consistency_loss(contour_logits, mask_logits, kernel_size=5):
    mask_probability = torch.sigmoid(mask_logits.detach().float())
    padding = kernel_size // 2
    dilated = F.max_pool2d(
        mask_probability, kernel_size=kernel_size, stride=1, padding=padding
    )
    eroded = -F.max_pool2d(
        -mask_probability, kernel_size=kernel_size, stride=1, padding=padding
    )
    soft_boundary = (dilated - eroded).clamp_(0.0, 1.0)
    contour_probability = torch.sigmoid(contour_logits.float())
    intersection = (contour_probability * soft_boundary).sum(dim=(1, 2, 3))
    union = contour_probability.sum(dim=(1, 2, 3)) + soft_boundary.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (union + 1.0)).mean()


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


def height_range_loss(gate_logits, target, valid, boundaries):
    if valid.dim() == 4 and valid.size(1) == 1:
        valid = valid.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    if not valid.any():
        return gate_logits.sum() * 0.0
    labels = torch.zeros_like(target, dtype=torch.long)
    for class_index, boundary in enumerate(boundaries, start=1):
        labels[target >= float(boundary)] = class_index
    valid_labels = labels[valid]
    class_count = gate_logits.shape[1]
    counts = torch.bincount(valid_labels, minlength=class_count).float()
    present = counts > 0
    class_weight = torch.zeros_like(counts)
    if present.any():
        total = counts[present].sum()
        class_weight[present] = torch.sqrt(
            total / (present.sum().to(counts.dtype) * counts[present])
        ).clamp_(0.5, 3.0)
        class_weight[present] /= class_weight[present].mean()
    return F.cross_entropy(
        gate_logits.float().permute(0, 2, 3, 1)[valid],
        valid_labels,
        weight=class_weight,
    )


def directional_tail_loss(
    prediction, target, valid, low_threshold=3.0, high_threshold=10.0
):
    low = valid & (target < low_threshold)
    high = valid & (target >= high_threshold)
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
    prediction, target, instance_map, valid, min_pixels=16, min_valid_ratio=0.5
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
        instance_losses.append((absolute + 0.25 * relative).mean())
    if not instance_losses:
        return prediction.sum() * 0.0
    return torch.stack(instance_losses).mean()


def instance_roof_shape_loss(
    prediction, target, instance_map, valid, min_pixels=16, min_valid_ratio=0.5
):
    """Match within-building roof relief after removing each building mean."""
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
    image_losses = []
    for batch_index in range(prediction.size(0)):
        ids = instance_map[batch_index].long()
        positive = ids > 0
        reliable = positive & valid[batch_index]
        if not reliable.any():
            continue
        max_id = int(ids.max().item())
        all_counts = torch.bincount(ids[positive], minlength=max_id + 1).to(
            prediction.dtype
        )
        reliable_ids = ids[reliable]
        counts = torch.bincount(reliable_ids, minlength=max_id + 1).to(prediction.dtype)
        keep = counts >= float(min_pixels)
        keep &= counts / all_counts.clamp_min(1.0) >= min_valid_ratio
        keep[0] = False
        if not keep.any():
            continue
        pred_sums = prediction.new_zeros(max_id + 1)
        target_sums = prediction.new_zeros(max_id + 1)
        pred_sums.scatter_add_(0, reliable_ids, prediction[batch_index][reliable])
        target_sums.scatter_add_(0, reliable_ids, target[batch_index][reliable])
        pred_means = pred_sums / counts.clamp_min(1.0)
        target_means = target_sums / counts.clamp_min(1.0)
        kept_pixels = reliable & keep[ids]
        kept_ids = ids[kept_pixels]
        centered_error = (
            prediction[batch_index][kept_pixels]
            - pred_means[kept_ids]
            - (target[batch_index][kept_pixels] - target_means[kept_ids])
        )
        pixel_loss = F.smooth_l1_loss(
            centered_error, torch.zeros_like(centered_error), beta=0.5, reduction="none"
        )
        loss_sums = prediction.new_zeros(max_id + 1)
        loss_sums.scatter_add_(0, kept_ids, pixel_loss)
        image_losses.append((loss_sums[keep] / counts[keep]).mean())
    if not image_losses:
        return prediction.sum() * 0.0
    return torch.stack(image_losses).mean()


def make_poly_scheduler(
    optimizer, steps_per_epoch, epochs, warmup_epochs, power, min_lr_ratio
):
    total_steps = max(steps_per_epoch * epochs, 1)
    warmup_steps = max(steps_per_epoch * warmup_epochs, 0)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return 0.1 + 0.9 * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        decay = max(1.0 - min(max(progress, 0.0), 1.0), 0.0) ** power
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay

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
            if self.bn_policy == "copy" and key in self.bn_buffers:
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
def evaluate(
    model,
    loader,
    device,
    min_height,
    boundary_band_width,
    amp_enabled,
    height_stages=False,
    val_inference="whole",
    val_window=768,
    val_stride=384,
):
    model.eval()
    tp = fp = fn = 0.0
    contour_tp = contour_fp = contour_fn = 0.0
    boundary_tp = boundary_fp = boundary_fn = 0.0
    absolute_error = squared_error = pixels = 0.0
    stage_errors = {
        key: 0.0 for key in ("height_head", "height_coarse", "height_after_hcr")
    }
    for batch in tqdm(
        loader, desc="val", ascii=True, leave=False, disable=not sys.stderr.isatty()
    ):
        rgb = batch["rgb"].to(device, non_blocking=True)
        sar = batch["sar"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        contour = batch["contour"].to(device, non_blocking=True)
        height = batch["height"].to(device, non_blocking=True)
        if val_inference == "sliding":
            from hdbf.spacenet6_validation_windows import predict_validation_windows

            output = predict_validation_windows(
                model, rgb, sar, val_window, val_stride, amp_enabled, height_stages
            )
            pred_mask = output["mask"] > 0.5
            pred_contour = output["contour"] > 0.5
        else:
            with autocast(enabled=amp_enabled):
                output = (
                    model(rgb, sar, return_intermediates=True)
                    if height_stages
                    else model(rgb, sar)
                )
            pred_mask = torch.sigmoid(output["mask"]) > 0.5
            pred_contour = torch.sigmoid(output["contour"]) > 0.5
        gt_mask = mask > 0.5
        gt_contour = contour > 0.5
        label_valid = batch["valid_height"].to(device, non_blocking=True) > 0.5
        valid_height = gt_mask & label_valid & (height > min_height)
        if valid_height.any():
            error = (
                output["height"][valid_height].float() - height[valid_height].float()
            )
            absolute_error += error.abs().sum().item()
            squared_error += error.square().sum().item()
            pixels += valid_height.sum().item()
            if height_stages:
                for key in stage_errors:
                    stage_errors[key] += (
                        (
                            output[key][valid_height].float()
                            - height[valid_height].float()
                        )
                        .abs()
                        .sum()
                        .item()
                    )
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
    recall = tp / max(tp + fn, 1.0)
    contour_precision = contour_tp / max(contour_tp + contour_fp, 1.0)
    contour_recall = contour_tp / max(contour_tp + contour_fn, 1.0)
    boundary_precision = boundary_tp / max(boundary_tp + boundary_fp, 1.0)
    boundary_recall = boundary_tp / max(boundary_tp + boundary_fn, 1.0)
    return {
        **(
            {
                f"{key}_mae": value / max(pixels, 1.0)
                for (key, value) in stage_errors.items()
            }
            if height_stages
            else {}
        ),
        "mae": absolute_error / max(pixels, 1.0),
        "rmse": math.sqrt(squared_error / max(pixels, 1.0)),
        "f1": 2 * precision * recall / max(precision + recall, 1e-08),
        "precision": precision,
        "recall": recall,
        "contour_f1": 2
        * contour_precision
        * contour_recall
        / max(contour_precision + contour_recall, 1e-08),
        "contour_precision": contour_precision,
        "contour_recall": contour_recall,
        "boundary_f1": 2
        * boundary_precision
        * boundary_recall
        / max(boundary_precision + boundary_recall, 1e-08),
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
    }


def save_checkpoint(
    path,
    model,
    ema,
    optimizer,
    scheduler,
    scaler,
    epoch,
    metrics,
    args,
    training_state=None,
):
    if args.checkpoint_mode == "inference":
        torch.save(
            {
                "format": "hdbfnet_spacenet6_inference_v1",
                "selected_epoch": epoch,
                "contour_objective": args.contour_objective,
                "validation_metrics": {
                    key: value
                    for (key, value) in metrics.items()
                    if key in ("mae", "rmse", "f1", "contour_f1")
                },
                "model_name": args.model_name,
                "model_config": {
                    "local_structure": args.local_structure,
                    "geometry_height_fusion": getattr(
                        args, "geometry_height_fusion", False
                    ),
                    "geometry_height_mode": getattr(
                        args, "geometry_height_mode", "joint-global"
                    ),
                    "unirep_structure": bool(args.unirep_pretrained),
                    "unirep_provenance": getattr(args, "unirep_provenance", None),
                    "architecture": getattr(model, "architecture", None),
                    "encoder": "dinov2",
                    "fusion_type": args.fusion_type,
                    "task_split_decoder": args.task_split_decoder,
                    "structure_isolation": False,
                    "building_affinity": args.building_affinity,
                    "sar_channels": 4,
                    "context_detail_height": getattr(
                        args, "context_detail_height", False
                    ),
                    "use_multiscale_height_head": False,
                    "height_head_kind": args.height_head_kind,
                    "use_height_multiscale_adapter": False,
                    "variant": args.variant,
                    "decoder_dim": args.decoder_dim,
                    "use_hrbr": bool(args.hrbr),
                    "use_hcr": bool(args.hcr),
                    "use_slmhc": bool(args.slmhc),
                    "use_bihm": bool(args.bihm),
                    "use_ttci": bool(args.ttci),
                    "ttci_bypass": bool(False),
                    "task_channels": args.task_channels,
                    "rgb_anchored_fusion": bool(False),
                    "auxiliary_initial_strength": args.auxiliary_initial_strength,
                    "rgb_anchored_hrbr": bool(False),
                    "hcr_residual_scales": tuple(args.hcr_residual_scales),
                    "slmhc_knots": tuple(args.slmhc_knots),
                    "structure_only_sar": bool(False),
                    "structure_fusion_initial_strength": args.structure_fusion_initial_strength,
                    "directional_ttci": bool(False),
                    "use_roof_context_refiner": bool(False),
                    "height_guidance_stop_gradient": bool(
                        args.height_guidance_stop_gradient
                    ),
                    "use_rgb_height_detail_refiner": bool(False),
                    "rgb_height_detail_channels": args.rgb_height_detail_channels,
                    "rgb_height_detail_initial_strength": args.rgb_height_detail_initial_strength,
                    "rgb_height_detail_max_residual": args.rgb_height_detail_max_residual,
                    "use_contour_structure_gate": bool(args.contour_structure_gate),
                    "contour_gate_kernel": args.contour_gate_kernel,
                    "contour_gate_floor": args.contour_gate_floor,
                    "contour_gate_initial_strength": args.contour_gate_initial_strength,
                },
                "data_config": {
                    "splits": getattr(args, "split_metadata", None),
                    "min_height": args.min_height,
                    "val_inference": args.val_inference,
                    "val_window": args.val_window,
                    "val_stride": args.val_stride,
                    "ema_bn_policy": args.ema_bn_policy,
                    "normalize_rgb": args.normalize_rgb,
                    "sar_clip": args.sar_clip,
                    "sar_normalization": args.sar_normalization,
                    "height_definition": args.height_definition,
                },
                "model_state_dict": ema.model_state_dict(),
                "evaluation_weights": "EMA" if args.ema_decay > 0 else "raw",
            },
            path,
        )
        return
    torch.save(
        {
            "epoch": epoch,
            "model_config": {
                "local_structure": args.local_structure,
                "geometry_height_fusion": getattr(
                    args, "geometry_height_fusion", False
                ),
                "geometry_height_mode": getattr(
                    args, "geometry_height_mode", "joint-global"
                ),
                "unirep_structure": bool(args.unirep_pretrained),
                "unirep_provenance": getattr(args, "unirep_provenance", None),
                "architecture": getattr(model, "architecture", None),
                "encoder": "dinov2",
                "fusion_type": args.fusion_type,
                "task_split_decoder": args.task_split_decoder,
                "structure_isolation": False,
                "building_affinity": args.building_affinity,
                "sar_channels": 4,
            },
            "training_state": training_state,
            "data_config": {
                "splits": getattr(args, "split_metadata", None),
                "min_height": args.min_height,
                "val_inference": args.val_inference,
                "val_window": args.val_window,
                "val_stride": args.val_stride,
                "ema_bn_policy": args.ema_bn_policy,
                "normalize_rgb": args.normalize_rgb,
                "sar_clip": args.sar_clip,
                "sar_normalization": args.sar_normalization,
                "height_definition": args.height_definition,
            },
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
        },
        path,
    )


HEIGHT_RESCUE_PREFIXES = (
    "height_head.",
    "height_conditional_residual.",
    "height_calibrator.",
    "task_interaction.height_update.",
    "task_interaction.height_correction.",
    "roof_context_refiner.",
)


def main():
    args = parse_args()
    resume_package = None
    (explicit_seed, explicit_data_seed) = (args.seed, args.data_seed)
    if resume_package is not None:
        print("Restoring complete training state:", None)
    args.context_detail_height = getattr(args, "context_detail_height", False)
    args.geometry_height_fusion = getattr(args, "geometry_height_fusion", False)
    args.geometry_height_mode = getattr(args, "geometry_height_mode", "joint-global")
    args.save_best_mae_only = getattr(args, "save_best_mae_only", False)
    args.split_dir = getattr(args, "split_dir", None)
    if (
        args.variant != "B4"
        or not args.normalize_rgb
        or (not args.pretrained)
        or (not args.sar_pretrained)
    ):
        raise ValueError(
            "DINOv2 requires B4 SAR, ImageNet RGB and both pretrained files"
        )
    if args.fusion_type == "task-aware" and (not args.task_split_decoder):
        raise ValueError("Task-aware fusion requires --task-split-decoder")
    if args.building_affinity and (
        not args.context_detail_height or args.height_head_weight <= 0
    ):
        raise ValueError(
            "Building affinity requires ContextDetail and positive head supervision"
        )
    if not math.isfinite(args.affinity_loss_weight) or args.affinity_loss_weight < 0:
        raise ValueError("Affinity loss weight must be finite and nonnegative")
    if args.context_detail_height:
        if "hdbf" != "hdbf" or args.height_head_weight <= 0 or False or False:
            raise ValueError(
                "ContextDetail requires experiment=hdbf, positive height-head-weight and no alternate head"
            )
    if args.save_best_mae_only and (None or args.stop_after_epoch is not None):
        raise ValueError("Best-only runs cannot resume or stop for later continuation")
    args.seed = (
        explicit_seed
        if explicit_seed is not None
        else args.seed if args.seed is not None else 42
    )
    args.data_seed = (
        explicit_data_seed
        if explicit_data_seed is not None
        else (
            args.data_seed
            if args.data_seed is not None
            and (
                getattr(args, "ttci_bypass", False)
                or "hdbf" == "hdbf-multiscale-height"
            )
            else args.seed
        )
    )
    if args.stop_after_epoch is not None:
        raise ValueError(
            "stop-after-epoch requires the joint height trial and must be within its full budget"
        )
    if (
        resume_package is not None
        and (args.stop_after_epoch or args.epochs) <= resume_package["epoch"]
    ):
        raise ValueError("Requested end epoch must be later than the saved epoch")
    if not all((0 <= seed < 2**32 for seed in (args.seed, args.data_seed))):
        raise ValueError("Seeds must be in [0, 2**32)")
    if args.val_inference == "sliding":
        if (
            args.val_window <= 0
            or args.val_window % 32
            or (not 0 < args.val_stride <= args.val_window)
        ):
            raise ValueError(
                "Sliding validation requires window divisible by 32 and 0 < stride <= window"
            )
    print(
        f"Experiment: {'hdbf'}; validation: {args.val_inference}; window={args.val_window}; stride={args.val_stride}; threshold > 0.5; select checkpoint by validation MAE"
    )
    if len(args.hcr_residual_scales) != len(args.height_range_boundaries) + 1:
        raise ValueError(
            "--hcr-residual-scales must have exactly one more value than --height-range-boundaries"
        )
    if any(
        (
            right <= left
            for (left, right) in zip(
                args.height_range_boundaries, args.height_range_boundaries[1:]
            )
        )
    ):
        raise ValueError("--height-range-boundaries must be strictly increasing")
    if len(args.slmhc_knots) < 2 or any(
        (right <= left for (left, right) in zip(args.slmhc_knots, args.slmhc_knots[1:]))
    ):
        raise ValueError("--slmhc-knots must contain increasing values")
    if not 0.0 <= 0.0 < 1.0:
        raise ValueError("--sar-dropout-probability must be in [0, 1)")
    if not 0.0 <= args.checkpoint_min_mask_f1 <= 1.0:
        raise ValueError("--checkpoint-min-mask-f1 must be in [0, 1]")
    if not 0.0 <= args.checkpoint_min_contour_f1 <= 1.0:
        raise ValueError("--checkpoint-min-contour-f1 must be in [0, 1]")
    if args.contour_gate_kernel < 1 or args.contour_gate_kernel % 2 == 0:
        raise ValueError("--contour-gate-kernel must be a positive odd integer")
    if not 0.0 < args.contour_gate_floor <= 1.0:
        raise ValueError("--contour-gate-floor must be in (0, 1]")
    if not 0.0 < args.contour_gate_initial_strength < 1.0:
        raise ValueError("--contour-gate-initial-strength must be in (0, 1)")
    contour_loss_weights = (
        args.contour_bce_weight,
        args.contour_tversky_weight,
        args.contour_far_weight,
        args.contour_consistency_weight,
    )
    if any((weight < 0.0 for weight in contour_loss_weights)):
        raise ValueError("contour loss weights must be non-negative")
    if sum(contour_loss_weights[:3]) <= 0.0:
        raise ValueError("at least one primary contour loss weight must be positive")
    if args.contour_tversky_fp_weight < 0.0 or args.contour_tversky_fn_weight < 0.0:
        raise ValueError("Tversky FP/FN weights must be non-negative")
    if args.contour_tversky_fp_weight + args.contour_tversky_fn_weight <= 0.0:
        raise ValueError("Tversky FP/FN weights cannot both be zero")
    if args.contour_far_band_width < 0:
        raise ValueError("--contour-far-band-width must be non-negative")
    if not 0.0 < args.contour_hard_negative_fraction <= 1.0:
        raise ValueError("--contour-hard-negative-fraction must be in (0, 1]")
    if args.contour_consistency_kernel < 1 or args.contour_consistency_kernel % 2 == 0:
        raise ValueError("--contour-consistency-kernel must be a positive odd integer")
    if args.rgb_height_detail_channels < 8:
        raise ValueError("--rgb-height-detail-channels must be at least 8")
    if not 0.0 < args.rgb_height_detail_initial_strength < 1.0:
        raise ValueError("--rgb-height-detail-initial-strength must be in (0, 1)")
    if args.rgb_height_detail_max_residual <= 0.0:
        raise ValueError("--rgb-height-detail-max-residual must be positive")
    if 0 < 0 or 0 < 0:
        raise ValueError("SAR hold/ramp epochs must be non-negative")
    if not 0.0 < args.rgb_backbone_lr_scale <= 1.0:
        raise ValueError("--rgb-backbone-lr-scale must be in (0, 1]")
    seed_everything(args.seed)
    base_dir = args.base_dir or os.environ.get(
        "SN6_HEIGHT_BASE", "D:\\znx\\data\\spacenet6\\processed_height_dense"
    )
    base_dir = os.path.abspath(base_dir)
    source_dir = os.path.abspath(args.source_dir or os.path.dirname(base_dir))
    dataset_metadata = load_dataset_metadata(base_dir)
    args.base_dir = base_dir
    args.source_dir = source_dir
    args.height_definition = dataset_metadata["height_definition"]
    args.split_dir = os.path.abspath(args.split_dir or os.path.join(base_dir, "splits"))
    args.split_metadata = split_metadata(args.split_dir)
    print("Split manifests:", json.dumps(args.split_metadata))
    run_dir = os.path.join(
        args.checkpoint_dir, datetime.now().strftime("%y%m%d_%H%M%S")
    )
    if resume_package is not None:
        run_dir = os.path.dirname(None)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and device.type == "cuda"
    model_class = CMNeXtMTL
    model_options = {"building_affinity": args.building_affinity}
    from semseg.models.hdbf_net import DinoHDBF

    model_class = DinoHDBF
    model_options.update(
        gradient_checkpointing=args.gradient_checkpointing,
        structure_isolation=False,
        fusion_type=args.fusion_type,
        task_split_decoder=args.task_split_decoder,
    )
    model = model_class(
        **model_options,
        context_detail_height=args.context_detail_height,
        use_multiscale_height_head=False,
        multiscale_height_head_kind=args.height_head_kind,
        variant=args.variant,
        decoder_dim=args.decoder_dim,
        use_hrbr=args.hrbr,
        use_hcr=args.hcr,
        use_slmhc=args.slmhc,
        use_ttci=args.ttci,
        task_channels=args.task_channels,
        sar_channels=4,
        rgb_anchored_fusion=False,
        auxiliary_initial_strength=args.auxiliary_initial_strength,
        rgb_anchored_hrbr=False,
        hcr_residual_scales=args.hcr_residual_scales,
        slmhc_knots=args.slmhc_knots,
        structure_only_sar=False,
        structure_fusion_initial_strength=args.structure_fusion_initial_strength,
        directional_ttci=False,
        use_roof_context_refiner=False,
        height_guidance_stop_gradient=args.height_guidance_stop_gradient,
        use_rgb_height_detail_refiner=False,
        rgb_height_detail_channels=args.rgb_height_detail_channels,
        rgb_height_detail_initial_strength=args.rgb_height_detail_initial_strength,
        rgb_height_detail_max_residual=args.rgb_height_detail_max_residual,
        use_contour_structure_gate=args.contour_structure_gate,
        contour_gate_kernel=args.contour_gate_kernel,
        contour_gate_floor=args.contour_gate_floor,
        contour_gate_initial_strength=args.contour_gate_initial_strength,
    )
    if args.pretrained and (not None) and (resume_package is None):
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(
                f"Encoder pretrained checkpoint not found: {args.pretrained}"
            )
        model.init_pretrained(args.pretrained)
        model.init_sar_pretrained(args.sar_pretrained)
    if args.local_structure:
        from hdbf.unirep_structure import UniRepStructure

        with torch.random.fork_rng(devices=[]):
            model.unirep_structure = UniRepStructure(
                model.backbone.channels, args.decoder_dim, local_structure=True
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
    from hdbf.spacenet6_reproducibility import seed_worker, record_initial_state

    if resume_package is None:
        record_initial_state(args, model, run_dir)
    model.to(device)
    mix_probability = getattr(args, "building_mix_probability", 0.0)
    if mix_probability and (
        "hdbf" != "hdbf-multiscale-height"
        or args.height_head_kind != "standard"
        or mix_probability != 0.25
    ):
        raise ValueError(
            "Building-mix trial requires joint standard head and probability 0.25"
        )
    dataset_class = SpaceNet6HeightDataset
    dataset_options = {}
    if getattr(args, "standard_scale_augmentation", False) and (
        not getattr(args, "standard_tune100", False)
    ):
        raise ValueError("Scale augmentation requires standard-tune100")
    if getattr(args, "standard_tune100", False):
        raise ValueError("This historical experiment is not part of the final release")
    if mix_probability:
        raise ValueError("This historical experiment is not part of the final release")
    train_set = dataset_class(
        base_dir,
        split="train",
        train=True,
        crop_size=args.crop_size,
        crops_per_image=args.crops_per_image,
        sar_clip=args.sar_clip,
        sar_normalization=args.sar_normalization,
        min_building_pixels=args.min_building_pixels,
        normalize_rgb=args.normalize_rgb,
        positive_crop_probability=args.positive_crop_probability,
        targeted_crop_probability=0.0,
        targeted_height_min=args.targeted_height_min,
        targeted_height_max=args.targeted_height_max,
        targeted_area_min_m2=args.targeted_area_min_m2,
        source_dir=source_dir,
        split_dir=args.split_dir,
        **dataset_options,
    )
    if mix_probability:
        train_set.write_summary(os.path.join(run_dir, "building_sampler_index.json"))
    val_set = SpaceNet6HeightDataset(
        base_dir,
        split="val",
        train=False,
        normalize_rgb=args.normalize_rgb,
        sar_clip=args.sar_clip,
        sar_normalization=args.sar_normalization,
        source_dir=source_dir,
        split_dir=args.split_dir,
    )
    train_generator = torch.Generator().manual_seed(args.data_seed)
    val_generator = torch.Generator().manual_seed(args.data_seed + 1)
    random.seed(args.data_seed)
    np.random.seed(args.data_seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        worker_init_fn=seed_worker,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        persistent_workers=args.num_workers > 0 and "hdbf" != "hdbf-multiscale-height",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        generator=val_generator,
        worker_init_fn=seed_worker,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0 and "hdbf" != "hdbf-multiscale-height",
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    rgb_backbone_parameters = []
    local_encoder_parameters = []
    unirep_parameters = []
    task_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_rgb_encoder = name.startswith("backbone.") and (
            not any(
                (
                    token in name
                    for token in ("extra_", "FRMs", "FFMs", "rgb_anchored_fusions")
                )
            )
        )
        is_rgb_encoder = name.startswith("backbone.dino.")
        if name.startswith(("unirep_structure.cnn.", "unirep_structure.ade_decoder.")):
            unirep_parameters.append(parameter)
        elif name.startswith("height_head.local_encoder."):
            local_encoder_parameters.append(parameter)
        elif is_rgb_encoder:
            rgb_backbone_parameters.append(parameter)
        else:
            task_parameters.append(parameter)
    optimizer_groups = [
        {
            "params": rgb_backbone_parameters,
            "lr": args.lr * args.rgb_backbone_lr_scale,
            "name": "rgb_backbone",
        },
        {"params": task_parameters, "lr": args.lr, "name": "task"},
    ]
    if local_encoder_parameters:
        optimizer_groups.append(
            {
                "params": local_encoder_parameters,
                "lr": args.lr * 0.1,
                "name": "local_rgb_encoder",
            }
        )
    if unirep_parameters:
        optimizer_groups.append(
            {
                "params": unirep_parameters,
                "lr": args.lr * args.unirep_lr_scale,
                "name": "unirep_pretrained",
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups, lr=args.lr, weight_decay=args.weight_decay
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
    ema = ModelEMA(model, args.ema_decay, args.ema_bn_policy)
    print(f"Run dir: {run_dir}")
    if args.local_structure:
        print(
            "Structure: UniRep 192-channel dense local decoder; native CNN bypass, two local affinity steps; four-band SAR; original height head/corrections; no query decoder or task isolation"
        )
        if args.geometry_height_fusion:
            print(
                "Geometry -> height: four-scale selection + ADE context + predicted geometry priors before ContextDetail/TTCI/HCR/SLMHC; zero-output initialization, fully trainable; CNN encoded once"
            )
            print(
                f"Geometry-height mode: {args.geometry_height_mode}; new-route CNN gradients={('stopped' if args.geometry_height_mode == 'asymmetric-local' else 'joint')}; geometry losses and original task interactions remain trainable"
            )
        print(
            f"Structure auxiliary={args.structure_branch_weight}; local auxiliary={args.local_structure_weight}; height guidance ramps epochs20..40; BN copied during training, no post-training calibration"
        )
    print(f"Processed labels: {base_dir}")
    print(f"SpaceNet6 source: {source_dir}")
    print(f"Height target: {dataset_metadata['height_definition']}")
    model_suffix = "-HRBR" if args.hrbr else ""
    if args.context_detail_height:
        model_suffix += "-ContextDetail"
    if args.hcr:
        model_suffix += "-HCR"
    if args.slmhc:
        model_suffix += "-SLMHC"
    if args.bihm:
        model_suffix += "-BIHM"
    if args.ttci:
        model_suffix += "-TTCI"
    if args.contour_structure_gate:
        model_suffix += "-SCContour"
    print(
        f"Model: {args.model_name} | Encoder: {'dinov2'} | Input: RGB 3ch + SpaceNet6 SAR 4ch"
    )
    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")
    print(
        f"Training crop: {args.crop_size}x{args.crop_size} | Crops per image: {args.crops_per_image} | Positive crop probability: {args.positive_crop_probability:.2f} | Validation: {args.val_inference} on native tiles"
        + (
            f" (window={args.val_window}, stride={args.val_stride})"
            if args.val_inference == "sliding"
            else ""
        )
    )
    print(
        f"Height-area targeted crops: probability={0.0:.2f} | height=[{args.targeted_height_min:g}, {args.targeted_height_max:g}) m | minimum area={args.targeted_area_min_m2:g} m2"
    )
    print(
        f"SpaceNet6 SAR normalization: {args.sar_normalization} | clip={args.sar_clip:g} dB"
    )
    print(
        f"SpaceNet6 adaptation: RGB-anchored fusion={False} | RGB-anchored HRBR={False} | structure-only SAR={False} | directional TTCI={False} | SAR dropout={0.0:g}"
    )
    print(
        f"SAR curriculum: hold={0} epochs | ramp={0} epochs | roof-shape weight={args.roof_shape_weight:g}"
    )
    print(f"Height guidance: stop_gradient={args.height_guidance_stop_gradient}")
    print(
        f"Height ranges: {args.height_range_boundaries} | HCR scales: {args.hcr_residual_scales} | SLMHC knots: {args.slmhc_knots}"
    )
    print(
        f"Batch size: {args.batch_size} | Val batch size: {args.val_batch_size} | LR: {args.lr} | AMP: {amp_enabled} | Seed: {args.seed}"
    )
    print(f"Pretrained encoder ({'dinov2'}): {args.pretrained}")
    print(
        f"Fusion: {args.fusion_type}; TaskSplit: {args.task_split_decoder}; affinity: {args.building_affinity}"
    )
    print(
        f"Contour objective: {args.contour_objective}; non-legacy modes disable far-negative and mask-consistency penalties"
    )
    print(f"Structure gradient isolation: {False}")
    print(
        f"RGB normalization: {('ImageNet' if args.normalize_rgb else '0-1 (legacy)')}"
    )
    print(f"Boundary validation band width: {args.boundary_band_width}px")
    print(
        f"Contour structure gate: enabled={args.contour_structure_gate} | kernel={args.contour_gate_kernel} | floor={args.contour_gate_floor:g} | initial_strength={args.contour_gate_initial_strength:g}"
    )
    print(
        (
            "Legacy contour settings (inactive): "
            if args.contour_objective != "legacy"
            else "Contour supervision: "
        )
        + f"precision-oriented BCE+Tversky+far-negative | task_weight={args.contour_weight:g} | component_weights=({args.contour_bce_weight:g}, {args.contour_tversky_weight:g}, {args.contour_far_weight:g}) | Tversky(FP={args.contour_tversky_fp_weight:g}, FN={args.contour_tversky_fn_weight:g}) | consistency_weight={args.contour_consistency_weight:g}"
    )
    print(
        f"Balanced-MAE checkpoint floors: mask_f1>={args.checkpoint_min_mask_f1} | contour_f1>={args.checkpoint_min_contour_f1}"
    )
    if args.lr_schedule == "cosine":
        print(
            f"LR schedule: cosine | min_ratio={args.min_lr_ratio} | warmup={args.warmup_epochs} epochs"
        )
    else:
        print(
            f"LR schedule: floor polynomial | power={args.poly_power} | min_ratio={args.min_lr_ratio} | warmup={args.warmup_epochs} epochs"
        )
    print(
        f"Evaluation weights: {('EMA' if args.ema_decay > 0 else 'raw (EMA decay=0)')} | decay={args.ema_decay}"
    )
    print(
        f"EMA BN policy: {args.ema_bn_policy}; validation segmentation threshold remains 0.5"
    )
    if not math.isfinite(args.height_head_weight) or args.height_head_weight < 0:
        raise ValueError("--height-head-weight must be finite and nonnegative")
    if (
        not math.isfinite(args.height_correction_guard_weight)
        or args.height_correction_guard_weight < 0
    ):
        raise ValueError(
            "--height-correction-guard-weight must be finite and nonnegative"
        )
    if args.height_correction_guard_warmup_epochs < 0:
        raise ValueError("--height-correction-guard-warmup-epochs must be nonnegative")
    use_height_supervision = (
        args.height_head_weight > 0 or args.height_correction_guard_weight > 0
    )
    if use_height_supervision:
        from hdbf.spacenet6_height_supervision import (
            head_supervision,
            correction_safeguard,
        )

        print(
            f"Direct head weight={args.height_head_weight}; correction guard weight={args.height_correction_guard_weight}; guard warmup={args.height_correction_guard_warmup_epochs} epochs"
        )
    print(
        f"Model params: {sum((p.numel() for p in model.parameters())) / 1000000.0:.1f}M"
    )
    print(
        f"Trainable params: {sum((p.numel() for p in trainable_parameters)) / 1000000.0:.3f}M"
    )
    best_mae = float("inf")
    best_f1 = 0.0
    best_contour_f1 = 0.0
    best_boundary_f1 = 0.0
    best_balanced_mae = float("inf")
    start_epoch = 1
    if resume_package is not None:
        raise ValueError("This historical experiment is not part of the final release")
    end_epoch = args.stop_after_epoch or args.epochs
    print(
        f"Training epochs {start_epoch}..{end_epoch}; LR schedule horizon remains {args.epochs} epochs"
    )
    for epoch in range(start_epoch, end_epoch + 1):
        if args.local_structure:
            model.unirep_structure.local_decoder.set_epoch(epoch)
        model.train()
        sums = {
            "local_structure": 0.0,
            "structure_branches": 0.0,
            "affinity": 0.0,
            "affinity_pos_pairs": 0.0,
            "affinity_neg_pairs": 0.0,
            "height_head_aux": 0.0,
            "ttci_guard": 0.0,
            "hcr_guard": 0.0,
            "loss": 0.0,
            "mask": 0.0,
            "contour": 0.0,
            "contour_bce": 0.0,
            "contour_tversky": 0.0,
            "contour_far": 0.0,
            "contour_consistency": 0.0,
            "height": 0.0,
            "bg": 0.0,
            "hcr_range": 0.0,
            "tail": 0.0,
            "bihm": 0.0,
            "roof_shape": 0.0,
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
        roof_shape_weight = args.roof_shape_weight * min(
            max((epoch - 1) / max(args.roof_shape_warmup_epochs, 1), 0.0), 1.0
        )
        if epoch <= 0:
            sar_scale = 0.0
        elif 0 > 0:
            sar_scale = min((epoch - 0) / 0, 1.0)
        else:
            sar_scale = 1.0
        guard_weight = (
            args.height_correction_guard_weight
            * min((epoch - 1) / args.height_correction_guard_warmup_epochs, 1.0)
            if args.height_correction_guard_warmup_epochs > 0
            else args.height_correction_guard_weight
        )
        batches = 0
        (mix_seen, mix_attempted, mix_used, mix_ids) = (0, 0, 0, set())
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            ascii=True,
            disable=not sys.stderr.isatty(),
        )
        for batch_index, batch in enumerate(progress, start=1):
            if mix_probability:
                ids = batch["building_mix_id"].tolist()
                mix_seen += len(ids)
                mix_attempted += int(batch["building_mix_attempted"].sum())
                mix_used += sum((identifier > 0 for identifier in ids))
                mix_ids.update((identifier for identifier in ids if identifier > 0))
            rgb = batch["rgb"].to(device, non_blocking=True)
            sar = batch["sar"].to(device, non_blocking=True)
            if 0.0 > 0.0:
                keep_sar = (
                    torch.rand(sar.shape[0], 1, 1, 1, device=sar.device) >= 0.0
                ).to(sar.dtype)
                sar = sar * keep_sar
            mask = batch["mask"].to(device, non_blocking=True)
            contour = batch["contour"].to(device, non_blocking=True)
            height = batch["height"].to(device, non_blocking=True)
            label_valid_height = batch["valid_height"].to(device, non_blocking=True)
            instance = batch["instance"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                output = (
                    model(rgb, sar, sar_scale=sar_scale, return_intermediates=True)
                    if use_height_supervision
                    else model(rgb, sar, sar_scale=sar_scale)
                )
                valid_height = (
                    (mask > 0.5)
                    & (label_valid_height > 0.5)
                    & (height > args.min_height)
                )
                loss_height = masked_height_loss(output["height"], height, valid_height)
                loss_affinity = loss_height.new_zeros(())
                affinity_pos_pairs = affinity_neg_pairs = 0
                if args.building_affinity:
                    from hdbf.dfc_building_affinity import affinity_supervision

                    (loss_affinity, affinity_pos_pairs, affinity_neg_pairs, _, _) = (
                        affinity_supervision(
                            output["building_affinity_logits"], instance, valid_height
                        )
                    )
                loss_head_aux = loss_height.new_zeros(())
                loss_ttci_guard = loss_height.new_zeros(())
                loss_hcr_guard = loss_height.new_zeros(())
                if use_height_supervision:
                    if args.height_head_weight > 0:
                        loss_head_aux = head_supervision(
                            output["height_head"], height, valid_height
                        )
                    if guard_weight > 0:
                        if args.ttci:
                            loss_ttci_guard = correction_safeguard(
                                output["height_head"],
                                output["height_coarse"],
                                height,
                                valid_height,
                            )
                        if args.hcr:
                            loss_hcr_guard = correction_safeguard(
                                output["height_coarse"],
                                output["height_after_hcr"],
                                height,
                                valid_height,
                            )
                loss_hcr_range = (
                    height_range_loss(
                        output["hcr_gate"],
                        height,
                        valid_height,
                        args.height_range_boundaries,
                    )
                    if args.hcr
                    else output["height"].sum() * 0.0
                )
                loss_tail = (
                    directional_tail_loss(
                        output["height"],
                        height,
                        valid_height,
                        args.tail_low_threshold,
                        args.tail_high_threshold,
                    )
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
                    )
                    if args.bihm
                    else output["height"].sum() * 0.0
                )
                loss_roof_shape = instance_roof_shape_loss(
                    output["height"],
                    height,
                    instance,
                    valid_height,
                    args.bihm_min_pixels,
                    args.bihm_min_valid_ratio,
                )
                loss_mask = bce_dice_loss(output["mask"], mask)
                loss_local = loss_mask.new_zeros(())
                loss_branches = loss_mask.new_zeros(())
                if args.local_structure:
                    from hdbf.local_structure import local_structure_loss

                    loss_local = local_structure_loss(
                        output, mask, contour, args.mask_weight, args.contour_weight
                    )
                    loss_branches = (
                        sum(
                            (
                                args.mask_weight
                                * bce_dice_loss(
                                    output[f"unirep_{branch}_mask"].float(), mask
                                )
                                + args.contour_weight
                                * bce_dice_loss(
                                    output[f"unirep_{branch}_contour"].float(), contour
                                )
                                for branch in ("cnn", "semantic")
                            )
                        )
                        / 2
                    )
                loss_contour_bce = F.binary_cross_entropy_with_logits(
                    output["contour"].float(), contour.float()
                )
                loss_contour_tversky = tversky_loss_with_logits(
                    output["contour"],
                    contour,
                    fp_weight=args.contour_tversky_fp_weight,
                    fn_weight=args.contour_tversky_fn_weight,
                )
                loss_contour_far = contour_far_negative_loss(
                    output["contour"],
                    contour,
                    band_width=args.contour_far_band_width,
                    fraction=args.contour_hard_negative_fraction,
                )
                loss_contour = (
                    args.contour_bce_weight * loss_contour_bce
                    + args.contour_tversky_weight * loss_contour_tversky
                    + args.contour_far_weight * loss_contour_far
                )
                loss_contour_consistency = contour_mask_consistency_loss(
                    output["contour"],
                    output["mask"],
                    kernel_size=args.contour_consistency_kernel,
                )
                if args.contour_objective == "balanced":
                    loss_contour = balanced_contour_loss(output["contour"], contour)
                    loss_contour_consistency = loss_contour_consistency * 0.0
                elif args.contour_objective == "bce-dice":
                    loss_contour = bce_dice_loss(
                        output["contour"].float(), contour.float()
                    )
                    loss_contour_consistency = loss_contour_consistency * 0.0
                background = mask <= 0.5
                loss_bg = (
                    output["height"][background].abs().mean()
                    if background.any()
                    else output["height"].sum() * 0.0
                )
                loss = (
                    args.mask_weight * loss_mask
                    + args.contour_weight
                    * (
                        loss_contour
                        + args.contour_consistency_weight * loss_contour_consistency
                    )
                    + args.height_weight * loss_height
                    + args.affinity_loss_weight * loss_affinity
                    + args.height_head_weight * loss_head_aux
                    + guard_weight * (loss_ttci_guard + loss_hcr_guard)
                    + args.hcr_range_weight * loss_hcr_range
                    + tail_weight * loss_tail
                    + bihm_weight * loss_bihm
                    + roof_shape_weight * loss_roof_shape
                    + args.bg_height_weight * loss_bg
                    + args.local_structure_weight * loss_local
                    + args.structure_branch_weight * loss_branches
                )
            if not torch.isfinite(loss):
                components = {
                    "local_structure": loss_local,
                    "structure_branches": loss_branches,
                    "mask": loss_mask,
                    "contour": loss_contour,
                    "contour_bce": loss_contour_bce,
                    "contour_tversky": loss_contour_tversky,
                    "contour_far": loss_contour_far,
                    "contour_consistency": loss_contour_consistency,
                    "height": loss_height,
                    "height_head_aux": loss_head_aux,
                    "ttci_guard": loss_ttci_guard,
                    "hcr_guard": loss_hcr_guard,
                    "hcr_range": loss_hcr_range,
                    "tail": loss_tail,
                    "bihm": loss_bihm,
                    "roof_shape": loss_roof_shape,
                    "background": loss_bg,
                }
                details = ", ".join(
                    (
                        f"{name}={float(value.detach().float()):.6g}"
                        for (name, value) in components.items()
                    )
                )
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}: {details}"
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
                "affinity": loss_affinity.item(),
                "affinity_pos_pairs": affinity_pos_pairs,
                "affinity_neg_pairs": affinity_neg_pairs,
                "loss": loss.item(),
                "mask": loss_mask.item(),
                "contour": loss_contour.item(),
                "contour_bce": loss_contour_bce.item(),
                "contour_tversky": loss_contour_tversky.item(),
                "contour_far": loss_contour_far.item(),
                "contour_consistency": loss_contour_consistency.item(),
                "height": loss_height.item(),
                "height_head_aux": loss_head_aux.item(),
                "ttci_guard": loss_ttci_guard.item(),
                "hcr_guard": loss_hcr_guard.item(),
                "bg": loss_bg.item(),
                "hcr_range": loss_hcr_range.item(),
                "tail": loss_tail.item(),
                "bihm": loss_bihm.item(),
                "roof_shape": loss_roof_shape.item(),
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
            if not sys.stderr.isatty() and (batch_index == 1 or batch_index % 50 == 0):
                print(
                    f"Epoch {epoch}/{args.epochs} batch {batch_index}/{len(train_loader)} loss={loss.item():.4f} height={loss_height.item():.4f} mask={loss_mask.item():.4f} contour={loss_contour.item():.4f} local={loss_local.item():.4f} branches={loss_branches.item():.4f}",
                    flush=True,
                )
        if args.building_affinity and (
            sums["affinity_pos_pairs"] == 0 or sums["affinity_neg_pairs"] == 0
        ):
            raise RuntimeError(
                "No same/different-building affinity pairs in this epoch; inspect instance labels"
            )
        if mix_probability:
            print(
                f"Building mixture epoch {epoch}: samples={mix_seen}; attempted={mix_attempted}; used={mix_used}; unique_IDs={len(mix_ids)}; fallback={mix_attempted - mix_used}"
            )
            with open(
                os.path.join(run_dir, "building_sampling_epochs.jsonl"),
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(
                        dict(
                            epoch=epoch,
                            samples=mix_seen,
                            attempted=mix_attempted,
                            used=mix_used,
                            unique_ids=sorted(mix_ids),
                            fallback=mix_attempted - mix_used,
                        )
                    )
                    + "\n"
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
                height_stages=use_height_supervision,
                val_inference=args.val_inference,
                val_window=args.val_window,
                val_stride=args.val_stride,
            )
        finally:
            ema.restore(model)
        metrics.update({key: value / max(batches, 1) for (key, value) in sums.items()})
        metrics["lr"] = max((group["lr"] for group in optimizer.param_groups))
        if args.geometry_height_fusion:
            print(
                f"Geometry-height train: relative_feature_residual={metrics['height_geometry_residual']:.4f}; gate={metrics['height_geometry_gate']:.4f}; scale_weights={[round(metrics[f'height_geometry_scale_{i}'], 4) for i in range(4)]}"
            )
        print(
            f"Epoch {epoch}: loss={metrics['loss']:.4f} mask_loss={metrics['mask']:.4f} contour_loss={metrics['contour']:.4f} contour_bce={metrics['contour_bce']:.4f} contour_tversky={metrics['contour_tversky']:.4f} contour_far={metrics['contour_far']:.4f} contour_consistency_loss={metrics['contour_consistency']:.4f} height_loss={metrics['height']:.4f} head_aux={metrics['height_head_aux']:.4f} affinity={metrics['affinity']:.4f} ttci_guard={metrics['ttci_guard']:.4f} hcr_guard={metrics['hcr_guard']:.4f} guard_weight={guard_weight:.4f} hcr_range_loss={metrics['hcr_range']:.4f} tail_loss={metrics['tail']:.4f} tail_weight={tail_weight:.4f} bihm_loss={metrics['bihm']:.4f} bihm_weight={bihm_weight:.4f} roof_shape_loss={metrics['roof_shape']:.4f} roof_shape_weight={roof_shape_weight:.4f} sar_scale={sar_scale:.3f} bg_loss={metrics['bg']:.4f} | val_f1={metrics['f1']:.4f} val_contour_f1={metrics['contour_f1']:.4f} val_contour_precision={metrics['contour_precision']:.4f} val_contour_recall={metrics['contour_recall']:.4f} val_boundary_f1={metrics['boundary_f1']:.4f} val_boundary_recall={metrics['boundary_recall']:.4f} val_mae={metrics['mae']:.4f} val_rmse={metrics['rmse']:.4f} lr={metrics['lr']:.6g}"
            + ""
            + ""
        )
        if use_height_supervision:
            print(
                f"Height stages val: head_mae={metrics['height_head_mae']:.4f} coarse_mae={metrics['height_coarse_mae']:.4f} hcr_mae={metrics['height_after_hcr_mae']:.4f} final_mae={metrics['mae']:.4f}"
            )
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
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
        with open(
            os.path.join(run_dir, "training_metrics.jsonl"), "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(dict(epoch=epoch, metrics=metrics)) + "\n")
        if args.save_best_mae_only:
            if args.save_task_best:
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    save_checkpoint(
                        os.path.join(run_dir, "checkpoint_best_f1.pth"),
                        model,
                        ema,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        metrics,
                        args,
                    )
                if metrics["contour_f1"] > best_contour_f1:
                    best_contour_f1 = metrics["contour_f1"]
                    save_checkpoint(
                        os.path.join(run_dir, "checkpoint_best_contour.pth"),
                        model,
                        ema,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        metrics,
                        args,
                    )
            continue
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_checkpoint(
                os.path.join(run_dir, "checkpoint_best_f1.pth"),
                model,
                ema,
                optimizer,
                scheduler,
                scaler,
                epoch,
                metrics,
                args,
            )
        if metrics["contour_f1"] > best_contour_f1:
            best_contour_f1 = metrics["contour_f1"]
            save_checkpoint(
                os.path.join(run_dir, "checkpoint_best_contour.pth"),
                model,
                ema,
                optimizer,
                scheduler,
                scaler,
                epoch,
                metrics,
                args,
            )
        if metrics["boundary_f1"] > best_boundary_f1:
            best_boundary_f1 = metrics["boundary_f1"]
            save_checkpoint(
                os.path.join(run_dir, "checkpoint_best_boundary.pth"),
                model,
                ema,
                optimizer,
                scheduler,
                scaler,
                epoch,
                metrics,
                args,
            )
        if (
            metrics["f1"] >= args.checkpoint_min_mask_f1
            and metrics["contour_f1"] >= args.checkpoint_min_contour_f1
            and (metrics["mae"] < best_balanced_mae)
        ):
            best_balanced_mae = metrics["mae"]
            save_checkpoint(
                os.path.join(run_dir, "checkpoint_best_balanced_mae.pth"),
                model,
                ema,
                optimizer,
                scheduler,
                scaler,
                epoch,
                metrics,
                args,
            )
        training_state = None
        last_path = os.path.join(run_dir, "checkpoint_last.pth")
        save_checkpoint(
            last_path + ".tmp",
            model,
            ema,
            optimizer,
            scheduler,
            scaler,
            epoch,
            metrics,
            args,
            training_state=training_state,
        )
        os.replace(last_path + ".tmp", last_path)
    if end_epoch < args.epochs:
        print(
            f"Observation boundary reached at epoch {end_epoch}; continue with --resume {os.path.join(run_dir, 'checkpoint_last.pth')} (schedule horizon {args.epochs})."
        )


if __name__ == "__main__":
    main()
