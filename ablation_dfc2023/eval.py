"""Nine-metric native-tile evaluation, matching the full DFC2023 evaluator."""
import argparse
import contextlib
import hashlib
import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from ablation_dfc2023.model import restore_model
from hdbf.dataset_dfc2023 import DFC2023Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DFC2023 ablation")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled, min_height):
    tp = fp = fn = ctp = cfp = cfn = 0
    count = ae = se = rel = d1 = d2 = d3 = 0
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        sar = None if model.rgb_only else batch["sar"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True) > .5
        contour = batch["contour"].to(device, non_blocking=True) > .5
        target = batch["height"].to(device, non_blocking=True).float()
        valid = batch["valid_height"].to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            output = model(rgb, sar, return_intermediates=hasattr(model, "unirep_structure"))
        pred_mask = torch.sigmoid(output["mask"]) > .5
        pred_contour = torch.sigmoid(output["contour"]) > .5
        tp += (pred_mask & mask).sum().item()
        fp += (pred_mask & ~mask).sum().item()
        fn += (~pred_mask & mask).sum().item()
        ctp += (pred_contour & contour).sum().item()
        cfp += (pred_contour & ~contour).sum().item()
        cfn += (~pred_contour & contour).sum().item()

        valid = valid & mask & torch.isfinite(target) & (target > min_height)
        pred = output["height"].float()[valid]
        gt = target[valid]
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Non-finite height on valid building pixels")
        pred, gt = pred.clamp_min(1e-6), gt.clamp_min(1e-6)
        error = pred - gt
        count += pred.numel()
        ae += error.abs().sum().item()
        se += error.square().sum().item()
        rel += (error.abs() / gt).sum().item()
        ratio = torch.maximum(pred / gt, gt / pred)
        d1 += (ratio < 1.25).sum().item()
        d2 += (ratio < 1.25 ** 2).sum().item()
        d3 += (ratio < 1.25 ** 3).sum().item()
    if not count:
        raise ValueError("No reliable building height pixels in the evaluation split")
    precision = tp / max(tp + fp, 1.)
    recall = tp / max(tp + fn, 1.)
    mask_f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1.)
    contour_f1 = 2 * ctp / max(2 * ctp + cfp + cfn, 1.)
    return mask_f1, iou, contour_f1, rel / count, (se / count) ** .5, d1 / count, d2 / count, d3 / count, ae / count


def format_metrics(values):
    header = "| MaskF1 | MaskIoU | ContourF1 | AbsRel | RMSE(m) | \u03b41 | \u03b42 | \u03b43 | MAE(m) |"
    return "\n".join((header, "| " + " | ".join(["---"] * 9) + " |",
                      "| " + " | ".join(f"{value:.4f}" for value in values) + " |"))


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("evaluation_weights") != "EMA":
        raise ValueError("This ablation evaluator expects the saved EMA checkpoint")
    data = checkpoint["data_config"]
    split_dir = args.split_dir or args.base_dir / "dfc23_newsplit"
    split_file = split_dir / f"{args.split}.txt"
    if hashlib.sha256(split_file.read_bytes()).hexdigest() != data["split_hashes"][args.split]:
        raise ValueError("Evaluation manifest differs from the training checkpoint")
    if (data["min_height"], data["sar_clip"], data["rgb_normalization"]) != (.1, .5, "imagenet"):
        raise ValueError("Unexpected DFC2023 data normalization or height validity settings")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with contextlib.redirect_stdout(io.StringIO()):
        model = restore_model(checkpoint).to(device).eval()
    del checkpoint
    dataset = DFC2023Dataset(str(args.base_dir), str(split_file), train=False,
                            sar_clip=data["sar_clip"], rgb_normalization=data["rgb_normalization"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    report = format_metrics(evaluate(model, loader, device, args.amp and device.type == "cuda", data["min_height"]))
    output = args.output or args.checkpoint.parent / f"{args.split}_best_mae.txt"
    output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
