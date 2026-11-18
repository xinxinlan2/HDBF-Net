import bootstrap
import argparse
from hdbf.dfc_height_distribution import range_confusion
import hashlib
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from hdbf.dataset_dfc2023 import DFC2023Dataset, instances_to_targets
from semseg.models.hdbf_net import (
    DinoHDBF as CMNeXtMTL,
    checkpoint_fusion_type,
    checkpoint_task_split,
)


def parse_args():
    parser = argparse.ArgumentParser("Evaluate DINOv2 HDBF-Net on DFC2023")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-height", type=float, default=None)
    parser.add_argument("--sar-clip", type=float, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--height-diagnostics",
        action="store_true",
        help="Also report native-resolution height stages by city and reference-height range.",
    )
    parser.set_defaults(old_test_overlap_diagnostics=False)
    result = parser.parse_args()
    if getattr(result, "height_transfer_audit", False) != False:
        parser.error("Unsupported historical option: height_transfer_audit")
    if getattr(result, "height_optimization_audit", False) != False:
        parser.error("Unsupported historical option: height_optimization_audit")
    if getattr(result, "high_building_diagnosis", False) != False:
        parser.error("Unsupported historical option: high_building_diagnosis")
    if getattr(result, "height_audit", False) != False:
        parser.error("Unsupported historical option: height_audit")
    if getattr(result, "height_cases_csv", None) != None:
        parser.error("Unsupported historical option: height_cases_csv")
    if getattr(result, "height_case_export_dir", None) != None:
        parser.error("Unsupported historical option: height_case_export_dir")
    if getattr(result, "compare_windows", False) != False:
        parser.error("Unsupported historical option: compare_windows")
    if getattr(result, "window_stages", False) != False:
        parser.error("Unsupported historical option: window_stages")
    if getattr(result, "diagnostics", False) != False:
        parser.error("Unsupported historical option: diagnostics")
    if getattr(result, "contour_spatial", False) != False:
        parser.error("Unsupported historical option: contour_spatial")
    if getattr(result, "export_contour_dir", None) != None:
        parser.error("Unsupported historical option: export_contour_dir")
    if getattr(result, "recalibrate_bn", False) != False:
        parser.error("Unsupported historical option: recalibrate_bn")
    if getattr(result, "save_calibrated_checkpoint", None) != None:
        parser.error("Unsupported historical option: save_calibrated_checkpoint")
    if getattr(result, "old_test_overlap_diagnostics", False) != False:
        parser.error("Unsupported historical option: old_test_overlap_diagnostics")
    return result


def empty_accumulator():
    return {
        "n": 0.0,
        "ae": 0.0,
        "se": 0.0,
        "rel": 0.0,
        "rel_n": 0.0,
        "d1": 0.0,
        "d2": 0.0,
        "d3": 0.0,
    }


def update_height(accumulator, prediction, target, valid, min_height):
    valid = valid & torch.isfinite(target)
    if not torch.isfinite(prediction[valid]).all():
        raise FloatingPointError(
            "Non-finite height predictions in the evaluation region."
        )
    count = valid.sum().item()
    if count == 0:
        return
    pred = prediction[valid].float().clamp_min(1e-06)
    gt = target[valid].float().clamp_min(1e-06)
    error = pred - gt
    accumulator["n"] += count
    accumulator["ae"] += error.abs().sum().item()
    accumulator["se"] += error.square().sum().item()
    reliable = gt > min_height
    if reliable.any():
        pred = pred[reliable]
        gt = gt[reliable]
        ratio = torch.maximum(pred / gt, gt / pred)
        accumulator["rel_n"] += reliable.sum().item()
        accumulator["rel"] += ((pred - gt).abs() / gt).sum().item()
        accumulator["d1"] += (ratio < 1.25).sum().item()
        accumulator["d2"] += (ratio < 1.25**2).sum().item()
        accumulator["d3"] += (ratio < 1.25**3).sum().item()


def finalize(accumulator):
    count = max(accumulator["n"], 1.0)
    relative_count = max(accumulator["rel_n"], 1.0)
    return {
        "pixels": int(accumulator["n"]),
        "rel_pixels": int(accumulator["rel_n"]),
        "MAE": accumulator["ae"] / count,
        "RMSE": (accumulator["se"] / count) ** 0.5,
        "AbsRel": accumulator["rel"] / relative_count,
        "d1": accumulator["d1"] / relative_count,
        "d2": accumulator["d2"] / relative_count,
        "d3": accumulator["d3"] / relative_count,
    }


HEIGHT_STAGES = (
    ("ContextBase", "height_context_base"),
    ("Head", "height_head"),
    ("TTCI", "height_coarse"),
    ("HCR", "height_after_hcr"),
    ("SLMHC", "height_after_slmhc"),
    ("Final", "height"),
)


def city_from_filename(filename):
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    parts = name.split("_")
    return parts[1] if len(parts) >= 4 and parts[0] in ("SV", "GF2") else "Unknown"


def sensor_from_filename(filename):
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    prefix = name.split("_", 1)[0]
    return prefix if prefix in ("SV", "GF2") else "Unknown"


def footprint_interior(footprint, width=4):
    padded = F.pad(footprint.float(), (width,) * 4, value=0.0)
    return -F.max_pool2d(-padded, 2 * width + 1, stride=1) > 0.5


def legacy_polygon_masks(annotation_map, filenames, shape, device):
    masks = []
    for name in filenames:
        polygons = [
            seg for seg in annotation_map.get(name, []) if not isinstance(seg, dict)
        ]
        (mask, _) = instances_to_targets(polygons, *shape)
        masks.append(torch.from_numpy(mask.copy()).bool())
    return torch.stack(masks).unsqueeze(1).to(device)


def update_diagnostic(accumulator, prediction, target, valid):
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Height diagnostics require aligned full-resolution arrays.")
    if not valid.any():
        return
    pred = prediction[valid].double().clamp_min(1e-06)
    gt = target[valid].double()
    if not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
        raise FloatingPointError("Non-finite values in the height diagnostic region.")
    error = pred - gt
    accumulator["n"] += gt.numel()
    accumulator["ae"] += error.abs().sum().item()
    accumulator["se"] += error.square().sum().item()
    accumulator["bias"] += error.sum().item()
    accumulator["gt"] += gt.sum().item()
    accumulator["pred"] += pred.sum().item()


def collect_height_diagnostics(
    rows, output, target, valid, filenames, footprint=None, legacy_mask=None
):
    if len(filenames) != target.shape[0]:
        raise ValueError("One source filename is required per image.")
    groups = [("All", valid)]
    for sensor in sorted({sensor_from_filename(name) for name in filenames}):
        selection = torch.tensor(
            [sensor_from_filename(name) == sensor for name in filenames],
            device=valid.device,
            dtype=torch.bool,
        ).reshape(-1, 1, 1, 1)
        groups.append(("Sensor:" + sensor, valid & selection))
    for city in sorted({city_from_filename(name) for name in filenames}):
        selection = torch.tensor(
            [city_from_filename(name) == city for name in filenames],
            device=valid.device,
            dtype=torch.bool,
        ).reshape(-1, 1, 1, 1)
        groups.append(("City:" + city, valid & selection))
    for city, sensor in sorted(
        {(city_from_filename(n), sensor_from_filename(n)) for n in filenames}
    ):
        selection = torch.tensor(
            [
                (city_from_filename(n), sensor_from_filename(n)) == (city, sensor)
                for n in filenames
            ],
            device=valid.device,
            dtype=torch.bool,
        ).reshape(-1, 1, 1, 1)
        groups.append((f"CitySensor:{city}/{sensor}", valid & selection))
    if footprint is not None:
        if footprint.shape != target.shape:
            raise ValueError(
                "Footprint diagnostics require the full reference footprint grid."
            )
        interior = footprint_interior(footprint)
        groups.extend(
            (
                ("Region:RoofInterior>4px", valid & interior),
                ("Region:RoofEdge<=4px", valid & ~interior),
            )
        )
    if legacy_mask is not None:
        if legacy_mask.shape != target.shape:
            raise ValueError(
                "Legacy label diagnostics require aligned reference masks."
            )
        groups.extend(
            (
                ("Label:LegacyPolygon", valid & legacy_mask),
                ("Label:RLEAdded", valid & ~legacy_mask),
            )
        )
    ranges = (
        ("<3m", target < 3.0),
        ("3-6m", (target >= 3.0) & (target < 6.0)),
        ("6-10m", (target >= 6.0) & (target < 10.0)),
        ("10-20m", (target >= 10.0) & (target < 20.0)),
        (">=20m", target >= 20.0),
    )
    groups.extend(
        (("Range:" + name, valid & selection) for (name, selection) in ranges)
    )
    for stage, key in HEIGHT_STAGES:
        if key not in output:
            continue
        prediction = output[key]
        for group, selection in groups:
            accumulator = rows.setdefault(
                (group, stage),
                {"n": 0, "ae": 0.0, "se": 0.0, "bias": 0.0, "gt": 0.0, "pred": 0.0},
            )
            update_diagnostic(accumulator, prediction, target, selection)


def height_diagnostic_report(rows, checkpoint, split, split_file, min_height):
    lines = [
        f"Checkpoint: {checkpoint}",
        f"Split: {split} | Split file: {split_file}",
        "Same saved weights, native tiles, no augmentation, BN fixed; no retraining.",
        f"Region: footprint & source validity & finite height & height > {min_height:g} m.",
        "Ranges are assigned by reference pixel height, not building mean height.",
        "Cities are parsed from SV/GF2_<city>_<latitude>_<longitude> filenames.",
        "Roof-edge groups use a four-pixel square erosion of the GT footprint, not the valid-height mask.",
        "LegacyPolygon/RLEAdded partition the CURRENT valid pixels; they are diagnostics, not a changed metric.",
        "Stage metrics are descriptive; disabling a trained module is not a retrained ablation.",
        "Bias = prediction - reference; all heights/errors are in metres.",
        "",
        "Group\tStage\tPixels\tMAE\tRMSE\tBias\tGT_mean\tPred_mean",
    ]
    groups = ["All"] + sorted(
        {group for (group, _) in rows if group.startswith("City:")}
    )
    for prefix in ("Sensor:", "CitySensor:", "Region:", "Label:"):
        groups += sorted({group for (group, _) in rows if group.startswith(prefix)})
    groups += ["Range:<3m", "Range:3-6m", "Range:6-10m", "Range:10-20m", "Range:>=20m"]
    for group in groups:
        for stage, _ in HEIGHT_STAGES:
            value = rows.get((group, stage))
            if value is None or value["n"] == 0:
                lines.append(f"{group}\t{stage}\t0\tNA\tNA\tNA\tNA\tNA")
                continue
            n = value["n"]
            lines.append(
                f"{group}\t{stage}\t{n}\t{value['ae'] / n:.4f}\t{(value['se'] / n) ** 0.5:.4f}\t{value['bias'] / n:+.4f}\t{value['gt'] / n:.4f}\t{value['pred'] / n:.4f}"
            )
    return "\n".join(lines)


def main():
    args = parse_args()
    base_dir = args.base_dir or os.environ.get("DFC23_BASE", "D:\\znx\\data\\dfc23")
    split_dir = args.split_dir or os.path.join(base_dir, "dfc23_dstfnet_split")
    split_file = os.path.join(split_dir, f"{args.split}.txt")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and device.type == "cuda"
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint.get("args", {})
    data_config = checkpoint.get("data_config", {})
    if args.min_height is None:
        args.min_height = data_config.get(
            "min_height", train_args.get("min_height", 0.1)
        )
    if args.sar_clip is None:
        args.sar_clip = data_config.get("sar_clip", train_args.get("sar_clip", 0.5))
    expected_hash = data_config.get(
        "split_hashes", train_args.get("split_hashes", {})
    ).get(args.split)
    if expected_hash:
        with open(split_file, "rb") as stream:
            actual_hash = hashlib.sha256(stream.read()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                "Evaluation split differs from the checkpoint's training manifest."
            )
    model_config = checkpoint.get("model_config", {})
    use_unirep = model_config.get(
        "unirep_structure", bool(train_args.get("unirep_pretrained"))
    )
    query_structure = model_config.get(
        "query_structure", train_args.get("query_structure", False)
    )
    local_structure = model_config.get(
        "local_structure", train_args.get("local_structure", False)
    )
    geometry_height_fusion = model_config.get(
        "geometry_height_fusion", train_args.get("geometry_height_fusion", False)
    )
    geometry_height_mode = model_config.get(
        "geometry_height_mode", train_args.get("geometry_height_mode", "joint-global")
    )
    if (
        geometry_height_fusion
        and "geometry_height_mode" in model_config
        and ("geometry_height_mode" in train_args)
        and (model_config["geometry_height_mode"] != train_args["geometry_height_mode"])
    ):
        raise ValueError(
            "Conflicting geometry-height mode in checkpoint config and args"
        )
    if geometry_height_fusion and (not local_structure):
        raise ValueError("Geometry-height conditioning requires Local UniRep")
    geometry_interaction = model_config.get(
        "geometry_interaction", train_args.get("geometry_interaction", False)
    )
    if geometry_interaction and (not use_unirep):
        raise ValueError("Geometry interaction checkpoint requires UniRep structure")
    if query_structure and (not use_unirep or geometry_interaction):
        raise ValueError(
            "Query structure requires UniRep without dual-scale geometry interaction"
        )
    if local_structure and (not use_unirep or query_structure or geometry_interaction):
        raise ValueError(
            "Local structure requires UniRep without query/geometry interaction"
        )
    fusion_type = checkpoint_fusion_type(model_config, train_args)
    print(f"Fusion: {fusion_type}")
    segmentation_refinement = model_config.get(
        "segmentation_refinement", train_args.get("segmentation_refinement", False)
    )
    print(f"Full-resolution segmentation refinement: {segmentation_refinement}")
    task_split_decoder = checkpoint_task_split(model_config, train_args)
    print(
        f"Decoder: {('task-split-pyramid' if task_split_decoder else 'shared-segformer')}"
    )
    variant = model_config.get("variant", train_args.get("variant", "B4"))
    decoder_dim = model_config.get("decoder_dim", train_args.get("decoder_dim", 512))
    use_hrbr = model_config.get("use_hrbr", train_args.get("hrbr", False))
    use_hcr = model_config.get("use_hcr", train_args.get("hcr", False))
    use_slmhc = model_config.get("use_slmhc", train_args.get("slmhc", False))
    use_bihm = model_config.get("use_bihm", train_args.get("bihm", False))
    use_ttci = model_config.get("use_ttci", train_args.get("ttci", False))
    task_channels = model_config.get(
        "task_channels", train_args.get("task_channels", 256)
    )
    model_state = checkpoint["model_state_dict"]
    ttci_legacy = use_ttci and (
        not any(
            (key.startswith("task_interaction.mask_correction") for key in model_state)
        )
    )
    lr_schedule = train_args.get("lr_schedule")
    if lr_schedule is None:
        lr_schedule = "poly" if train_args.get("poly_power", 0.9) < 0.9 else None
    use_ema = (
        checkpoint.get(
            "evaluation_weights",
            "EMA" if train_args.get("ema_decay", 0.0) > 0.0 else "raw",
        )
        == "EMA"
    )
    model = CMNeXtMTL(
        geometry_interaction=geometry_interaction,
        fusion_type=fusion_type,
        segmentation_refinement=segmentation_refinement,
        task_split_decoder=task_split_decoder,
        gradient_checkpointing=False,
        rgb_only=model_config.get("rgb_only", train_args.get("rgb_only", False)),
        variant=variant,
        decoder_dim=decoder_dim,
        use_hrbr=use_hrbr,
        use_hcr=use_hcr,
        use_slmhc=use_slmhc,
        use_ttci=use_ttci,
        task_channels=task_channels,
        ttci_legacy=ttci_legacy,
        height_priority=model_config.get(
            "height_priority", train_args.get("height_priority", False)
        ),
        distribution_height=model_config.get(
            "distribution_height", train_args.get("distribution_height", False)
        ),
        context_detail_height=model_config.get(
            "context_detail_height", train_args.get("context_detail_height", False)
        ),
        building_affinity=model_config.get(
            "building_affinity", train_args.get("building_affinity", False)
        ),
    )
    if use_unirep:
        from hdbf.unirep_structure import UniRepStructure

        model.unirep_structure = UniRepStructure(
            model.backbone.channels,
            decoder_dim,
            query_structure=query_structure,
            local_structure=local_structure,
        )
        print(
            "Structure: UniRepLKNet-S ADE20K; native RGB; no post-training BN calibration"
        )
    if geometry_height_fusion:
        from hdbf.geometry_height import GeometryHeightFusion

        model.height_geometry_fusion = GeometryHeightFusion(
            decoder_dim, mode=geometry_height_mode
        )
    model.load_state_dict(model_state)
    if geometry_height_fusion:
        print(
            f"Geometry-height mode: {geometry_height_mode}; selected by checkpoint metadata"
        )
    if local_structure:
        print(
            f"Structure: 192-channel dense local decoder; quarter/half/native reconstruction; saved height guidance={model.unirep_structure.local_decoder.guidance_strength().item():.3f}; independent contour, no post-training BN calibration"
        )
    if query_structure:
        print(
            f"Structure: semantic region queries + native point refinement; independent contour; saved height guidance={model.unirep_structure.point_refiner.guidance_strength().item():.3f}; original height path; no GT input or BN recalibration"
        )
    if geometry_interaction:
        print(
            f"Geometry interaction: 1/8 + 1/4 TTCI, native shape decoding, saved guidance={model.geometry_interaction.guidance_strength().item():.3f}; final structure guides height; no GT input or BN recalibration"
        )
        print(
            "Semantic auxiliary diagnostics now refer to initial pre-interaction logits, not the previous model's post-TTCI logits; neither is a retrained ablation"
        )
    if model_config.get("height_priority", train_args.get("height_priority", False)):
        print(
            "Height priority: independent height decoder; structure gradients detached from encoder"
        )
    model.to(device).eval()
    rgb_normalization = data_config.get(
        "rgb_normalization", train_args.get("rgb_normalization", "none")
    )
    dataset = DFC2023Dataset(
        base_dir,
        split_file,
        train=False,
        sar_clip=args.sar_clip,
        rgb_normalization=rgb_normalization,
    )
    print(f"RGB normalization from checkpoint: {rgb_normalization}")
    overlap_diagnostic = None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    full = empty_accumulator()
    height_positive = empty_accumulator()
    building = empty_accumulator()
    building_reliable = empty_accumulator()
    tp = fp = fn = 0.0
    contour_tp = contour_fp = contour_fn = 0.0
    before_counts = {"mask": [0.0, 0.0, 0.0], "contour": [0.0, 0.0, 0.0]}
    diagnostic_rows = {}
    height_range_confusion = None
    branch_counts = {
        f"{b}_{t}": [0, 0, 0] for b in ("cnn", "semantic") for t in ("mask", "contour")
    }
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{args.split}: metrics", ascii=True):
            rgb = batch["rgb"].to(device, non_blocking=True)
            sar = batch["sar"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True) > 0.5
            contour = batch["contour"].to(device, non_blocking=True) > 0.5
            target_height = batch["height"].to(device, non_blocking=True).float()
            label_valid = batch["valid_height"].to(device, non_blocking=True)
            with autocast(enabled=amp_enabled):
                output = (
                    model(rgb, sar, return_intermediates=True)
                    if args.height_diagnostics or segmentation_refinement or use_unirep
                    else model(rgb, sar)
                )
            pred_height = output["height"].float()
            pred_mask = torch.sigmoid(output["mask"]) > 0.5
            pred_contour = torch.sigmoid(output["contour"]) > 0.5
            if use_unirep:
                for key, counts in branch_counts.items():
                    p = output["unirep_" + key] > 0
                    truth = mask if key.endswith("_mask") else contour
                    for i, v in enumerate(
                        ((p & truth).sum(), (p & ~truth).sum(), (~p & truth).sum())
                    ):
                        counts[i] += v.item()
            if segmentation_refinement:
                for task, target in (("mask", mask), ("contour", contour)):
                    before = output[task + "_before_refinement"].sigmoid() > 0.5
                    counts = before_counts[task]
                    counts[0] += (before & target).sum().item()
                    counts[1] += (before & ~target).sum().item()
                    counts[2] += (~before & target).sum().item()
            height_mask = (
                label_valid
                & torch.isfinite(target_height)
                & (target_height > args.min_height)
            )
            if "height_range_logits" in output:
                current = range_confusion(
                    output, target_height, mask & height_mask
                ).cpu()
                height_range_confusion = (
                    current
                    if height_range_confusion is None
                    else height_range_confusion + current
                )
            update_height(
                full, pred_height, target_height, label_valid, args.min_height
            )
            update_height(
                height_positive,
                pred_height,
                target_height,
                height_mask,
                args.min_height,
            )
            update_height(
                building,
                pred_height,
                target_height,
                mask & label_valid,
                args.min_height,
            )
            update_height(
                building_reliable,
                pred_height,
                target_height,
                mask & height_mask,
                args.min_height,
            )
            if overlap_diagnostic is not None:
                overlap_diagnostic.update(
                    batch["filename"], pred_height, target_height, mask & height_mask
                )
            if args.height_diagnostics:
                legacy_mask = (
                    legacy_polygon_masks(
                        dataset.annotation_map,
                        batch["filename"],
                        target_height.shape[-2:],
                        device,
                    )
                    if hasattr(dataset, "annotation_map")
                    else None
                )
                collect_height_diagnostics(
                    diagnostic_rows,
                    output,
                    target_height,
                    mask & height_mask,
                    batch["filename"],
                    footprint=mask,
                    legacy_mask=legacy_mask,
                )
            tp += (pred_mask & mask).sum().item()
            fp += (pred_mask & ~mask).sum().item()
            fn += (~pred_mask & mask).sum().item()
            contour_tp += (pred_contour & contour).sum().item()
            contour_fp += (pred_contour & ~contour).sum().item()
            contour_fn += (~pred_contour & contour).sum().item()
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-08)
    iou = tp / max(tp + fp + fn, 1.0)
    contour_f1 = 2 * contour_tp / max(2 * contour_tp + contour_fp + contour_fn, 1.0)
    if building_reliable["n"] == 0:
        raise ValueError("Evaluation contains no valid building-height pixels.")
    reliable_metrics = finalize(building_reliable)
    inferred_model_name = f"CMNeXt-MTL-{variant}{('-HRBR' if use_hrbr else '')}{('-HCR' if use_hcr else '')}{('-SLMHC' if use_slmhc else '')}{('-BIHM' if use_bihm else '')}{('-TTCI' if use_ttci else '')}{('-Cosine' if lr_schedule == 'cosine' else '')}{('-SlowPoly' if lr_schedule == 'poly' else '')}{('-EMA' if use_ema else '')}"
    model_name = checkpoint.get("model_name", inferred_model_name)
    method_name = args.method_name or model_name
    lines = [
        f"Checkpoint: {args.checkpoint}",
        f"Split: {args.split} | Weights: {('EMA' if use_ema else 'raw')} model_state_dict | AMP: {amp_enabled}",
        f"Split file: {split_file}",
        "Height validity: footprint & source validity & finite height & height > min_height",
    ]
    if "epoch" in checkpoint:
        lines.append(f"Loaded checkpoint epoch: {checkpoint['epoch']}")
    if use_unirep:
        lines.append(
            "UniRepLKNet-S + ADE20K feature decoder; new binary heads; native RGB detail"
        )
        if local_structure:
            lines.append(
                "Dense local structure: width192, learned upsampling, two local-affinity steps; thresholds=0.5"
            )
            lines.append(
                f"Height guidance={model.unirep_structure.local_decoder.guidance_strength().item():.3f}; original height head/corrections, no GT input or BN recalibration"
            )
            if geometry_height_fusion:
                lines.append(
                    f"Geometry-height mode: {geometry_height_mode}; selected by checkpoint metadata"
                )
        if query_structure:
            lines.append(
                "Semantic query decoder + native point refinement; thresholds=0.5; original height path"
            )
            lines.append(
                f"Predicted-height guidance={model.unirep_structure.point_refiner.guidance_strength().item():.3f}; no GT input or post-training BN recalibration"
            )
        lines.append(
            "Joint-model path diagnostics, not independently retrained ablations:"
        )
        for key, (btp, bfp, bfn) in branch_counts.items():
            lines.append(f"{key} F1={2 * btp / max(2 * btp + bfp + bfn, 1):.4f}")
    if "epoch" in checkpoint:
        lines.append(f"Loaded checkpoint epoch: {checkpoint['epoch']}")
    if model_config.get(
        "context_detail_height", train_args.get("context_detail_height", False)
    ):
        lines.append(
            "Context-detail height head: active | continuous base + bounded local detail"
        )
    if height_range_confusion is not None:
        accuracy = height_range_confusion.diag().sum().item() / max(
            height_range_confusion.sum().item(), 1
        )
        lines.append(
            f"Distribution height head: active | valid-pixel range accuracy={accuracy:.4f}"
        )
        lines.append(
            "Range confusion matrix: rows=GT range, columns=predicted range; 16 log1p bins + open tail"
        )
        lines.extend(
            (
                " ".join((str(int(v)) for v in row))
                for row in height_range_confusion.tolist()
            )
        )
    lines.extend(
        [
            f"Model: {model_name} | Input: {('RGB' if model.rgb_only else 'RGB+SAR')}",
            "",
            "Markdown row candidate:",
            f"| {method_name} | {('RGB' if model.rgb_only else 'RGB+SAR')} | {reliable_metrics['AbsRel']:.4f} | {reliable_metrics['RMSE']:.4f} | {reliable_metrics['d1']:.4f} | {reliable_metrics['d2']:.4f} | {reliable_metrics['d3']:.4f} | {reliable_metrics['MAE']:.4f} |",
            "",
            "[Mask]",
            f"F1={f1:.4f} Precision={precision:.4f} Recall={recall:.4f} IoU={iou:.4f}",
            "",
            "[Contour]",
            f"F1={contour_f1:.4f}",
        ]
    )
    for name, accumulator in (
        ("Full Image", full),
        (f"GT Height > {args.min_height}m", height_positive),
        ("Building Mask All Pixels", building),
        (f"Building Mask + GT Height > {args.min_height}m", building_reliable),
    ):
        metrics = finalize(accumulator)
        lines.extend(
            [
                "",
                f"[{name}]",
                f"pixels={metrics['pixels']}  rel_pixels={metrics['rel_pixels']}  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  AbsRel={metrics['AbsRel']:.4f}  d1={metrics['d1']:.4f}  d2={metrics['d2']:.4f}  d3={metrics['d3']:.4f}",
            ]
        )
    if segmentation_refinement:
        lines.extend(
            ["", "[Segmentation refinement: same checkpoint, not retrained ablation]"]
        )
        for task, after in (("mask", f1), ("contour", contour_f1)):
            (a, b, c) = before_counts[task]
            before_f1 = 2 * a / max(2 * a + b + c, 1.0)
            lines.append(
                f"{task}: before_F1={before_f1:.4f} after_F1={after:.4f} delta={after - before_f1:+.4f}"
            )
    report = "\n".join(lines)
    print("\n" + report)
    default_name = f"{args.split}_metrics.txt"
    output_path = args.output or os.path.join(
        os.path.dirname(args.checkpoint), default_name
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved to {output_path}")
    if overlap_diagnostic is not None:
        overlap_report = overlap_diagnostic.report(args.checkpoint)
        overlap_path = os.path.splitext(output_path)[0] + "_coverage.txt"
        with open(overlap_path, "w", encoding="utf-8") as stream:
            stream.write(overlap_report + "\n")
        print("\n[Old test: spatial coverage groups]")
        print(
            "\n".join(
                (
                    line
                    for line in overlap_report.splitlines()
                    if line.startswith("LabelSet\t") or "\tAll\t" in line
                )
            )
        )
        print(f"\nSaved overlap diagnostics to {overlap_path}")
    if args.height_diagnostics:
        diagnostic_report = height_diagnostic_report(
            diagnostic_rows, args.checkpoint, args.split, split_file, args.min_height
        )
        diagnostic_path = os.path.splitext(output_path)[0] + "_height_diagnostics.txt"
        with open(diagnostic_path, "w", encoding="utf-8") as stream:
            stream.write(diagnostic_report + "\n")
        print("\n[Height stages, same valid pixels]")
        print(
            "\n".join(
                (
                    line
                    for line in diagnostic_report.splitlines()
                    if line.startswith("Group\t") or line.startswith("All\t")
                )
            )
        )
        print("\n[Final height: city, sensor, roof region and label source]")
        print(
            "\n".join(
                (
                    line
                    for line in diagnostic_report.splitlines()
                    if "\tFinal\t" in line and (not line.startswith("All\t"))
                )
            )
        )
        print(f"\nSaved height diagnostics to {diagnostic_path}")


if __name__ == "__main__":
    main()
