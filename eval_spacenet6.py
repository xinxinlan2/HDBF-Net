import argparse
import json
import os
import bootstrap
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from hdbf.dataset_spacenet6 import SpaceNet6HeightDataset, split_metadata
from semseg.models.cmnext_mtl import CMNeXtMTL


def parse_args():
    parser = argparse.ArgumentParser("Evaluate HDBF-Net on SpaceNet6 height labels")
    parser.set_defaults(height_transfer_audit=False)
    parser.set_defaults(height_optimization_audit=False)
    parser.add_argument(
        "--optimization-config",
        default=None,
        help="Original effective_config.json; required if absent beside the checkpoint.",
    )
    parser.add_argument("--audit-limit", type=int, default=32)
    parser.add_argument("--audit-seed", type=int, default=42)
    parser.add_argument(
        "--contour-threshold",
        type=float,
        default=0.5,
        help="Contour decision threshold; choose on val, never tune on test. Mask stays 0.5.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-height", type=float, default=0.1)
    parser.add_argument(
        "--sar-clip",
        type=float,
        default=None,
        help="Override checkpoint SAR clipping value.",
    )
    parser.add_argument(
        "--sar-normalization",
        choices=["linear", "log"],
        default=None,
        help="Override checkpoint SAR preprocessing. Old checkpoints default to legacy log scaling.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    parser.set_defaults(save_calibrated_checkpoint=None)
    parser.set_defaults(compare_windows=False)
    parser.set_defaults(window_stages=False)
    parser.set_defaults(high_building_diagnosis=False)
    parser.add_argument(
        "--sliding-window",
        action="store_true",
        help="Formal sliding-window evaluation using mean heights/probabilities.",
    )
    parser.set_defaults(height_audit=False)
    parser.set_defaults(height_cases_csv=None)
    parser.set_defaults(height_case_export_dir=None)
    parser.add_argument("--window", type=int, default=768)
    parser.add_argument("--stride", type=int, default=384)
    parser.set_defaults(diagnostics=False)
    parser.add_argument(
        "--weights",
        choices=["saved", "raw"],
        default="saved",
        help="Saved evaluation weights (default), or raw training weights for diagnosis.",
    )
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=False
    )
    parser.set_defaults(recalibrate_bn=False)
    parser.set_defaults(contour_spatial=False)
    parser.set_defaults(export_contour_dir=None)
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
    valid = valid & torch.isfinite(prediction) & torch.isfinite(target)
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


def load_dataset_metadata(base_dir):
    summary_path = os.path.join(base_dir, "summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"SpaceNet6 dataset summary not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    height_definition = str(metadata.get("height_definition", ""))
    if "pixel-level" not in height_definition.lower():
        raise ValueError(
            f"Evaluation requires the dense pixel-level SpaceNet6 height labels, got: {height_definition or 'height definition missing'}"
        )
    return metadata


def main():
    args = parse_args()
    if not 0.0 < args.contour_threshold < 1.0:
        raise ValueError("Contour threshold must be strictly between 0 and 1")
    if args.contour_threshold != 0.5:
        if not args.output:
            raise ValueError(
                "Nondefault contour threshold requires an explicit --output path"
            )
        if os.path.exists(args.output):
            raise FileExistsError(
                "Threshold evaluation output already exists; use a new output path"
            )
        if not args.sliding_window or any(
            (False, False, None, False, False, False, False, False, None, None)
        ):
            raise ValueError(
                "Nondefault contour threshold requires plain sliding evaluation with no diagnostics"
            )
    if args.sliding_window:
        probability_diagnostics = False
        if args.window <= 0 or args.window % 32 or (not 0 < args.stride <= args.window):
            raise ValueError(
                "Require positive window divisible by 32 and 0 < stride <= window"
            )
        from hdbf.spacenet6_validation_windows import (
            predict_validation_windows as predict_windows,
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and device.type == "cuda"
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint.get("args", {})
    model_config = checkpoint.get("model_config", {})
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
            "Conflicting geometry-height mode in model_config and training args"
        )
    if geometry_height_fusion and (not local_structure):
        raise ValueError(
            "Geometry-height fusion checkpoint requires Local UniRep structure"
        )
    use_unirep = model_config.get(
        "unirep_structure", bool(train_args.get("unirep_pretrained"))
    )
    if local_structure != bool(use_unirep):
        raise ValueError(
            "SpaceNet6 UniRep checkpoints require the matching local structure decoder"
        )
    data_config = checkpoint.get("data_config", {})
    normalize_rgb = bool(
        data_config.get("normalize_rgb", train_args.get("normalize_rgb", False))
    )
    base_dir = (
        args.base_dir
        or os.environ.get("SN6_HEIGHT_BASE")
        or train_args.get("base_dir")
        or "D:\\znx\\data\\spacenet6\\processed_height_dense"
    )
    base_dir = os.path.abspath(base_dir)
    source_dir = os.path.abspath(
        args.source_dir
        or os.environ.get("SN6_SOURCE_BASE", "")
        or train_args.get("source_dir", "")
        or os.path.dirname(base_dir)
    )
    dataset_metadata = load_dataset_metadata(base_dir)
    split_dir = args.split_dir or os.path.join(base_dir, "splits")
    if data_config.get("splits") and split_metadata(split_dir) != data_config["splits"]:
        raise ValueError(
            "Evaluation split manifests differ from the training checkpoint"
        )
    if "min_height" in data_config and args.min_height != data_config["min_height"]:
        raise ValueError("Evaluation min-height differs from the training checkpoint")
    if data_config.get("val_inference") == "sliding":
        args.sliding_window = True
        args.window = data_config["val_window"]
        args.stride = data_config["val_stride"]
        if args.window <= 0 or args.window % 32 or (not 0 < args.stride <= args.window):
            raise ValueError("Invalid sliding-window configuration in checkpoint")
        from hdbf.spacenet6_validation_windows import (
            predict_validation_windows as predict_windows,
        )
    sar_clip = (
        args.sar_clip
        if args.sar_clip is not None
        else float(data_config.get("sar_clip", train_args.get("sar_clip", 60.0)))
    )
    sar_normalization = args.sar_normalization or data_config.get(
        "sar_normalization", train_args.get("sar_normalization", "log")
    )
    variant = model_config.get("variant", train_args.get("variant", "B4"))
    decoder_dim = model_config.get("decoder_dim", train_args.get("decoder_dim", 512))
    use_hrbr = model_config.get("use_hrbr", train_args.get("hrbr", False))
    use_hcr = model_config.get("use_hcr", train_args.get("hcr", False))
    use_slmhc = model_config.get("use_slmhc", train_args.get("slmhc", False))
    use_bihm = model_config.get("use_bihm", train_args.get("bihm", False))
    use_ttci = model_config.get("use_ttci", train_args.get("ttci", False))
    use_multiscale_height_head = model_config.get(
        "use_multiscale_height_head",
        train_args.get("use_multiscale_height_head", False),
    )
    height_head_kind = model_config.get(
        "height_head_kind", train_args.get("height_head_kind", "lightweight")
    )
    ttci_bypass = model_config.get("ttci_bypass", train_args.get("ttci_bypass", False))
    task_channels = model_config.get(
        "task_channels", train_args.get("task_channels", 256)
    )
    rgb_anchored_fusion = model_config.get(
        "rgb_anchored_fusion", train_args.get("rgb_anchored_fusion", False)
    )
    rgb_anchored_hrbr = model_config.get(
        "rgb_anchored_hrbr", train_args.get("rgb_anchored_hrbr", False)
    )
    structure_only_sar = model_config.get(
        "structure_only_sar", train_args.get("structure_only_sar", False)
    )
    directional_ttci = model_config.get(
        "directional_ttci", train_args.get("directional_ttci", False)
    )
    use_roof_context_refiner = model_config.get(
        "use_roof_context_refiner", train_args.get("roof_context_refiner", False)
    )
    height_guidance_stop_gradient = model_config.get(
        "height_guidance_stop_gradient",
        train_args.get("height_guidance_stop_gradient", False),
    )
    use_rgb_height_detail_refiner = model_config.get(
        "use_rgb_height_detail_refiner",
        train_args.get("rgb_height_detail_refiner", False),
    )
    rgb_height_detail_channels = model_config.get(
        "rgb_height_detail_channels", train_args.get("rgb_height_detail_channels", 48)
    )
    rgb_height_detail_initial_strength = model_config.get(
        "rgb_height_detail_initial_strength",
        train_args.get("rgb_height_detail_initial_strength", 0.02),
    )
    rgb_height_detail_max_residual = model_config.get(
        "rgb_height_detail_max_residual",
        train_args.get("rgb_height_detail_max_residual", 4.0),
    )
    use_contour_structure_gate = model_config.get(
        "use_contour_structure_gate", train_args.get("contour_structure_gate", False)
    )
    contour_gate_kernel = model_config.get(
        "contour_gate_kernel", train_args.get("contour_gate_kernel", 9)
    )
    contour_gate_floor = model_config.get(
        "contour_gate_floor", train_args.get("contour_gate_floor", 0.15)
    )
    contour_gate_initial_strength = model_config.get(
        "contour_gate_initial_strength",
        train_args.get("contour_gate_initial_strength", 0.2),
    )
    structure_fusion_initial_strength = model_config.get(
        "structure_fusion_initial_strength",
        train_args.get("structure_fusion_initial_strength", 0.1),
    )
    auxiliary_initial_strength = model_config.get(
        "auxiliary_initial_strength", train_args.get("auxiliary_initial_strength", 0.02)
    )
    hcr_residual_scales = model_config.get(
        "hcr_residual_scales", train_args.get("hcr_residual_scales", (6.0, 3.0, 6.0))
    )
    slmhc_knots = model_config.get(
        "slmhc_knots",
        train_args.get("slmhc_knots", (0.0, 3.0, 6.0, 10.0, 20.0, 40.0, 100.0)),
    )
    state_key = (
        "model_state_dict" if args.weights == "saved" else "raw_model_state_dict"
    )
    if state_key not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain {state_key}; raw weights cannot be inferred."
        )
    model_state = checkpoint[state_key]
    ttci_legacy = use_ttci and (
        not any(
            (key.startswith("task_interaction.mask_correction") for key in model_state)
        )
    )
    lr_schedule = train_args.get("lr_schedule")
    if lr_schedule is None:
        lr_schedule = "poly" if train_args.get("poly_power", 0.9) < 0.9 else None
    use_ema = checkpoint.get(
        "evaluation_weights", "EMA" if train_args.get("ema_decay", 0.0) > 0.0 else "raw"
    ).startswith("EMA")
    use_height_multiscale_adapter = model_config.get(
        "use_height_multiscale_adapter",
        train_args.get("use_height_multiscale_adapter", False),
    )
    model_class = CMNeXtMTL
    model_options = {
        "building_affinity": model_config.get(
            "building_affinity", train_args.get("building_affinity", False)
        )
    }
    architecture = model_config.get("architecture") or ""
    encoder = model_config.get("encoder", train_args.get("encoder", "mit"))
    if encoder == "dinov2" or architecture.startswith("hdbf_dinov2_"):
        from semseg.models.hdbf_net import (
            DinoHDBF,
            checkpoint_fusion_type,
            checkpoint_task_split,
        )

        if use_height_multiscale_adapter:
            raise ValueError("DINOv2 checkpoint cannot use the legacy height adapter")
        model_class = DinoHDBF
        model_options.update(
            gradient_checkpointing=False,
            structure_isolation=model_config.get(
                "structure_isolation", train_args.get("structure_isolation", False)
            ),
            fusion_type=checkpoint_fusion_type(model_config, train_args),
            task_split_decoder=checkpoint_task_split(model_config, train_args),
        )
    if model_config.get("sar_channels", 4) != 4:
        raise ValueError("SpaceNet6 evaluation requires a four-band SAR checkpoint")
    if use_height_multiscale_adapter:
        raise ValueError("This historical experiment is not part of the final release")
    model = model_class(
        **model_options,
        context_detail_height=model_config.get(
            "context_detail_height", train_args.get("context_detail_height", False)
        ),
        use_multiscale_height_head=use_multiscale_height_head,
        multiscale_height_head_kind=height_head_kind,
        variant=variant,
        decoder_dim=decoder_dim,
        use_hrbr=use_hrbr,
        use_hcr=use_hcr,
        use_slmhc=use_slmhc,
        use_ttci=use_ttci,
        task_channels=task_channels,
        ttci_legacy=ttci_legacy,
        sar_channels=4,
        rgb_anchored_fusion=rgb_anchored_fusion,
        auxiliary_initial_strength=auxiliary_initial_strength,
        rgb_anchored_hrbr=rgb_anchored_hrbr,
        hcr_residual_scales=hcr_residual_scales,
        slmhc_knots=slmhc_knots,
        structure_only_sar=structure_only_sar,
        structure_fusion_initial_strength=structure_fusion_initial_strength,
        directional_ttci=directional_ttci,
        use_roof_context_refiner=use_roof_context_refiner,
        height_guidance_stop_gradient=height_guidance_stop_gradient,
        use_rgb_height_detail_refiner=use_rgb_height_detail_refiner,
        rgb_height_detail_channels=rgb_height_detail_channels,
        rgb_height_detail_initial_strength=rgb_height_detail_initial_strength,
        rgb_height_detail_max_residual=rgb_height_detail_max_residual,
        use_contour_structure_gate=use_contour_structure_gate,
        contour_gate_kernel=contour_gate_kernel,
        contour_gate_floor=contour_gate_floor,
        contour_gate_initial_strength=contour_gate_initial_strength,
    )
    if local_structure:
        if (
            not (encoder == "dinov2" or architecture.startswith("hdbf_dinov2_"))
            or use_hrbr
            or use_height_multiscale_adapter
            or (not model.task_split_decoder)
            or model.structure_isolation
            or directional_ttci
            or height_guidance_stop_gradient
        ):
            raise ValueError(
                "Local structure requires DINO TaskFusion/TaskSplit without old HRBR/alternate heads"
            )
        from hdbf.unirep_structure import UniRepStructure

        model.unirep_structure = UniRepStructure(
            model.backbone.channels, decoder_dim, local_structure=True
        )
    if geometry_height_fusion:
        from hdbf.geometry_height import GeometryHeightFusion

        model.height_geometry_fusion = GeometryHeightFusion(
            decoder_dim, mode=geometry_height_mode
        )
    model.load_state_dict(model_state)
    if ttci_bypass:
        from hdbf.spacenet6_reproducibility import activate_ttci_bypass

        activate_ttci_bypass(model)
    model.to(device).eval()
    bn_calibration = None
    dataset = SpaceNet6HeightDataset(
        base_dir,
        split=args.split,
        train=False,
        split_dir=split_dir,
        normalize_rgb=normalize_rgb,
        sar_clip=sar_clip,
        sar_normalization=sar_normalization,
        source_dir=source_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    full = empty_accumulator()
    height_positive = empty_accumulator()
    building = empty_accumulator()
    building_reliable = empty_accumulator()
    tp = fp = fn = 0.0
    contour_tp = contour_fp = contour_fn = 0.0
    diagnostics = {}
    spatial = exporter = None
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{args.split}: metrics", ascii=True):
            rgb = batch["rgb"].to(device, non_blocking=True)
            sar = batch["sar"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True) > 0.5
            contour = batch["contour"].to(device, non_blocking=True) > 0.5
            target_height = batch["height"].to(device, non_blocking=True).float()
            label_valid_height = (
                batch["valid_height"].to(device, non_blocking=True) > 0.5
            )
            if args.sliding_window:
                output = predict_windows(
                    model, rgb, sar, args.window, args.stride, amp_enabled
                )
                pred_height = output["height"]
                pred_mask = output["mask"] > 0.5
                pred_contour = output["contour"] > args.contour_threshold
            else:
                with autocast(enabled=amp_enabled):
                    output = model(rgb, sar)
                pred_height = output["height"].float()
                pred_mask = torch.sigmoid(output["mask"]) > 0.5
                pred_contour = torch.sigmoid(output["contour"]) > args.contour_threshold
            for name, diagnostic in diagnostics.items():
                diagnostic.update(output[name], mask if name == "mask" else contour)
            if spatial is not None:
                spatial.update(output["contour"], contour, mask)
            if exporter is not None:
                exporter.update(batch, output)
            height_mask = label_valid_height & (target_height > args.min_height)
            update_height(
                full, pred_height, target_height, torch.ones_like(mask), args.min_height
            )
            update_height(
                height_positive,
                pred_height,
                target_height,
                height_mask,
                args.min_height,
            )
            update_height(building, pred_height, target_height, mask, args.min_height)
            update_height(
                building_reliable,
                pred_height,
                target_height,
                mask & height_mask,
                args.min_height,
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
    contour_precision = contour_tp / max(contour_tp + contour_fp, 1.0)
    contour_recall = contour_tp / max(contour_tp + contour_fn, 1.0)
    contour_f1 = (
        2
        * contour_precision
        * contour_recall
        / max(contour_precision + contour_recall, 1e-08)
    )
    reliable_metrics = finalize(building_reliable)
    inferred_model_name = f"CMNeXt-MTL-{variant}{('-MultiScaleHeight-' + height_head_kind if use_multiscale_height_head else '')}{('-HeightAdapter' if use_height_multiscale_adapter else '')}{('-HRBR' if use_hrbr else '')}{('-HCR' if use_hcr else '')}{('-SLMHC' if use_slmhc else '')}{('-BIHM' if use_bihm else '')}{('-TTCI-Bypass' if ttci_bypass else '-TTCI' if use_ttci else '')}{('-RDSF' if structure_only_sar else '')}{('-RCR' if use_roof_context_refiner else '')}{('-RGBDetail' if use_rgb_height_detail_refiner else '')}{('-SCContour' if use_contour_structure_gate else '')}{('-Cosine' if lr_schedule == 'cosine' else '')}{('-SlowPoly' if lr_schedule == 'poly' else '')}{('-EMA' if use_ema else '')}"
    model_name = checkpoint.get("model_name", inferred_model_name)
    method_name = args.method_name or model_name
    lines = [f"Checkpoint: {args.checkpoint}"]
    weight_kind = (
        "raw"
        if args.weights == "raw"
        else checkpoint.get(
            "evaluation_weights",
            "EMA" if train_args.get("ema_decay", 0.0) > 0 else "saved",
        )
    )
    lines.append(
        f"Split: {args.split} | Weights: {weight_kind} {state_key} | AMP: {amp_enabled}"
    )
    if use_multiscale_height_head:
        lines.append(f"Multiscale height head: {height_head_kind}; active")
    if use_height_multiscale_adapter:
        lines.append("Height multiscale residual adapter: active")
    if ttci_bypass:
        lines.append(
            "TTCI bypass: active; serialized adapters/corrections are not used"
        )
    lines.append(
        f"Inference: sliding window={args.window}, stride={args.stride}; uniform height/probability averaging; mask threshold=0.5; contour threshold={args.contour_threshold:g}; BN fixed"
        if args.sliding_window
        else "Inference: whole image"
    )
    if bn_calibration is not None:
        lines.append(
            f"BN recalibration: {bn_calibration[0]} modules, {bn_calibration[1]} train forward batches/windows; window={(args.window if args.sliding_window else 'whole')}, stride={(args.stride if args.sliding_window else 'N/A')}, batch size 1, {('AMP' if amp_enabled else 'FP32')}, cumulative statistics; parameters unchanged."
        )
    elif checkpoint.get("bn_calibration"):
        lines.append(
            "Saved train-only BN calibration: "
            + json.dumps(checkpoint["bn_calibration"])
        )
    if "epoch" in checkpoint:
        lines.append(f"Loaded checkpoint epoch: {checkpoint['epoch']}")
    elif "selected_epoch" in checkpoint:
        lines.append(f"Loaded checkpoint epoch: {checkpoint['selected_epoch']}")
    if local_structure:
        lines.append("Structure: UniRep 192-channel dense local decoder; four-band SAR")
        lines.append(
            f"CNN geometry -> initial height: {bool(geometry_height_fusion)}; selected by checkpoint metadata"
        )
        if geometry_height_fusion:
            lines.append(
                f"Geometry-height mode: {geometry_height_mode}; selected by checkpoint metadata"
            )
        lines.append(
            f"Predicted-height guidance={model.unirep_structure.local_decoder.guidance_strength().item():.3f}; original height head/corrections; no GT input"
        )
    lines.extend(
        [
            f"Model: {model_name} | Input: RGB+SpaceNet6 SAR (4 bands)",
            (
                f"Contour refinement: soft footprint-neighborhood gate (kernel={contour_gate_kernel}, floor={contour_gate_floor:g})"
                if use_contour_structure_gate
                else "Contour refinement: standard"
            ),
            (
                f"Fusion: {model.backbone.fusion_type} | Encoder: DINOv2 ViT-B/14 | TaskSplit: {model.task_split_decoder}"
                if encoder == "dinov2" or architecture.startswith("hdbf_dinov2_")
                else (
                    "Fusion: role-decoupled structural SAR residual"
                    if structure_only_sar
                    else (
                        "Fusion: RGB-anchored reliability-gated SAR residual"
                        if rgb_anchored_fusion
                        else "Fusion: standard CMNeXt"
                    )
                )
            ),
            f"Height target: {dataset_metadata['height_definition']}",
            f"SAR normalization: {sar_normalization} | clip={sar_clip:g} dB",
            f"RGB normalization: {('ImageNet' if normalize_rgb else '0-1 (legacy)')}",
            "",
            "Markdown row candidate:",
            f"| {method_name} | RGB+SAR | {reliable_metrics['AbsRel']:.4f} | {reliable_metrics['RMSE']:.4f} | {reliable_metrics['d1']:.4f} | {reliable_metrics['d2']:.4f} | {reliable_metrics['d3']:.4f} | {reliable_metrics['MAE']:.4f} |",
            "",
            "[Mask]",
            f"F1={f1:.4f} Precision={precision:.4f} Recall={recall:.4f} IoU={iou:.4f}",
            "",
            "[Contour]",
            f"F1={contour_f1:.4f} Precision={contour_precision:.4f} Recall={contour_recall:.4f}",
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
    if spatial is not None:
        lines.extend(spatial.lines())
    if exporter is not None:
        exporter.save(
            None,
            {
                "checkpoint": args.checkpoint,
                "split": args.split,
                "weights": state_key,
                "bn_recalibrated": False,
                "selection": "four evenly spaced dataset indices and four largest false-positive counts at threshold 0.5; groups may overlap",
            },
        )
        lines.append(f"Contour examples exported: {None}")
    report = "\n".join(lines)
    print("\n" + report)
    default_name = (
        "test_metrics.txt"
        if args.weights == "saved"
        else f"metrics_{args.split}_raw.txt"
    )
    if args.sliding_window:
        default_name = f"metrics_{args.split}_{args.weights}_sliding{args.window}_s{args.stride}.txt"
    output_path = args.output or os.path.join(
        os.path.dirname(args.checkpoint), default_name
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
