"""DFC2023 recipe from the completed LocalConditioning reference run."""
import json
from pathlib import Path

CASES = ("wo_a", "wo_b", "wo_c", "rgb_only")
CHECKPOINT_ROOT = "checkpoints_dfc2023_ablation"
REFERENCE_RUN = "checkpoints_dfc2023/hdbf_dinov2_unireplknet_localconditioning100/261110_221208"
FULL_REFERENCE = f"{REFERENCE_RUN}/checkpoint_best_mae.pth"
BASE = """
--pretrained checkpoints/pretrained/dinov2/dinov2_vitb14_pretrain.pth
--sar-pretrained checkpoints/pretrained/segformer/segformer_b4_ade20k.bin
--unirep-lr-scale 0.1 --structure-branch-weight 0.2 --fusion-type task-aware
--local-structure --local-structure-weight 0.2 --no-query-structure --no-geometry-interaction
--geometry-height-fusion --geometry-height-mode asymmetric-local
--no-segmentation-refinement --task-split-decoder --no-hrbr
--epochs 100 --batch-size 8 --num-workers 4 --lr 3e-5
--rgb-backbone-lr-scale 0.1 --weight-decay 0.01 --warmup-epochs 5
--lr-schedule cosine --min-lr-ratio 0.05 --ema-decay 0.999 --ema-bn-policy copy
--context-detail-height --building-affinity --affinity-loss-weight 0.05
--ttci --hcr --bihm --slmhc --slmhc-tail-weight 0.05 --slmhc-warmup-epochs 50
--height-head-aux-weight 0.2 --bihm-weight 0.05 --bihm-warmup-epochs 50
--height-weight 1.2 --final-height-loss plain --mask-weight 1 --contour-weight 0.4
--bg-height-weight 0.02 --hcr-range-weight 0.05 --rgb-normalization imagenet
--rgb-jitter-probability 0 --min-height 0.1 --sar-clip 0.5
--seed 42 --data-seed 42 --gradient-checkpointing --checkpoint-mode inference
--save-best-mae-only --log-interval 25 --amp
""".split()


def train_arguments(case, data, split_dir, cnn, reference):
    if case not in CASES:
        raise ValueError(case)
    return BASE + [
        "--ablation-case", case, "--base-dir", str(data), "--split-dir", str(split_dir),
        "--reference-config", str(reference), "--unirep-pretrained", str(cnn),
        "--checkpoint-dir", f"{CHECKPOINT_ROOT}/{case}", "--model-name", f"HDBF-Net-DFC23-Ablation-{case}",
    ]


def verify_reference(args):
    """Check the full recipe and all three manifests before applying removals."""
    path = Path(args.reference_config)
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed full run's effective_config.json: {path}. "
                                "Use --reference-config to point to the original file if it was moved.")
    reference = json.loads(path.read_text(encoding="utf-8"))["args"]
    excluded = {
        "ablation_case", "reference_config", "base_dir", "split_dir", "checkpoint_dir", "model_name",
        "device", "log_interval", "pretrained", "sar_pretrained", "unirep_pretrained",
    }
    mismatches = [
        f"{name}: current={value!r}, reference={reference.get(name)!r}"
        for name, value in vars(args).items()
        if name not in excluded and (name not in reference or reference[name] != value)
    ]
    if mismatches:
        raise ValueError("DFC2023 ablation differs from its full reference before module removal:\n"
                         + "\n".join(mismatches))
