"""Frozen training recipe matching full run 261109_191439."""
from pathlib import Path

CASES = ("wo_a", "wo_b", "wo_c", "rgb_only")
CHECKPOINT_ROOT = "checkpoints_spacenet6_ablation"
FULL_REFERENCE = "checkpoints_spacenet6/hdbf_dinov2_unireplknet_localconditioning100/261109_191439/checkpoint_best_mae.pth"
REFERENCE_SPLITS = {
    "train": {"count": 1645, "sha256": "80d4e95b2c4f681018b690f01cab33e7ef8e347354f36e766436b10180b7d021"},
    "val": {"count": 349, "sha256": "915a8b66ab03b9550e48aa62d90b957bbbaa1542ceffb4c994110fec71c8c913"},
    "test": {"count": 330, "sha256": "8ef6c506a79a26a9793478507c6bff7b569ce28c9a10a2328b1cd338d3e6c34c"},
}
FIXED_RECIPE = dict(encoder="dinov2", fusion_type="task-aware", task_split_decoder=True,
    context_detail_height=True, building_affinity=True, hcr=True, bihm=True, hrbr=False,
    normalize_rgb=True, contour_objective="bce-dice", contour_structure_gate=False,
    structure_isolation=False, directional_ttci=False, height_guidance_stop_gradient=False,
    rgb_anchored_fusion=False, structure_only_sar=False, roof_context_refiner=False,
    rgb_height_detail_refiner=False, epochs=100, batch_size=8, crop_size=512, crops_per_image=1,
    positive_crop_probability=.7, targeted_crop_probability=0., building_mix_probability=0.,
    standard_scale_augmentation=False, lr=3e-5, rgb_backbone_lr_scale=.1, unirep_lr_scale=.1,
    weight_decay=.01, warmup_epochs=5, lr_schedule="cosine", min_lr_ratio=.05,
    ema_decay=.999, ema_bn_policy="copy", height_weight=1.2, height_head_weight=.2,
    mask_weight=1., contour_weight=.4, hcr_range_weight=.05, bihm_weight=.05,
    bihm_warmup_epochs=50, slmhc_warmup_epochs=50, bg_height_weight=.02,
    height_correction_guard_weight=0., roof_shape_weight=0., contour_consistency_weight=0.,
    val_inference="sliding", val_window=768, val_stride=384, sar_normalization="linear",
    sar_clip=60., min_height=.1, seed=42, data_seed=42, decoder_dim=512, task_channels=256,
    experiment="hdbf", gradient_checkpointing=True, checkpoint_mode="inference", num_workers=4,
    height_range_boundaries=(3., 20.), hcr_residual_scales=(6., 3., 6.),
    slmhc_knots=(0., 3., 6., 10., 20., 40., 100.), tail_low_threshold=3., tail_high_threshold=20.)
FIXED_RECIPE.update(standard_tune100=False, height_rescue=False, resume=None,
    finetune_checkpoint=None, stop_after_epoch=None, sar_dropout_probability=0.,
    sar_hold_epochs=0, sar_ramp_epochs=0, ttci_bypass=False, val_batch_size=1,
    pin_memory=False, amp=True, affinity_loss_weight=.05, bihm_min_pixels=16, bihm_min_valid_ratio=.5)
BASE = """
--encoder dinov2 --fusion-type task-aware --task-split-decoder
--pretrained checkpoints/pretrained/dinov2/dinov2_vitb14_pretrain.pth
--sar-pretrained checkpoints/pretrained/segformer/segformer_b4_ade20k.bin
--unirep-lr-scale 0.1 --structure-branch-weight 0.2
--local-structure --local-structure-weight 0.2 --geometry-height-fusion
--geometry-height-mode asymmetric-local
--epochs 100 --batch-size 8 --num-workers 4 --crop-size 512 --crops-per-image 1
--positive-crop-probability 0.7 --targeted-crop-probability 0
--val-batch-size 1 --val-inference sliding --val-window 768 --val-stride 384
--lr 3e-5 --rgb-backbone-lr-scale 0.1 --weight-decay 0.01
--warmup-epochs 5 --lr-schedule cosine --min-lr-ratio 0.05 --ema-decay 0.999 --ema-bn-policy copy
--context-detail-height --building-affinity --affinity-loss-weight 0.05
--ttci --hcr --no-hrbr --bihm --slmhc --no-contour-structure-gate
--height-head-weight 0.2 --height-weight 1.2 --mask-weight 1 --contour-weight 0.4
--slmhc-tail-weight 0.05 --slmhc-warmup-epochs 50
--bihm-weight 0.05 --bihm-warmup-epochs 50 --hcr-range-weight 0.05 --bg-height-weight 0.02
--normalize-rgb --sar-normalization linear --sar-clip 60 --min-height 0.1
--seed 42 --data-seed 42 --gradient-checkpointing --amp
--contour-objective bce-dice --contour-consistency-weight 0
--checkpoint-mode inference --save-best-mae-only --save-task-best
""".split()


def train_arguments(case, data, cnn):
    if case not in CASES:
        raise ValueError(case)
    return BASE + ["--ablation-case", case, "--base-dir", str(data), "--split-dir", str(Path(data) / "splits"),
        "--unirep-pretrained", str(cnn), "--checkpoint-dir", f"{CHECKPOINT_ROOT}/{case}",
        "--model-name", "HDBF-Net-SN6-Ablation-" + case]
