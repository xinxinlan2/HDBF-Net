#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
DATA="${SN6_BASE:-/public/home/znx_221124120159/dataset/spacenet6/processed_height_dense}"
CNN="${1:-$ROOT/checkpoints/pretrained/unireplknet/upernet_unireplknet_s_in22k_512_160k_ade20k_miou51.86.pth}"
test -s "$CNN" || { echo "Missing ADE20K checkpoint: $CNN" >&2; exit 1; }
echo "67efb2494f12e2bccaa19282b8d2eb0d466fe5755584e5fb9903cd188ebc0e64  $CNN" | sha256sum -c -
mkdir -p logs
echo "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73  checkpoints/pretrained/dinov2/dinov2_vitb14_pretrain.pth" | sha256sum -c -
python -u tests/test_geometry_height.py
python -u tests/test_spacenet6_local_structure.py --checkpoint "$CNN"
python -u train_spacenet6.py \
  --encoder dinov2 --fusion-type task-aware --task-split-decoder \
  --base-dir "$DATA" --split-dir "$DATA/splits" \
  --pretrained checkpoints/pretrained/dinov2/dinov2_vitb14_pretrain.pth \
  --sar-pretrained checkpoints/pretrained/segformer/segformer_b4_ade20k.bin \
  --unirep-pretrained "$CNN" --unirep-lr-scale 0.1 --structure-branch-weight 0.2 \
  --local-structure --local-structure-weight 0.2 --geometry-height-fusion \
  --geometry-height-mode asymmetric-local \
  --checkpoint-dir checkpoints_spacenet6/hdbf_dinov2_unireplknet_localconditioning100 \
  --model-name HDBF-Net-DINOv2-UniRepLKNet-LocalConditioning \
  --epochs 100 --batch-size 8 --num-workers 4 --crop-size 512 --crops-per-image 1 \
  --positive-crop-probability 0.7 --targeted-crop-probability 0 \
  --val-batch-size 1 --val-inference sliding --val-window 768 --val-stride 384 \
  --lr 3e-5 --rgb-backbone-lr-scale 0.1 --weight-decay 0.01 \
  --warmup-epochs 5 --lr-schedule cosine --min-lr-ratio 0.05 --ema-decay 0.999 --ema-bn-policy copy \
  --context-detail-height --building-affinity --affinity-loss-weight 0.05 \
  --ttci --hcr --no-hrbr --bihm --slmhc --no-contour-structure-gate \
  --height-head-weight 0.2 --height-weight 1.2 --mask-weight 1 --contour-weight 0.4 \
  --slmhc-tail-weight 0.05 --slmhc-warmup-epochs 50 \
  --bihm-weight 0.05 --bihm-warmup-epochs 50 --hcr-range-weight 0.05 --bg-height-weight 0.02 \
  --normalize-rgb --sar-normalization linear --sar-clip 60 --min-height 0.1 \
  --seed 42 --data-seed 42 --gradient-checkpointing --amp \
  --contour-objective bce-dice --contour-consistency-weight 0 \
  --checkpoint-mode inference --save-best-mae-only --save-task-best \
  2>&1 | tee "logs/train_hdbf_dinov2_unireplknet_localconditioning_sn6_$(date +%Y%m%d_%H%M%S).log"
