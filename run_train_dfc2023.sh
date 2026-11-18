#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
CNN="${1:-$ROOT/checkpoints/pretrained/unireplknet/upernet_unireplknet_s_in22k_512_160k_ade20k_miou51.86.pth}"
test -s "$CNN" || { echo "Missing ADE20K checkpoint: $CNN" >&2; exit 1; }
echo "67efb2494f12e2bccaa19282b8d2eb0d466fe5755584e5fb9903cd188ebc0e64  $CNN" | sha256sum -c -
echo "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73  checkpoints/pretrained/dinov2/dinov2_vitb14_pretrain.pth" | sha256sum -c -
python -u tests/test_geometry_height.py
python -u tests/test_dfc_geometry_height.py
python -u tests/test_local_structure.py --checkpoint "$CNN" --geometry-height-fusion
mkdir -p logs
DATA="${DFC23_BASE:-/public/home/znx_221124120159/dataset/dfc2023}"
python -u train_dfc2023.py \
  --base-dir "$DATA" --split-dir "$DATA/dfc23_newsplit" \
  --pretrained checkpoints/pretrained/dinov2/dinov2_vitb14_pretrain.pth \
  --sar-pretrained checkpoints/pretrained/segformer/segformer_b4_ade20k.bin \
  --unirep-pretrained "$CNN" --unirep-lr-scale 0.1 --structure-branch-weight 0.2 \
  --checkpoint-dir checkpoints_dfc2023/hdbf_dinov2_unireplknet_localconditioning100 \
  --model-name HDBF-Net-DINOv2-UniRepLKNet-LocalConditioning --fusion-type task-aware \
  --local-structure --local-structure-weight 0.2 --no-query-structure --no-geometry-interaction \
  --geometry-height-fusion --geometry-height-mode asymmetric-local \
  --no-segmentation-refinement --task-split-decoder --no-hrbr \
  --epochs 100 --batch-size 8 --num-workers 4 --lr 3e-5 \
  --rgb-backbone-lr-scale 0.1 --weight-decay 0.01 --warmup-epochs 5 \
  --lr-schedule cosine --min-lr-ratio 0.05 --ema-decay 0.999 --ema-bn-policy copy \
  --context-detail-height --building-affinity --affinity-loss-weight 0.05 \
  --ttci --hcr --bihm --slmhc --slmhc-tail-weight 0.05 --slmhc-warmup-epochs 50 \
  --height-head-aux-weight 0.2 --bihm-weight 0.05 --bihm-warmup-epochs 50 \
  --height-weight 1.2 --final-height-loss plain --mask-weight 1 --contour-weight 0.4 \
  --bg-height-weight 0.02 --hcr-range-weight 0.05 --rgb-normalization imagenet \
  --rgb-jitter-probability 0 --min-height 0.1 --sar-clip 0.5 \
  --seed 42 --data-seed 42 --gradient-checkpointing --checkpoint-mode inference \
  --save-best-mae-only --log-interval 25 --amp \
  2>&1 | tee "logs/train_dfc2023_unireplknet_localconditioning_$(date +%Y%m%d_%H%M%S).log"
