# HDBF-Net

Joint building footprint, contour, and height estimation from optical--SAR imagery.

## Getting started

- **Evaluate the released models:** install the environment, download the final
  checkpoints, prepare the matching data/split manifests, and follow Evaluation.
  Separate encoder initialization weights are not needed.
- **Train or run ablations:** additionally download all three initialization
  checkpoints listed below. Run commands from the repository root in Linux/Bash.
- Model weights and datasets are not included in this repository.

## Implementation

This package contains the final LocalConditioning implementation used for
DFC2023 and SpaceNet6+AHN. Historical audit tools, retired experiment
entry points and unused architecture modules have been removed. This is
not the earlier Inria, query, or structure-isolated experiment. Checkpoint
parameter names and the final model's initialization order are preserved.

## Environment

The cluster used Python 3.10, PyTorch 2.0.1/CUDA 11.7 and MMCV 1.7.2.
Install the matching PyTorch/torchvision pair first, then:

```bash
pip install -r requirements_dfc2023.txt
```

Retain bootstrap.py and vendor/. The vendor code and LICENSE files must
accompany redistribution. Initialization weights retain their upstream
terms; inclusion here is not a grant of redistribution rights.

## Final trained checkpoints

Download [checkpoint.zip from Baidu Netdisk](https://pan.baidu.com/s/1p0t_UHA059_A4BB8q-f6Dw?pwd=vsgq).
Extraction code: `vsgq`.

Use checkpoint_best_mae.pth for BOTH datasets, not best_contour or best_f1.
Validate each extracted checkpoint against the reference SHA256 below
(`sha256sum /path/to/checkpoint_best_mae.pth` on Linux):

| Dataset | Run | Selected epoch | SHA256 |
|---|---|---:|---|
| DFC2023 | 261110_221208 | 54 | 0367203839c386f3f6c833a4d942a894a40f71c5f3156f04222660673feab3e6 |
| SpaceNet6+AHN | 261109_191439 | 43 | 45555711f96fa2cfd31159363abd99728a6ecc9d2dd7a2ad05fbefbc07e981c6 |

These inference checkpoints contain EMA model state and configuration, not
a complete optimizer/resume state. No post-training BN recalibration is
used for these final results. BN running statistics are copied to EMA
during training. Mask and contour thresholds are 0.5.

## Which pretrained weights are needed?

Evaluation with a final trained checkpoint does NOT require initialization
weights. Reproducing training from initialization requires all THREE:

1. checkpoints/pretrained/dinov2/dinov2_vitb14_pretrain.pth
2. checkpoints/pretrained/segformer/segformer_b4_ade20k.bin
3. checkpoints/pretrained/unireplknet/upernet_unireplknet_s_in22k_512_160k_ade20k_miou51.86.pth

SegFormer-B4 is still used: 16 patch-projection/normalization tensors
initialize the four SAR stages. It is not the RGB encoder. DINOv2 is the
RGB encoder; UniRepLKNet-S and its ADE20K decoder features initialize the
CNN geometry branch. Inria, SegFormer-B5 and the old pytorch_model.bin are
not required and are excluded. Initialization weights are distributed
separately from this GitHub package; place them at the three paths above
before training. Their hashes are in PRETRAINED_SHA256.json.

### Official sources and downloads

| Initialization | Source | Role in HDBF-Net |
|---|---|---|
| DINOv2 ViT-B/14, without registers | [Meta DINOv2](https://github.com/facebookresearch/dinov2#pretrained-models) | RGB encoder |
| SegFormer-B4, ADE20K | [NVIDIA checkpoint](https://huggingface.co/nvidia/segformer-b4-finetuned-ade-512-512/blob/main/pytorch_model.bin) | SAR patch-projection/normalization initialization |
| UniRepLKNet-S, ImageNet-22K then ADE20K | [Author checkpoint](https://huggingface.co/DingXiaoH/UniRepLKNet/blob/main/upernet_unireplknet_s_in22k_512_160k_ade20k_miou51.86.pth), [upstream project](https://github.com/AILab-CVC/UniRepLKNet) | CNN encoder and UPerNet decoder features |

Run from the repository root. These commands download initialization weights,
not the final trained HDBF-Net checkpoints:

```bash
mkdir -p checkpoints/pretrained/{dinov2,segformer,unireplknet}

curl -fL --retry 3 \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth \
  -o checkpoints/pretrained/dinov2/dinov2_vitb14_pretrain.pth

curl -fL --retry 3 \
  https://huggingface.co/nvidia/segformer-b4-finetuned-ade-512-512/resolve/main/pytorch_model.bin \
  -o checkpoints/pretrained/segformer/segformer_b4_ade20k.bin

curl -fL --retry 3 \
  https://huggingface.co/DingXiaoH/UniRepLKNet/resolve/main/upernet_unireplknet_s_in22k_512_160k_ade20k_miou51.86.pth \
  -o checkpoints/pretrained/unireplknet/upernet_unireplknet_s_in22k_512_160k_ade20k_miou51.86.pth
```

The SegFormer download is named `pytorch_model.bin` upstream; save it as
`segformer_b4_ade20k.bin` as above. Do not substitute a classification-only
UniRepLKNet checkpoint or a DINOv2 variant with registers.

Verify all three files before training. Paths in `PRETRAINED_SHA256.json`
are relative to `checkpoints/`:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

expected = json.loads(Path("PRETRAINED_SHA256.json").read_text())
for name, digest in expected.items():
    path = Path("checkpoints") / name
    actual = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            actual.update(chunk)
    if actual.hexdigest() != digest:
        raise SystemExit(f"SHA256 mismatch: {path}; check the download/version.")
    print(f"OK: {path}")
PY
```

If a download fails, use its source page to download the same file manually.
Keep the expected hash unchanged; a mismatch must be resolved before training.

## Training

### Data prerequisites

The commands expect prepared labels and the original split manifests, not
just raw imagery. DFC2023 uses `dfc23_newsplit/{train,val,test}.txt` under
the dataset directory; SpaceNet6+AHN uses `splits/{train,val,test}.txt`
under `processed_height_dense`. The latter includes the prepared AHN-derived
height supervision. Downloading SpaceNet6 imagery alone is not sufficient.

This cleaned package does not include raw-data preparation tools or the
prepared datasets. The metadata records the experimental configuration;
it does not replace the required label files and split manifests.

### Launch training

Use the original prepared data and split manifests. Set local data paths
explicitly; the scripts' fallback paths refer to the original cluster.
Do not regenerate partitions or alter height masks to match rounded metrics.

```bash
export DFC23_BASE=/absolute/path/to/dfc2023
bash run_train_dfc2023.sh

export SN6_BASE=/absolute/path/to/spacenet6/processed_height_dense
bash run_train_spacenet6_taskfusion.sh
```

The launchers include preflight tests and full 100-epoch training from
pretrained encoders. Final configuration: task-aware fusion, task-split
decoders, UniRep dense local geometry, asymmetric-local geometry-to-height
conditioning, ContextDetail with affinity, TTCI, HCR, SLMHC and BIHM.
Dataset-specific SAR channels/normalization and supervision are preserved.
No task-specific learning-rate scheduling experiment or targeted height
sampling is enabled. See the launchers for all exact arguments.

## Evaluation

Run from this directory. Set CKPT to the corresponding extracted final
checkpoint; data paths must refer to the original prepared labels/splits.

```bash
CKPT=/absolute/path/to/DFC2023/checkpoint_best_mae.pth
DATA=/absolute/path/to/dfc2023
for SPLIT in val test; do
  python -u eval_dfc2023.py --checkpoint "$CKPT" --base-dir "$DATA" \
    --split-dir "$DATA/dfc23_newsplit" --split "$SPLIT" \
    --batch-size 4 --num-workers 4 --amp --output "dfc23_${SPLIT}.txt"
done

CKPT=/absolute/path/to/SpaceNet6/checkpoint_best_mae.pth
DATA=/absolute/path/to/spacenet6/processed_height_dense
for SPLIT in val test; do
  python -u eval_spacenet6.py --checkpoint "$CKPT" --base-dir "$DATA" \
    --split-dir "$DATA/splits" --split "$SPLIT" --batch-size 1 \
    --num-workers 4 --amp --sliding-window --window 768 --stride 384 \
    --output "sn6_${SPLIT}.txt"
done
```

Do not add BN recalibration or tune thresholds using test results.
Height columns below use reliable building pixels with GT height > 0.1 m;
mask/contour metrics use their own segmentation evaluation domains.

| Dataset | MaskF1 | MaskIoU | ContourF1 | AbsRel | RMSE(m) | d1 | d2 | d3 | MAE(m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DFC2023 | 0.8982 | 0.8152 | 0.6902 | 0.5846 | 6.5160 | 0.6032 | 0.8039 | 0.8995 | 3.7890 |
| SpaceNet6+AHN | 0.8371 | 0.7199 | 0.4964 | 0.2569 | 2.7021 | 0.5976 | 0.8546 | 0.9519 | 1.8368 |

## Ablations and provenance

ablation_dfc2023/ and ablation_spacenet6/ retain their recipes and separate
outputs. Their source locks identify this cleaned release, not the original
byte-for-byte experiment tree. The cleanup report records removed files
and original source-lock hashes. Read their README files before running.
A = custom fusion + TTCI; B = CNN structure + geometry-to-height path;
C = SLMHC. HCR, BIHM and ContextDetail remain fixed. rgb_only removes SAR.
Do not update source locks simply to bypass a mismatch error.

final_run_metadata/ contains the original config/metric reports. Trained
checkpoints, initialization weights and prepared datasets are distributed
separately. Historical tools/ and configs/ folders are not required.

## Release checks

Both final checkpoints strictly load. On a fixed synthetic 64x64 CUDA AMP
input per dataset, all returned tensors match the pre-cleanup code exactly;
the freshly initialized model state hashes also match. Both full training
recipes and eight ablation recipe parsers retain the same effective values.
The retained preflight tests cover losses, gradient isolation, pretrained
loading, optimizer groups, EMA and sliding-window inference. These checks
are not a rerun of full-dataset evaluation or 100-epoch training.

For GitHub, upload this directory's contents directly. It contains no model
weights. Keep the seven code/provenance folders and root files together;
do not combine these files with older ablation source manifests.
