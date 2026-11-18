# DFC2023 Ablations

Full reference: DINOv2-UniRepLKNet-LocalConditioning, run
`261110_221208`, using its lowest-validation-MAE checkpoint. The full model is
not retrained. This folder uses DFC2023 native tiles and single-band SAR, not
the SpaceNet6 training or sliding-window protocol.

| Case | Removed | Replacement |
| --- | --- | --- |
| wo_a | Task-aware RGB/SAR fusion and TTCI | One shared 1x1 projection after concatenation at each scale |
| wo_b | UniRep CNN, local structure decoder, geometry-height bridge and their auxiliary losses | Original prediction heads and bilinear upsampling |
| wo_c | SLMHC and its directional tail auxiliary | Use the HCR output directly |
| rgb_only | SAR encoder and bimodal fusion | Direct RGB feature pyramid; TTCI retained |

ContextDetail, building-affinity aggregation, HCR and BIHM stay enabled. The
training loop, remaining losses, initialization of retained weights, 100-epoch
budget, seeds, EMA/BN-copy policy and data splits match the full recipe. The
RGB-only row is a modality control, not a claim that its cross-modal fusion is
unchanged.

## Run

Upload this folder to cluster `HDBF_FIX/ablation_dfc2023/`. Keep the existing
shared code and pretrained weights. Select **one** command for each job:

```bash
cd /public/home/znx_221124120159/projects/HDBF_FIX
bash ablation_dfc2023/run.sh wo_a
bash ablation_dfc2023/run.sh wo_b
bash ablation_dfc2023/run.sh wo_c
bash ablation_dfc2023/run.sh rgb_only
```

Each line is a separate experiment, not a queue. Separate single-GPU jobs need
no GPU flag; in a multi-GPU allocation append `--gpu-index 0` or
`--gpu-index 1` using indices in the scheduler-visible list.

The launcher checks the active shared sources and the full run's
`effective_config.json`. The original configuration is read from the
`261110_221208` directory, including train/val/test manifest hashes. If that
directory was moved, use `--reference-config PATH/effective_config.json`. The
data defaults to the existing DFC2023 root and `dfc23_newsplit`.

## Results

Each run stores one inference-only `checkpoint_best_mae.pth`; it does not
write optimizer state or periodic resume checkpoints. After 100 epochs it
evaluates that same checkpoint on val and test, then stops.

```text
checkpoints_dfc2023_ablation/<case>/console.log
checkpoints_dfc2023_ablation/<case>/<timestamp>/val_best_mae.txt
checkpoints_dfc2023_ablation/<case>/<timestamp>/test_best_mae.txt
```

Reports contain only MaskF1, MaskIoU, ContourF1, AbsRel, RMSE(m), delta1,
delta2, delta3 and MAE(m). Height uses GT Building Mask + valid, finite GT
Height >0.1m; segmentation metrics retain full-tile false positives and
false negatives. A partially trained run is not automatically replaced
or evaluated as complete.

Local check: `python ablation_dfc2023/tests.py`. It uses small synthetic tiles
and reduced decoder widths to check model removal, initialization, backward
passes, state loading and metric scope. It is not a batch-8 memory or
dataset-accuracy test.
