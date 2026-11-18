# SpaceNet6 Module Ablations

Full reference: `261109_191439/checkpoint_best_mae.pth`, selected at epoch43
by minimum validation MAE. Do not retrain or replace this row based on test scores.

## Five Rows, Four New Runs

| Case | A | B | C | SAR | Replacement |
| --- | --- | --- | --- | --- | --- |
| Existing full | on | on | on | on | Existing full checkpoint |
| wo_a | off | on | on | on | Per-scale RGB/SAR concatenation + one shared Conv1x1; TTCI removed |
| wo_b | on | off | on | on | Existing coarse mask/contour heads + bilinear output resize |
| wo_c | on | on | off | on | Identity after HCR; no SLMHC or its tail auxiliary |
| rgb_only | RGB routing | on | on | off | DINO pyramid goes directly to TaskSplit; TTCI retained |

A = learned local correspondence/task-aware fusion plus TTCI.
B = UniRep CNN/ADE/local structure decoder plus asymmetric-local geometry-to-height bridge.
C = SLMHC calibration and its dedicated directional-tail auxiliary.
ContextDetail, building-affinity height aggregation, HCR and BIHM stay enabled.

Removing B also removes CNN/semantic/local structure auxiliaries. No replacement
HRBR, attention, query decoder or learned upsampling is added. Removing A adds
only a 1x1 projection of concatenated features: no normalization or activation.
Removing C adds no learnable parameter. RGB-only physically removes the SAR
encoder and all bimodal fusion modules; it does not pass zeros or duplicated RGB
through them. Because fusion is bimodal, RGB-only is a modality control, not a
claim that the cross-modal part of A remains unchanged.

## Fixed Protocol

The isolated trainer/evaluator retain the successful full recipe. This release
removes historical diagnostic branches without changing that recipe. Its
source lock belongs to the cleaned release; do not mix with an older tree.
The module subclass constructs the full parent before deleting A/C/SAR modules,
and new Conv1x1 initialization uses fork_rng. Retained initial tensors match.
Removal happens before optimizer/EMA creation; unused modules are not retained.

All new runs start from encoder pretraining for100 epochs, batch8, crop512,
one crop/image, positive probability0.7, seed42/data-seed42, same LR/EMA/BN-copy,
same valid-height pixels and same remaining loss functions/weights. No new data,
resampling, fine-tuning, post-training BN calibration or threshold search.
Train/val/test manifest hashes are checked against the full run.
Validation and test use768/384 sliding averaging and thresholds0.5. Every row
uses one lowest-validation-MAE checkpoint for all metrics; task-best checkpoints
are retained only for inspection, never mixed into the row.

## Upload And Run

Upload this entire `ablation_spacenet6` folder beside `train_spacenet6.py` in
cluster HDBF_FIX. Keep existing shared modules and pretrained weights available.
`sources.json` rejects changed shared Python sources before training. A mismatch
must be resolved by checking the named file, not by disabling the check.

Run each experiment manually. Each command trains ONE case, evaluates its
best-MAE checkpoint on val/test, then stops. No next experiment starts.

```bash
cd /public/home/znx_221124120159/projects/HDBF_FIX
bash ablation_spacenet6/run.sh wo_a
```

```bash
cd /public/home/znx_221124120159/projects/HDBF_FIX
bash ablation_spacenet6/run.sh wo_b
```

The remaining commands are:

```bash
bash ablation_spacenet6/run.sh wo_c
bash ablation_spacenet6/run.sh rgb_only
```

If both GPUs belong to ONE allocation/container, use two terminals and append
`--gpu-index 0` and `--gpu-index 1` respectively. These indices select within the
scheduler-visible GPU list, preserving physical IDs/UUIDs assigned by it.
On separate single-GPU allocations, omit this option or use index0 in each.
Each command refuses training unless exactly one CUDA GPU is visible.

An individual case can be run with `bash ablation_spacenet6/run.sh wo_a`.
Use `--dry-run` to print commands without loading weights or training.
`SN6_BASE` overrides only the dataset location; split hashes still must match.

## Results And Interruption

Everything is under a separate root:

```text
checkpoints_spacenet6_ablation/
  wo_a/console.log
  wo_a/<timestamp>/checkpoint_best_mae.pth
  wo_a/<timestamp>/training_metrics.jsonl
  wo_a/<timestamp>/effective_config.json
  wo_a/<timestamp>/val_best_mae.txt
  wo_a/<timestamp>/test_best_mae.txt
  wo_a/completed.json
  wo_b/...
  wo_c/...
  rgb_only/...
```

Each run automatically evaluates val then test after100 epochs. A failed run
stops that command; the other GPU's experiment is independent. Fully completed cases
are skipped. Completed training with failed evaluation can retry evaluation
without retraining. Partial training is not silently resumed or overwritten.
Per-case atomic locks prevent duplicate launches. Normal cancellation forwards
termination to the Linux subprocess group. SIGKILL/power loss can leave an empty
`.running.lock` directory; verify no job is running before removing that lock.
Use this folder's `eval.py`, not the root evaluator, for ablation checkpoints.

The evaluator prints and saves only nine columns, in this order:
MaskF1, MaskIoU, ContourF1, AbsRel, RMSE(m), delta1, delta2, delta3, MAE(m).
The six height metrics use Building Mask + source-valid finite GT Height >0.1m.
Mask/contour metrics retain the full segmentation evaluation region; restricting
them to positive GT-height building pixels would discard background errors.
No precision, recall, pixel counts or other height-region groups are reported.
Model construction/progress output is hidden on success and restored on errors.

## Verification Limits

`python ablation_spacenet6/tests.py --full-size` checks original/full parity,
retained initial tensors, effective flags/loss removals, physical module removal,
RGB independence, AMP backward and512x512 inference using64/32 decoder/task widths.
`python ablation_spacenet6/test_eval.py` exercises real checkpoint/evaluator
construction with synthetic IO, including missing-metadata rejection.
These are software checks, not accuracy measurements or production batch8 VRAM tests.
