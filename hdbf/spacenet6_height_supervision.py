"""Optional direct head supervision and label-based correction safeguards."""

from pathlib import Path as _HDBFPath
import sys as _hdbf_sys

_HDBF_ROOT = _HDBFPath(__file__).resolve().parents[1]
if str(_HDBF_ROOT) not in _hdbf_sys.path:
    _hdbf_sys.path.insert(0, str(_HDBF_ROOT))


import torch
import torch.nn.functional as F


def head_supervision(prediction, target, valid):
    if not valid.any():
        return prediction.float().sum() * 0.0
    return F.smooth_l1_loss(prediction[valid].float(), target[valid].float(), beta=1.0)


def correction_safeguard(before, after, target, valid):
    if not valid.any():
        return after.float().sum() * 0.0
    target = target[valid].float()
    # Detach the reference error so it cannot be deliberately increased to relax the bound.
    reference = (before[valid].detach().float() - target).abs()
    corrected = (after[valid].float() - target).abs()
    return (corrected - reference).clamp_min(0).mean()
