"""Sliding validation: stitch predictions before scoring each source pixel once."""

from pathlib import Path as _HDBFPath
import sys as _hdbf_sys

_HDBF_ROOT = _HDBFPath(__file__).resolve().parents[1]
if str(_HDBF_ROOT) not in _hdbf_sys.path:
    _hdbf_sys.path.insert(0, str(_HDBF_ROOT))

import torch
from torch.cuda.amp import autocast


def window_starts(length, window, stride):
    if window <= 0 or not 0 < stride <= window or length < window:
        raise ValueError("Require image length >= window >= stride > 0")
    positions = list(range(0, length - window + 1, stride))
    if positions[-1] != length - window:
        positions.append(length - window)
    return positions


@torch.no_grad()
def predict_validation_windows(
    model, rgb, sar, window, stride, amp, height_stages=False
):
    if model.training:
        raise ValueError("Sliding validation requires model.eval(); BN must stay fixed")
    if rgb.shape[0] != sar.shape[0] or rgb.shape[-2:] != sar.shape[-2:]:
        raise ValueError("RGB and SAR batches must share spatial dimensions")
    h, w = rgb.shape[-2:]
    keys = ["mask", "contour", "height"]
    if height_stages:
        keys += ["height_head", "height_coarse", "height_after_hcr"]
    sums = {
        key: torch.zeros(
            (rgb.shape[0], 1, h, w), device=rgb.device, dtype=torch.float32
        )
        for key in keys
    }
    count = torch.zeros_like(sums["height"])
    for y in window_starts(h, window, stride):
        for x in window_starts(w, window, stride):
            region = (..., slice(y, y + window), slice(x, x + window))
            with autocast(enabled=amp):
                output = (
                    model(rgb[region], sar[region], return_intermediates=True)
                    if height_stages
                    else model(rgb[region], sar[region])
                )
            for key in keys:
                value = output[key].float()
                if (
                    value.shape != sums[key][region].shape
                    or not torch.isfinite(value).all()
                ):
                    raise FloatingPointError(
                        f"Invalid shape or nonfinite output: {key}"
                    )
                sums[key][region] += (
                    value.sigmoid() if key in ("mask", "contour") else value
                )
            count[region] += 1
    if not (count > 0).all():
        raise RuntimeError("Sliding validation left uncovered pixels")
    return {key: value / count for key, value in sums.items()}
