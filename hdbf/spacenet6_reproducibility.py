"""SpaceNet6 run provenance and independent DataLoader worker seeding."""

from pathlib import Path as _HDBFPath
import sys as _hdbf_sys

_HDBF_ROOT = _HDBFPath(__file__).resolve().parents[1]
if str(_HDBF_ROOT) not in _hdbf_sys.path:
    _hdbf_sys.path.insert(0, str(_HDBF_ROOT))

import hashlib
import json
import os
import random

import numpy as np
import torch


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def record_initial_state(args, model, run_dir):
    args.initial_state_sha256 = state_digest(model)
    expected = getattr(args, "expected_initial_state_sha256", None)
    if expected and args.initial_state_sha256 != expected:
        raise ValueError(
            f"Initial state mismatch: expected {expected}, got {args.initial_state_sha256}"
        )
    args.randomness_protocol = "independent-loader-v1"
    path = os.path.join(run_dir, "effective_config.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
    print(f"Effective seeds: model={args.seed}; data={args.data_seed}")
    print(f"Initial state SHA256: {args.initial_state_sha256}")
    print(f"Effective configuration: {path}")


def activate_ttci_bypass(model):
    """Keep serialized TTCI tensors but disable their forward and optimizer paths."""
    if not model.use_ttci or model.ttci_legacy:
        raise ValueError("TTCI bypass requires a constructed non-legacy TTCI model")
    modules = [
        getattr(model, name)
        for name in (
            "mask_adapter",
            "contour_adapter",
            "height_adapter",
            "task_interaction",
        )
    ]
    if any(module is None for module in modules):
        raise ValueError("TTCI modules must exist before bypass activation")
    for module in modules:
        module.requires_grad_(False)
    model.use_ttci = False
    model.ttci_bypass = True
    print(
        "TTCI bypass active: initialized tensors retained; adapters and corrections skipped."
    )
