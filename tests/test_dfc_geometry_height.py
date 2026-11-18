"""DFC23 loss/config/checkpoint checks; no dataset scores or sampling changes."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import train_dfc2023 as training
    from hdbf.geometry_height import GeometryHeightFusion
    torch.set_num_threads(2)
    options = ["--unirep-pretrained", "synthetic.pth", "--fusion-type", "task-aware",
               "--task-split-decoder", "--no-hrbr", "--local-structure",
               "--geometry-height-fusion", "--geometry-height-mode", "asymmetric-local"]
    with patch.object(sys, "argv", ["train_dfc2023.py"] + options):
        args = training.parse_args()
    assert args.geometry_height_fusion and args.geometry_height_mode == "asymmetric-local"
    assert (args.height_weight, args.height_head_aux_weight, args.hcr_range_weight) == (1.2, .2, .05)
    assert (args.slmhc_tail_weight, args.bihm_weight, args.sar_clip) == (.05, .05, .5)
    for invalid in (["--geometry-height-mode", "asymmetric-local"],
                    options + ["--no-local-structure"], options + ["--no-context-detail-height"]):
        with patch.object(sys, "argv", ["train_dfc2023.py"] + invalid), contextlib.redirect_stderr(io.StringIO()):
            try:
                training.parse_args()
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError("Invalid geometry conditioning configuration accepted")
    with patch.object(sys, "argv", ["train_dfc2023.py"]):
        legacy_args = training.parse_args()
    assert not legacy_args.geometry_height_fusion and legacy_args.geometry_height_mode == "joint-global"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values = torch.tensor([.1, 2.999, 3., 19.999, 20., 50.], device=device).view(1, 1, 2, 3)
    labels = torch.tensor([0, 0, 1, 1, 2, 2], device=device).view(1, 2, 3)
    weights = torch.tensor([1.90, .47, .64], device=device)
    for selection in (None, 0, 1, 2, "empty"):
        valid = torch.ones_like(values, dtype=torch.bool)
        if selection == "empty":
            valid.zero_()
        elif selection is not None:
            valid = (labels == selection).unsqueeze(1)
        logits = torch.randn(1, 3, 2, 3, device=device, requires_grad=True)
        actual = training.height_range_loss(logits, values, valid)
        expected = (F.cross_entropy(logits.permute(0, 2, 3, 1)[valid[:, 0]], labels[valid[:, 0]], weight=weights)
                    if valid.any() else logits.sum() * 0)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        assert torch.isfinite(logits.grad).all()
    # More valid cells than FP16 can represent; test outside and inside AMP.
    for amp in (False, True) if device.type == "cuda" else (False,):
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        logits = torch.zeros(2, 3, 192, 192, dtype=dtype, device=device, requires_grad=True)
        target = torch.ones(2, 1, 192, 192, device=device)
        valid = torch.ones_like(target, dtype=torch.bool)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss = training.height_range_loss(logits, target, valid)
        torch.testing.assert_close(loss, torch.tensor(3., device=device).log(), rtol=1e-4, atol=1e-4)
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        empty_logits = torch.ones_like(logits, requires_grad=True)
        empty = training.height_range_loss(empty_logits, target, torch.zeros_like(valid))
        assert empty.item() == 0
        empty.backward()
        assert torch.isfinite(empty_logits.grad).all() and not empty_logits.grad.any()
    print("PASS: DFC23 fixed-weight CE, exact 3/20m boundaries, empty/single-class/dense labels, FP16/AMP backward", flush=True)

    model = torch.nn.Module()
    model.architecture = training.ARCHITECTURE
    model.height_geometry_fusion = GeometryHeightFusion(32, 16, 16, mode="asymmetric-local")
    ema = training.ModelEMA(model, .999, bn_policy="copy")
    args.split_counts = {key: 1 for key in ("train", "val", "test")}
    args.split_hashes = {key: key for key in args.split_counts}
    stateful = type("Stateful", (), {"state_dict": lambda self: {}})()
    with tempfile.TemporaryDirectory(prefix="dfc-conditioning-test-") as folder:
        for mode in ("inference", "full"):
            args.checkpoint_mode = mode
            path = Path(folder) / (mode + ".pth")
            training.save_checkpoint(path, model, ema, stateful, stateful, stateful, 43, {"mae": 1.}, args)
            saved = torch.load(path, map_location="cpu")
            config = saved["model_config"]
            assert config["geometry_height_fusion"] and config["geometry_height_mode"] == "asymmetric-local"
            assert saved["args"]["geometry_height_mode"] == "asymmetric-local"
            restored = torch.nn.Module()
            restored.height_geometry_fusion = GeometryHeightFusion(32, 16, 16, mode=config["geometry_height_mode"])
            restored.load_state_dict(saved["model_state_dict"], strict=True)
            assert saved["epoch"] == 43
    print("PASS: optional mode, invalid flag rejection, unchanged DFC23 recipe, both checkpoint formats and strict EMA reload", flush=True)


if __name__ == "__main__":
    main()
