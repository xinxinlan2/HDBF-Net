"""Checkpoint/evaluator integration with real models and synthetic IO only."""
import contextlib
import gc
import io
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bootstrap
import torch
from ablation_spacenet6 import train as training
from ablation_spacenet6 import eval as evaluation
from ablation_spacenet6.tests import build
from ablation_spacenet6.protocol import CASES, REFERENCE_SPLITS, train_arguments


class Synthetic(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs): pass
    def __len__(self): return 1
    def __getitem__(self, index):
        return dict(rgb=torch.zeros(3, 64, 64), sar=torch.ones(4, 64, 64), mask=torch.ones(1, 64, 64),
                    contour=torch.zeros(1, 64, 64), height=torch.ones(1, 64, 64) * 8,
                    valid_height=torch.ones(1, 64, 64, dtype=torch.bool))


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="sn6-ablation-eval-") as folder:
        for case in CASES:
            with patch.object(sys, "argv", ["train.py"] + train_arguments(case, folder, "fixture.pth")):
                args = training.parse_args()
            args.decoder_dim, args.task_channels = 64, 32
            args.val_window, args.val_stride = 64, 32
            args.height_definition = "pixel-level synthetic fixture"
            args.split_metadata = REFERENCE_SPLITS
            model = build(case).eval()
            ema = type("EMA", (), {"model_state_dict": lambda self: model.state_dict()})()
            with patch.object(training.torch, "save") as save:
                training.save_checkpoint("unused.pth", model, ema, None, None, None, 43, {"mae": 1.}, args)
            package = save.call_args.args[0]
            assert package["model_config"]["ablation_config"] == args.ablation_config
            del ema, model
            gc.collect()
            destination = Path(folder) / f"{case}.txt"
            argv = ["eval.py", "--checkpoint", "fixture.pth", "--base-dir", folder, "--split", "val",
                    "--device", "cpu", "--no-amp", "--num-workers", "0", "--output", str(destination)]
            with patch.object(sys, "argv", argv), patch.object(evaluation.torch, "load", return_value=package), \
                 patch.object(evaluation, "SpaceNet6HeightDataset", Synthetic), \
                 patch.object(evaluation, "split_metadata", return_value=REFERENCE_SPLITS), \
                 patch.object(evaluation, "load_dataset_metadata", return_value={"height_definition": "pixel-level synthetic fixture"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluation.main()
            report = destination.read_text(encoding="utf-8")
            lines = report.strip().splitlines()
            assert len(lines) == 3 and all(len(line.split("|")) == 11 for line in lines)
            assert lines[0] == "| MaskF1 | MaskIoU | ContourF1 | AbsRel | RMSE(m) | \u03b41 | \u03b42 | \u03b43 | MAE(m) |"
            assert len([float(value) for value in lines[2].split("|")[1:-1]]) == 9
            assert all(word not in report for word in ("Checkpoint", "Precision", "Full Image", "pixels="))
            # Metadata and the strict state marker are required, not guessed.
            for bad in ("missing_config", "missing_marker"):
                modified = dict(package, model_config=dict(package["model_config"]))
                if bad == "missing_config":
                    modified["model_config"].pop("ablation_config")
                else:
                    modified["model_state_dict"] = dict(package["model_state_dict"])
                    modified["model_state_dict"].pop("ablation_marker")
                with patch.object(sys, "argv", argv), patch.object(evaluation.torch, "load", return_value=modified), \
                     patch.object(evaluation, "split_metadata", return_value=REFERENCE_SPLITS), \
                     patch.object(evaluation, "load_dataset_metadata", return_value={"height_definition": "pixel-level synthetic fixture"}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    try:
                        evaluation.main()
                    except (ValueError, RuntimeError) as error:
                        assert "ablation" in str(error)
                    else:
                        raise AssertionError((case, bad))
            del package, modified
            gc.collect()
            print(f"PASS: {case}: real evaluator, exactly nine metrics, strict rejection of missing metadata/state marker", flush=True)


if __name__ == "__main__":
    main()
