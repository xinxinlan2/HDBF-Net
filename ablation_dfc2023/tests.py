"""Local regression tests with synthetic tiles, not dataset performance tests."""
import contextlib
import gc
import io
import json
from pathlib import Path
import shlex
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import torch
import train_dfc2023 as original
from ablation_dfc2023 import eval as evaluation, run, train
from ablation_dfc2023.model import AblationDinoHDBF, ConcatenateProjection, apply_case, configuration, restore_model
from ablation_dfc2023.protocol import CASES, train_arguments, verify_reference
from hdbf.geometry_height import GeometryHeightFusion
from hdbf.local_structure import local_structure_loss
from hdbf.unirep_structure import UniRepStructure
from semseg.models.hdbf_net import DinoHDBF


def parse(parser, arguments):
    with patch.object(sys, "argv", ["train.py", *arguments]):
        return parser()


def build(case, original_model=False):
    spec = configuration(case)
    torch.manual_seed(42)
    kwargs = dict(fusion_type="task-aware", task_split_decoder=True, decoder_dim=64, task_channels=32,
        use_hrbr=False, use_hcr=True, use_ttci=spec["a"], use_slmhc=spec["c"],
        sar_channels=1, building_affinity=True, context_detail_height=True)
    model = DinoHDBF(**kwargs) if original_model else AblationDinoHDBF(ablation_case=case, **kwargs)
    if spec["b"]:
        with torch.random.fork_rng(devices=[]):
            model.unirep_structure = UniRepStructure(model.backbone.channels, 64, local_structure=True)
        with torch.random.fork_rng(devices=[]):
            model.height_geometry_fusion = GeometryHeightFusion(64, mode="asymmetric-local")
        model.unirep_structure.local_decoder.set_epoch(50)
    return model


def test_metric_region():
    class FixedModel:
        rgb_only = True

        def __call__(self, rgb, sar, return_intermediates=False):
            assert sar is None
            return {k: torch.tensor(v).reshape(1, 1, 1, 4) for k, v in dict(
                mask=[1., -1., 1., 1.], contour=[-1., 1., -1., 1.], height=[500., 24., 500., 100.]).items()}

    tile = {"rgb": torch.zeros(1, 3, 1, 4)}
    tile.update({k: torch.tensor(v).reshape(1, 1, 1, 4) for k, v in dict(
        mask=[1., 1., 0., 1.], contour=[0., 1., 1., 0.], height=[10., 20., 70., .05],
        valid_height=[False, True, True, True]).items()})
    actual = evaluation.evaluate(FixedModel(), [tile], torch.device("cpu"), False, .1)
    torch.testing.assert_close(torch.tensor(actual), torch.tensor([2/3, .5, .5, .2, 4., 1., 1., 1., 4.]))
    assert all(len(row.split("|")) == 11 for row in evaluation.format_metrics(actual).splitlines())


def main():
    torch.set_num_threads(2)
    test_metric_region()
    run.ROOT = bootstrap.ROOT
    run.verify_sources()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    launcher = (bootstrap.ROOT / "run_train_dfc2023.sh").read_text(encoding="utf-8")
    command = launcher.split("python -u train_dfc2023.py \\", 1)[1].split("2>&1", 1)[0]
    reference_args = parse(original.parse_args, shlex.split(command.replace("\\\n", " ")))
    with tempfile.TemporaryDirectory(prefix="dfc23-ablation-") as tmp:
        reference = Path(tmp) / "effective_config.json"
        reference.write_text(json.dumps({"args": vars(reference_args)}), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            full = build("full", original_model=True)
            same = build("full")
        state = full.state_dict()
        assert state.keys() == same.state_dict().keys()
        assert all(torch.equal(value, state[key]) for key, value in same.state_dict().items())
        del same, full
        for case in CASES:
            argv = train_arguments(case, tmp, tmp, "cnn.pth", reference)
            args = parse(train.parse_args, argv)
            verify_reference(args)
            args.lr = 6e-5
            try:
                verify_reference(args)
            except ValueError as error:
                assert "lr" in str(error)
            else:
                raise AssertionError("Changed learning rate passed the reference guard")
            args = apply_case(parse(train.parse_args, argv))
            spec = args.ablation_config
            assert args.hcr and args.bihm and args.context_detail_height and args.building_affinity
            assert args.local_structure == spec["b"]
            assert args.slmhc_tail_weight == (.05 if spec["c"] else 0.)
            with contextlib.redirect_stdout(io.StringIO()):
                model = build(case)
            assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items() if key in state)
            assert model.sar_channels == 1
            assert model.rgb_only == (case == "rgb_only")
            if case == "wo_a":
                assert not hasattr(model, "task_interaction")
                assert all(isinstance(layer, ConcatenateProjection) for layer in model.backbone.task_fusions)
            if case == "wo_b":
                assert not hasattr(model, "unirep_structure") and not hasattr(model, "height_geometry_fusion")
            if case == "wo_c":
                assert not hasattr(model, "height_calibrator")
            if case == "rgb_only":
                assert not any(key.startswith(("backbone.extra_", "backbone.task_fusions.")) for key in model.state_dict())

            model.to(device).train()
            rgb = torch.randn(2, 3, 64, 64, device=device)
            sar = torch.randn(2, 1, 64, 64, device=device, requires_grad=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
                out = model(rgb, sar, return_intermediates=True)
                mask = torch.zeros_like(out["mask"])
                mask[..., 16:48, 16:48] = 1
                target = torch.full_like(mask, 12.)
                valid = mask > .5
                loss = (train.bce_dice_loss(out["mask"], mask) + .4 * train.bce_dice_loss(out["contour"], mask)
                        + 1.2 * train.final_height_loss(out["height"], target, valid)
                        + .2 * train.height_head_auxiliary_loss(out, target, valid)
                        + .05 * train.height_range_loss(out["hcr_gate"], target, valid))
                if spec["b"]:
                    loss += .2 * local_structure_loss(out, mask, mask, 1., .4)
                if not spec["a"]:
                    torch.testing.assert_close(out["height_head"], out["height_coarse"], atol=0, rtol=0)
                if not spec["c"]:
                    torch.testing.assert_close(out["height_after_hcr"], out["height_after_slmhc"], atol=0, rtol=0)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if model.rgb_only:
                assert sar.grad is None
                model.eval()
                with torch.no_grad():
                    for key, value in model(rgb, None).items():
                        torch.testing.assert_close(value, model(rgb, sar)[key], atol=0, rtol=0)
            model.zero_grad(set_to_none=True)
            del out, loss, rgb, sar, mask, target, valid
            model.cpu().eval()
            args.decoder_dim, args.task_channels = 64, 32
            args.split_counts, args.split_hashes = {}, {}
            ema = type("EMA", (), {"model_state_dict": lambda self: model.state_dict()})()
            with patch.object(torch, "save") as save:
                best = {"mae": float("inf")}
                train.save_epoch_checkpoints(tmp, model, ema, None, None, None, 30, {"mae": 2.}, args, best)
                train.save_epoch_checkpoints(tmp, model, ema, None, None, None, 100, {"mae": 3.}, args, best)
                assert save.call_count == 1
                package, path = save.call_args.args
                assert path.endswith("checkpoint_best_mae.pth")
            assert "optimizer_state_dict" not in package
            with contextlib.redirect_stdout(io.StringIO()):
                restored = restore_model(package).eval()
            assert all(torch.equal(value, package["model_state_dict"][key])
                       for key, value in restored.state_dict().items())
            with torch.no_grad():
                rgb, sar = torch.randn(1, 3, 64, 64), torch.randn(1, 1, 64, 64)
                expected, actual = model(rgb, sar), restored(rgb, sar)
                for key in ("mask", "contour", "height"):
                    # CPU inference after a GPU training pass may differ in
                    # reduction order even with exact checkpoint tensors.
                    torch.testing.assert_close(actual[key], expected[key], atol=2e-5, rtol=1e-5)
            print(f"PASS {case}: reference recipe, shared initialization, AMP backward, best-only EMA, strict reload", flush=True)
            del restored, package, model, ema, save, expected, actual
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print("PASS: nine metrics with GT-building validity, active source guard, four isolated DFC23 cases")


if __name__ == "__main__":
    main()
