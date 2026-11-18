"""Numerical/dataflow tests only, not dataset accuracy or batch-eight VRAM."""
import argparse
import contextlib
import gc
import hashlib
import io
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bootstrap
import torch
from torch.nn import functional as F
from ablation_spacenet6 import train as training
from ablation_spacenet6.model import AblationDinoHDBF, ConcatenateProjection, configuration, read_configuration
from ablation_spacenet6.protocol import CASES, train_arguments
from hdbf.unirep_structure import UniRepStructure
from hdbf.geometry_height import GeometryHeightFusion
from semseg.models.hdbf_net import DinoHDBF


def check_weights(root, cnn):
    expected = {Path(cnn): "67efb2494f12e2bccaa19282b8d2eb0d466fe5755584e5fb9903cd188ebc0e64",
        root / "checkpoints/pretrained/dinov2/dinov2_vitb14_pretrain.pth": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"}
    for path, digest in expected.items():
        actual = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                actual.update(chunk)
        if actual.hexdigest() != digest:
            raise ValueError(f"Pretrained weight mismatch: {path}")
    if not (root / "checkpoints/pretrained/segformer/segformer_b4_ade20k.bin").is_file():
        raise FileNotFoundError("Missing reference B4 SAR pretraining")


def build(case, original=False):
    torch.manual_seed(42)
    spec = configuration(case)
    kwargs = dict(fusion_type="task-aware", task_split_decoder=True, decoder_dim=64, task_channels=32,
        gradient_checkpointing=True, use_hrbr=False, context_detail_height=True, building_affinity=True,
        use_ttci=spec["a"], use_hcr=True, use_slmhc=spec["c"], sar_channels=4)
    model = DinoHDBF(**kwargs) if original else AblationDinoHDBF(ablation_case=case, **kwargs)
    if spec["b"]:
        with torch.random.fork_rng(devices=[]):
            model.unirep_structure = UniRepStructure(model.backbone.channels, 64, local_structure=True)
        with torch.random.fork_rng(devices=[]):
            model.height_geometry_fusion = GeometryHeightFusion(64, mode="asymmetric-local")
        model.unirep_structure.local_decoder.set_epoch(43)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-size", action="store_true")
    options = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    control = build("full", original=True).eval()
    reference = control.state_dict()
    full = build("full").eval()
    assert set(reference) == set(full.state_dict())
    for key, value in full.state_dict().items():
        assert torch.equal(value, reference[key]), key
    rgb, sar = torch.randn(1, 3, 65, 79), torch.randn(1, 4, 65, 79)
    with torch.no_grad():
        a, b = control(rgb, sar), full(rgb, sar)
    for key in ("mask", "contour", "height"):
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    del full, a, b
    print("PASS: full-subclass state and forward are bit-identical to original full model", flush=True)

    for case in CASES:
        with patch.object(sys, "argv", ["train.py"] + train_arguments(case, "/synthetic", "cnn.pth")):
            args = training.parse_args()
        spec = configuration(case)
        assert args.ablation_config == spec and args.hcr and args.bihm and args.context_detail_height
        assert args.ttci == spec["a"] and args.slmhc == spec["c"]
        assert args.local_structure == spec["b"] and args.geometry_height_fusion == spec["b"]
        assert args.slmhc_tail_weight == (.05 if spec["c"] else 0)
        assert (args.local_structure_weight, args.structure_branch_weight) == ((.2, .2) if spec["b"] else (0, 0))
        assert (args.epochs, args.batch_size, args.seed, args.data_seed) == (100, 8, 42, 42)
        assert (args.lr, args.rgb_backbone_lr_scale, args.ema_bn_policy) == (3e-5, .1, "copy")
        for wrong in (["--no-hcr"], ["--no-bihm"], ["--no-local-structure"], ["--lr", "6e-5"]):
            with patch.object(sys, "argv", ["train.py"] + train_arguments(case, "/synthetic", "cnn.pth") + wrong), \
                 contextlib.redirect_stderr(io.StringIO()):
                try:
                    training.parse_args()
                except SystemExit as error:
                    assert error.code == 2
                else:
                    raise AssertionError((case, wrong))
        model = build(case)
        count = 0
        for key, value in model.state_dict().items():
            if key in reference:
                assert torch.equal(value, reference[key]), (case, key)
                count += 1
        assert count > 100
        keys = list(model.state_dict())
        if not spec["a"]:
            assert all(isinstance(m, ConcatenateProjection) for m in model.backbone.task_fusions)
            assert not hasattr(model, "task_interaction") and not hasattr(model, "mask_adapter")
        if not spec["b"]:
            assert not hasattr(model, "unirep_structure") and not hasattr(model, "height_geometry_fusion")
        if not spec["c"]:
            assert not hasattr(model, "height_calibrator")
        if not spec["sar"]:
            assert not any(k.startswith("backbone.extra_") or k.startswith("backbone.task_fusions") for k in keys)
        # Strict round trip, and no unused/deleted parameters in the optimizer.
        model.load_state_dict(model.state_dict(), strict=True)
        config = dict(ablation_config=spec, encoder="dinov2", fusion_type="task-aware", task_split_decoder=True,
            local_structure=spec["b"], unirep_structure=spec["b"], geometry_height_fusion=spec["b"],
            context_detail_height=True, building_affinity=True, use_hrbr=False, use_hcr=True, use_bihm=True,
            use_ttci=spec["a"], use_slmhc=spec["c"])
        assert read_configuration(config, {}) == spec
        try:
            read_configuration(dict(config, use_hcr=False), {})
        except ValueError:
            pass
        else:
            raise AssertionError("Contradictory effective flags accepted")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
        del optimizer
        model.to(device).train()
        if spec["b"]:
            torch.nn.init.normal_(model.height_geometry_fusion.output.weight, std=.01)
        image = torch.randn(2, 3, 64, 64, device=device)
        radar = torch.randn(2, 4, 64, 64, device=device, requires_grad=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
            output = model(image, radar, return_intermediates=True)
            mask = torch.zeros_like(output["mask"])
            mask[..., 12:50, 12:50] = 1
            target = torch.full_like(mask, 8.)
            valid = mask > .5
            loss = (training.bce_dice_loss(output["mask"], mask)
                + .4 * training.bce_dice_loss(output["contour"], mask)
                + 1.2 * training.masked_height_loss(output["height"].float(), target, valid)
                + .2 * F.smooth_l1_loss(output["height_head"].float()[valid], target[valid])
                + .05 * training.height_range_loss(output["hcr_gate"], target, valid, (3., 20.)))
            if spec["c"]:
                loss += .05 * training.directional_tail_loss(output["height"], target, valid, 3., 20.)
            if spec["b"]:
                from hdbf.local_structure import local_structure_loss
                loss += .2 * local_structure_loss(output, mask, mask)
                loss += .2 * sum(training.bce_dice_loss(output[f"unirep_{branch}_mask"], mask)
                                  for branch in ("cnn", "semantic")) / 2
            if not spec["a"]:
                torch.testing.assert_close(output["height_head"], output["height_coarse"], atol=0, rtol=0)
            if not spec["c"]:
                torch.testing.assert_close(output["height_after_hcr"], output["height_after_slmhc"], atol=0, rtol=0)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if not spec["sar"]:
            assert radar.grad is None
        model.zero_grad(set_to_none=True)
        del output, loss
        model.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
            image = torch.randn(1, 3, 65, 79, device=device)
            output = model(image, torch.randn(1, 4, 65, 79, device=device))
            if not spec["sar"]:
                no_sar = model(image, None)
                for key in ("mask", "contour", "height"):
                    torch.testing.assert_close(output[key], no_sar[key], atol=0, rtol=0)
            for key in ("mask", "contour", "height"):
                assert output[key].shape == (1, 1, 65, 79) and torch.isfinite(output[key]).all()
            if options.full_size:
                del output
                output = model(torch.randn(1, 3, 512, 512, device=device), torch.randn(1, 4, 512, 512, device=device))
                assert all(output[key].shape == (1, 1, 512, 512) and torch.isfinite(output[key]).all() for key in ("mask", "contour", "height"))
        print(f"PASS: {case}: shared initialization, real removal, effective recipe, strict state, AMP backward and odd-size inference", flush=True)
        del model, output, image, radar
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("PASS: four isolated ablations; no dataset accuracy or full batch-eight training claim", flush=True)


if __name__ == "__main__":
    main()
