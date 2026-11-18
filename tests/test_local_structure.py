"""Numerical/dataflow preflight, not dataset accuracy or batch-8 memory testing."""
import argparse
import gc
import inspect
import io
from pathlib import Path
import sys
import types

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--full-size", action="store_true")
    parser.add_argument("--module-only", action="store_true")
    parser.add_argument("--geometry-height-fusion", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from hdbf.local_structure import LocalStructureDecoder, local_structure_loss
    decoder = LocalStructureDecoder().to(device)
    for epoch, expected in ((0, 0), (20, 0), (30, .5), (40, 1)):
        decoder.set_epoch(epoch)
        assert decoder.guidance_strength().item() == expected
    for head in (decoder.mask_seed[-1], decoder.contour_seed[-1]):
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    with torch.no_grad():
        flat = decoder(torch.zeros(1, 192, 8, 10, device=device), torch.zeros(1, 32, 8, 10, device=device),
            torch.zeros(1, 16, 31, 39, device=device), torch.full((1, 2, 8, 10), 3., device=device),
            torch.zeros(1, 1, 31, 39, device=device))
    torch.testing.assert_close(flat["mask"], torch.full_like(flat["mask"], 3.), atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(flat["contour"], torch.full_like(flat["contour"], 3.), atol=2e-6, rtol=1e-6)
    # Exercise the learned path rather than only its zero-initialized residuals.
    for head in (decoder.half_mask, decoder.half_contour, decoder.native_mask[-1], decoder.native_contour[-1]):
        torch.nn.init.normal_(head.weight, std=.01)
    joint = torch.randn(2, 192, 8, 10, device=device, requires_grad=True)
    height = torch.rand(2, 1, 31, 39, device=device, requires_grad=True)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
        out = decoder(joint, torch.randn(2, 32, 8, 10, device=device),
            torch.randn(2, 16, 31, 39, device=device), torch.randn(2, 2, 8, 10, device=device), height)
        for mode in ("empty", "full", "thin"):
            mask = torch.zeros_like(out["mask"])
            if mode == "full":
                mask.fill_(1)
            elif mode == "thin":
                mask[..., 15, 10:30] = 1
            auxiliary = local_structure_loss(out, mask, mask)
            assert torch.isfinite(auxiliary), mode
        loss = out["mask"].square().mean() + out["contour"].square().mean() + auxiliary
    loss.backward()
    assert height.grad is None and joint.grad.abs().max() > 0
    assert decoder.affinity.weight.grad is not None and decoder.affinity.weight.grad.abs().max() > 0
    assert all(torch.isfinite(p.grad).all() for p in decoder.parameters() if p.grad is not None)
    import bootstrap
    from train_dfc2023 import ModelEMA, make_optimizer
    container = torch.nn.ModuleDict({"decoder": decoder, "bn": torch.nn.BatchNorm2d(2).to(device)})
    ema = ModelEMA(container, .9, bn_policy="copy")
    decoder.set_epoch(60)
    container["bn"].running_mean.fill_(3)
    ema.update(container)
    assert ema.shadow["decoder.training_epoch"].item() == 60
    torch.testing.assert_close(ema.shadow["bn.running_mean"], container["bn"].running_mean)
    stream = io.BytesIO()
    torch.save(ema.model_state_dict(), stream)
    stream.seek(0)
    container.load_state_dict(torch.load(stream, map_location=device), strict=True)
    print("PASS: empty/full/thin labels, constant-logit preservation including tile borders, odd sizes, AMP backward, height detach, EMA/BN copy and strict helper reload", flush=True)
    del decoder, container, ema, stream, out, loss, auxiliary, flat, joint, height
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if args.module_only:
        return

    from semseg.models.hdbf_net import DinoHDBF
    from semseg.models.cmnext_mtl import CMNeXtMTL
    from hdbf.unirep_structure import UniRepStructure
    if "local_structure" not in inspect.getsource(CMNeXtMTL.forward):
        raise RuntimeError("Mixed source files: upload semseg/models/cmnext_mtl.py before running this experiment")
    model = DinoHDBF(fusion_type="task-aware", task_split_decoder=True, geometry_interaction=False,
        gradient_checkpointing=True, use_hrbr=False, use_ttci=True, use_hcr=True, use_slmhc=True,
        context_detail_height=True, building_affinity=True)
    model.unirep_structure = UniRepStructure(model.backbone.channels, local_structure=True)
    assert not hasattr(model, "geometry_interaction") and not model.unirep_structure.query_structure
    assert model.unirep_structure.cnn_projection[0].out_channels == 192
    assert model.unirep_structure.load_pretrained(args.checkpoint)["loaded"] == {"cnn": 1309, "ade_decoder": 72}
    if args.geometry_height_fusion:
        from hdbf.geometry_height import GeometryHeightFusion
        model.height_geometry_fusion = GeometryHeightFusion(mode="asymmetric-local")
        model.eval()
        probe_rgb, probe_sar = torch.randn(1, 3, 64, 64), torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            bridged = model(probe_rgb, probe_sar)
            bridge = model.height_geometry_fusion
            del model.height_geometry_fusion
            original = model(probe_rgb, probe_sar)
            model.height_geometry_fusion = bridge
        for key in ("mask", "contour", "height"):
            torch.testing.assert_close(bridged[key], original[key], atol=0, rtol=0)
        # Exercise nonzero bridge gradients, not only its identity initialization.
        torch.nn.init.normal_(bridge.output.weight, std=.01)
        del bridged, original, probe_rgb, probe_sar
        print("PASS: zero bridge preserves DFC23 original Local outputs exactly", flush=True)
    optimizer = make_optimizer(model, 3e-5, .01, .1, .1)
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert {g["name"]: g["lr"] for g in optimizer.param_groups} == {
        "rgb_backbone": 3e-6, "fusion_and_tasks": 3e-5, "unirep_pretrained": 3e-6}
    del optimizer
    model.to(device).train()
    model.unirep_structure.local_decoder.set_epoch(40)
    rgb, sar = torch.randn(2, 3, 64, 64, device=device), torch.randn(2, 1, 64, 64, device=device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
        out = model(rgb, sar, return_intermediates=True)
        mask = torch.zeros_like(out["mask"])
        mask[..., 16:48, 16:48] = 1
        auxiliary = local_structure_loss(out, mask, mask)
        height_loss = F.smooth_l1_loss(out["height"].float(), torch.full_like(out["height"], 10.))
        if args.geometry_height_fusion:
            from train_dfc2023 import (final_height_loss, height_head_auxiliary_loss,
                height_range_loss, directional_tail_loss, building_instance_height_loss)
            target = torch.full_like(out["height"], 10.)
            valid = mask > .5
            height_loss = (1.2 * final_height_loss(out["height"], target, valid)
                           + .2 * height_head_auxiliary_loss(out, target, valid)
                           + .05 * height_range_loss(out["hcr_gate"], target, valid)
                           + .05 * directional_tail_loss(out["height"], target, valid)
                           + .05 * building_instance_height_loss(out["height"], target, valid.long(), valid))
        loss = F.binary_cross_entropy_with_logits(out["mask"].float(), mask) + .4 * F.binary_cross_entropy_with_logits(
            out["contour"].float(), mask) + height_loss + .2 * auxiliary
    assert torch.autograd.grad(height_loss, next(model.unirep_structure.local_decoder.parameters()),
                               allow_unused=True, retain_graph=True)[0] is None
    if args.geometry_height_fusion:
        cnn_parameter = next(model.unirep_structure.cnn.parameters())
        assert torch.autograd.grad(height_loss, cnn_parameter, allow_unused=True, retain_graph=True)[0] is None
        bridge_grad = torch.autograd.grad(height_loss, model.height_geometry_fusion.geometry_projections[0][0].weight,
                                         retain_graph=True)[0]
        assert torch.isfinite(bridge_grad).all() and bridge_grad.abs().max() > 0
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.decoder.parameters())
    if args.geometry_height_fusion:
        assert cnn_parameter.grad is not None and cnn_parameter.grad.abs().max() > 0
        print("PASS: DFC23 one-band SAR; height trains bridge but not CNN reference; geometry still trains CNN", flush=True)
    print("PASS: actual UniRep loading, complete optimizer groups, full-model small-batch AMP backward and shared feature gradients", flush=True)
    model.zero_grad(set_to_none=True)
    del out, loss, height_loss, auxiliary
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.eval()
    rgb, sar = torch.randn(1, 3, 65, 73, device=device), torch.randn(1, 1, 65, 73, device=device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
        out = model(rgb, sar, return_intermediates=True)
        method = "decode_geometry" if args.geometry_height_fusion else "forward"
        old_forward = getattr(model.unirep_structure, method)
        def altered_structure(self, *inputs, **kwargs):
            geometry = old_forward(*inputs, **kwargs)
            geometry["mask"] = geometry["mask"] + 10
            geometry["contour"] = geometry["contour"] - 10
            return geometry
        setattr(model.unirep_structure, method, types.MethodType(altered_structure, model.unirep_structure))
        altered = model(rgb, sar, return_intermediates=True)
        setattr(model.unirep_structure, method, old_forward)
    for key in ("height_context_base", "height_head", "height_coarse", "height_after_hcr", "height_after_slmhc", "height"):
        torch.testing.assert_close(out[key], altered[key], atol=0, rtol=0)
    assert all(out[k].shape == (1, 1, 65, 73) and torch.isfinite(out[k]).all() for k in ("mask", "contour", "height"))
    odd_mask = torch.zeros_like(out["mask"])
    odd_mask[..., 1, :] = 1
    assert torch.isfinite(local_structure_loss(out, odd_mask, odd_mask))
    print("PASS: odd rectangular inference; final geometry cannot change any height stage at fixed weights", flush=True)
    if args.full_size:
        del out, altered
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
            out = model(torch.randn(1, 3, 512, 512, device=device), torch.randn(1, 1, 512, 512, device=device))
        assert all(out[k].shape == (1, 1, 512, 512) and torch.isfinite(out[k]).all() for k in ("mask", "contour", "height"))
        print("PASS: 512x512 batch-1 inference, NOT full-size batch-8 training", flush=True)


if __name__ == "__main__":
    main()
