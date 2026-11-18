"""Four-band SpaceNet6 local-structure preflight; no dataset accuracy claim."""
import argparse
import gc
import inspect
from pathlib import Path
import shlex
import sys
import tempfile
import textwrap

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--full-size", action="store_true")
    options = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    import bootstrap
    import train_spacenet6 as trainer
    from hdbf.local_structure import LocalStructureDecoder, local_structure_loss
    from hdbf.spacenet6_validation_windows import predict_validation_windows
    from hdbf.unirep_structure import UniRepStructure
    from hdbf.geometry_height import GeometryHeightFusion
    from semseg.models.hdbf_net import DinoHDBF
    launcher = (ROOT / "run_train_spacenet6_taskfusion.sh").read_text().replace("\\\n", " ")
    tokens = shlex.split(launcher.split("python -u train_spacenet6.py", 1)[1].split("2>&1", 1)[0])
    saved_argv = sys.argv
    sys.argv = ["train_spacenet6.py"] + tokens
    args = trainer.parse_args()
    sys.argv = saved_argv
    assert args.local_structure and not args.hrbr and not args.structure_isolation
    assert args.geometry_height_fusion
    assert args.geometry_height_mode=='asymmetric-local'
    assert args.ttci and args.hcr and args.slmhc and args.contour_objective == "bce-dice"
    assert args.epochs == 100 and args.lr == 3e-5 and args.ema_bn_policy == "copy"
    assert args.val_inference == "sliding" and args.val_window == 768 and args.val_stride == 384
    assert args.save_task_best and args.save_best_mae_only
    args.split_metadata = {}
    args.height_definition = "synthetic checkpoint metadata test"
    toy = torch.nn.ModuleDict({"decoder": LocalStructureDecoder(), "bn": torch.nn.BatchNorm2d(2),
                              "height_geometry_fusion": GeometryHeightFusion(mode=args.geometry_height_mode)})
    toy.architecture = "hdbf_dinov2_metadata_test"
    toy["decoder"].set_epoch(40)
    ema = trainer.ModelEMA(toy, args.ema_decay, args.ema_bn_policy)
    toy["bn"].running_mean.fill_(3)
    ema.update(toy)
    torch.testing.assert_close(ema.shadow["bn.running_mean"], toy["bn"].running_mean)
    optimizer = torch.optim.AdamW(toy.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    with tempfile.TemporaryDirectory(prefix="sn6-local-metadata-") as folder:
        for mode in ("inference", "full"):
            args.checkpoint_mode = mode
            path = Path(folder) / (mode + ".pth")
            trainer.save_checkpoint(path, toy, ema, optimizer, scheduler, scaler, 40, {}, args)
            ckpt = torch.load(path, map_location="cpu")
            assert ckpt["model_config"]["local_structure"] and ckpt["model_config"]["unirep_structure"]
            assert ckpt["model_config"]["sar_channels"] == 4
            assert ckpt["model_config"]["geometry_height_fusion"]
            assert ckpt["model_config"]["geometry_height_mode"]==args.geometry_height_mode
            assert ckpt['model_state_dict']['height_geometry_fusion.local_conditioning'].item()==1
            toy.load_state_dict(ckpt["model_state_dict"], strict=True)
            assert toy["decoder"].guidance_strength().item() == 1
    del toy, ema, optimizer, scheduler, ckpt
    print("PASS: actual launcher flags, sliding protocol, inference/full metadata, BN-copy and epoch-buffer reload", flush=True)
    model = DinoHDBF(fusion_type="task-aware", task_split_decoder=True, gradient_checkpointing=True,
        use_hrbr=False, use_ttci=True, use_hcr=True, use_slmhc=True, context_detail_height=True,
        building_affinity=True, sar_channels=4, hcr_residual_scales=args.hcr_residual_scales,
        slmhc_knots=args.slmhc_knots)
    model.unirep_structure = UniRepStructure(model.backbone.channels, local_structure=True)
    assert model.sar_channels == 4 and not hasattr(model, "geometry_interaction")
    weight_root = Path(bootstrap.ROOT)
    model.init_pretrained(weight_root / "checkpoints/pretrained/dinov2/dinov2_vitb14_pretrain.pth")
    model.init_sar_pretrained(weight_root / "checkpoints/pretrained/segformer/segformer_b4_ade20k.bin")
    assert model.unirep_structure.load_pretrained(options.checkpoint)["loaded"] == {"cnn": 1309, "ade_decoder": 72}
    model.eval()
    with torch.no_grad():
        rgb,sar=torch.randn(1,3,65,79),torch.randn(1,4,65,79)
        original=model(rgb,sar,return_intermediates=True)
        with torch.random.fork_rng(devices=[]):
            model.height_geometry_fusion=GeometryHeightFusion(args.decoder_dim,mode=args.geometry_height_mode)
        events=[]
        hooks=[module.register_forward_hook(lambda m,x,y,name=name: events.append(name))
               for name,module in (('cnn',model.unirep_structure.cnn),('ade',model.unirep_structure.ade_decoder),
                   ('bridge',model.height_geometry_fusion),('head',model.height_head),
                   ('ttci',model.task_interaction),('hcr',model.height_conditional_residual),
                   ('slmhc',model.height_calibrator),('local',model.unirep_structure.local_decoder))]
        enhanced=model(rgb,sar,return_intermediates=True)
        for hook in hooks: hook.remove()
        assert events==['cnn','ade','bridge','head','ttci','hcr','slmhc','local'], events
        for key in ('mask','contour','height','height_head','height_coarse','height_after_hcr','height_after_slmhc'):
            torch.testing.assert_close(original[key],enhanced[key],rtol=0,atol=0)
        assert enhanced['mask'].shape==(1,1,65,79)
    del original,enhanced,rgb,sar
    print('PASS: zero-bridge full-model identity at odd65x79, one CNN/ADE encode, acyclic height-before-local sequence',flush=True)
    # Exercise the real optimizer grouping block, without loading a dataset.
    source = inspect.getsource(trainer.main)
    begin = source.index("    rgb_backbone_parameters = []")
    end = source.index("    if args.lr_schedule", begin)
    scope = dict(model=model, args=args, trainable_parameters=list(model.parameters()), torch=torch)
    exec(compile(textwrap.dedent(source[begin:end]), "sn6-optimizer-block", "exec"), scope)
    optimizer = scope["optimizer"]
    groups = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert groups == {"rgb_backbone": 3e-6, "task": 3e-5, "unirep_pretrained": 3e-6}
    ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    del optimizer, scope
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    model.unirep_structure.local_decoder.set_epoch(40)
    rgb, sar = torch.randn(2, 3, 64, 64, device=device), torch.randn(2, 4, 64, 64, device=device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
        out = model(rgb, sar, return_intermediates=True)
        mask = torch.zeros_like(out["mask"])
        mask[..., 16:48, 16:48] = 1
        auxiliary = local_structure_loss(out, mask, mask)
        branch_loss = sum(trainer.bce_dice_loss(out[f"unirep_{branch}_mask"].float(), mask)
                          + .4 * trainer.bce_dice_loss(out[f"unirep_{branch}_contour"].float(), mask)
                          for branch in ("cnn", "semantic")) / 2
        loss = trainer.bce_dice_loss(out["mask"].float(), mask) + .4 * trainer.bce_dice_loss(
            out["contour"].float(), mask) + F.smooth_l1_loss(out["height"].float(),
            torch.full_like(out["height"], 10.)) + .2 * auxiliary + .2 * branch_loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.unirep_structure.local_decoder.parameters())
    assert next(p for p in model.unirep_structure.cnn.parameters() if p.ndim==4).grad.abs().sum()>0
    print("PASS: four-band SAR, actual UniRep pretraining, optimizer coverage/LRs and joint AMP backward", flush=True)
    model.zero_grad(set_to_none=True)
    del out, loss, auxiliary, branch_loss
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.eval()
    # Height-only updates train the bridge, not its CNN references. The original
    # geometry decoder still trains the CNN, as checked in the joint loss above.
    for attempt in range(2):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=False):
            height_out=model(torch.randn(1,3,64,64,device=device),
                             torch.randn(1,4,64,64,device=device),return_intermediates=True)
            height_loss=height_out['height_head'].float().mean()
        height_loss.backward()
        stem=next(p for p in model.unirep_structure.cnn.parameters() if p.ndim==4)
        if attempt==0:
            assert model.height_geometry_fusion.output.weight.grad.abs().sum()>0
            assert stem.grad is None or stem.grad.abs().sum()==0
            with torch.no_grad():
                model.height_geometry_fusion.output.weight.add_(model.height_geometry_fusion.output.weight.grad,alpha=-.001)
        else:
            assert stem.grad is None or stem.grad.abs().sum()==0
            assert model.height_geometry_fusion.geometry_projections[0][0].weight.grad.abs().sum()>0
            assert model.height_geometry_fusion.ade_projection[0].weight.grad.abs().sum()>0
            assert model.backbone.rgb_projections[0][0].weight.grad.abs().sum()>0
        assert model.unirep_structure.cnn_head.weight.grad is None
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        del height_out,height_loss
    model.zero_grad(set_to_none=True)
    print('PASS: height-only gradients train bridge/shared height features, not CNN references; geometry loss still trains CNN',flush=True)
    # Four overlapping windows, using the same helper in formal train/eval paths.
    with torch.no_grad():
        bn_mean = next(m.running_mean for m in model.unirep_structure.modules() if isinstance(m, torch.nn.BatchNorm2d)).clone()
        stitched = predict_validation_windows(model, torch.randn(1, 3, 80, 88, device=device),
            torch.randn(1, 4, 80, 88, device=device), 64, 32, device.type == "cuda", True)
        torch.testing.assert_close(bn_mean, next(m.running_mean for m in model.unirep_structure.modules()
                                                if isinstance(m, torch.nn.BatchNorm2d)))
    assert all(v.shape == (1, 1, 80, 88) and torch.isfinite(v).all() for v in stitched.values())
    assert all(((stitched[k] >= 0) & (stitched[k] <= 1)).all() for k in ("mask", "contour"))
    print("PASS: four-window overlap stitching, probability averaging, height-stage outputs and fixed-BN inference", flush=True)
    if options.full_size:
        del stitched
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
            out = model(torch.randn(1, 3, 768, 768, device=device), torch.randn(1, 4, 768, 768, device=device))
        assert all(out[k].shape == (1, 1, 768, 768) and torch.isfinite(out[k]).all() for k in ("mask", "contour", "height"))
        print("PASS: actual 768x768 batch1 inference window, NOT 512x512 batch8 training memory", flush=True)
        del out
        gc.collect()
        if device.type=='cuda':
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        model.train()
        # PPM has a1x1 BN branch; batch1 only works with BN fixed in this
        # memory/numerics smoke test. Production training still uses batch8 BN.
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm): module.eval()
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
            large=model(torch.randn(1,3,512,512,device=device),
                        torch.randn(1,4,512,512,device=device),return_intermediates=True)
            target=torch.zeros_like(large['mask']);target[...,128:384,128:384]=1
            loss=trainer.bce_dice_loss(large['mask'].float(),target)+.4*trainer.bce_dice_loss(
                large['contour'].float(),target)+F.smooth_l1_loss(large['height'].float(),
                torch.full_like(large['height'],10.))+.2*local_structure_loss(large,target,target)
        test_optimizer=torch.optim.SGD(model.parameters(),lr=0.)
        test_scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda',init_scale=128.)
        test_scaler.scale(loss).backward();test_scaler.unscale_(test_optimizer)
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        peak=torch.cuda.max_memory_allocated()/1e9 if device.type=='cuda' else None
        print(f'PASS: actual512x512 batch1 AMP backward, BN fixed for smoke only; no optimizer step; peak allocated GB={peak}; NOT batch8 memory',flush=True)


if __name__ == "__main__":
    main()
