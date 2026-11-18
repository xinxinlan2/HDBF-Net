"""Feature bridge invariants; no real labels or accuracy claims."""
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from hdbf.geometry_height import GeometryHeightFusion,ChannelNorm2d
    torch.set_num_threads(2)
    for mode in ('joint-global','asymmetric-local'):
        torch.manual_seed(42)
        bridge=GeometryHeightFusion(32,16,16,mode=mode)
        height=torch.randn(2,32,9,13,requires_grad=True)
        pyramid=[torch.randn(2,16,h,w,requires_grad=True) for h,w in ((9,13),(5,7),(3,4),(2,2))]
        context=torch.randn(2,16,9,13,requires_grad=True)
        mask=torch.randn(2,1,9,13,requires_grad=True)
        cnn=torch.randn(2,2,9,13,requires_grad=True)
        adapted,stats=bridge(height,pyramid,context,mask,cnn)
        assert torch.equal(adapted,height)
        torch.testing.assert_close(stats['height_geometry_scale_weights'].sum(1),torch.ones(2,9,13))
        optimizer=torch.optim.SGD(bridge.parameters(),lr=.1)
        adapted.square().mean().backward()
        assert bridge.output.weight.grad.abs().sum()>0
        assert all(x.grad is None or x.grad.abs().sum()==0 for x in pyramid+[context])
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        for tensor in pyramid+[context,height]: tensor.grad=None
        adapted,_=bridge(height,pyramid,context,mask,cnn)
        adapted.square().mean().backward()
        if mode=='joint-global':
            assert all(x.grad is not None and x.grad.abs().sum()>0 for x in pyramid+[context])
        else:
            assert all(x.grad is None for x in pyramid+[context])
            assert bridge.geometry_projections[0][0].weight.grad.abs().sum()>0
            assert bridge.ade_projection[0].weight.grad.abs().sum()>0
            assert height.grad.abs().sum()>0
        assert mask.grad is None and cnn.grad is None
        assert all(torch.isfinite(p.grad).all() for p in bridge.parameters() if p.grad is not None)
        # Cached CNN features can be BF16 while the main height feature is FP16.
        if torch.cuda.is_available():
            bridge=bridge.cuda()
            with torch.cuda.amp.autocast():
                value,stats=bridge(height.detach().cuda().half(),[x.detach().cuda().bfloat16() for x in pyramid],
                    context.detach().cuda().bfloat16(),mask.detach().cuda().half(),cnn.detach().cuda().bfloat16())
                loss=value.float().square().mean()
            loss.backward()
            assert value.dtype==torch.float16 and value.shape==height.shape and torch.isfinite(value).all()
    old=GeometryHeightFusion(32,16,16)
    local=GeometryHeightFusion(32,16,16,mode='asymmetric-local')
    assert sum(p.numel() for p in old.parameters())==sum(p.numel() for p in local.parameters())
    assert sum(isinstance(m,ChannelNorm2d) for m in local.modules())==8
    assert not any(isinstance(m,torch.nn.GroupNorm) for m in local.modules())
    assert local.state_dict()['local_conditioning'].dtype==torch.long
    assert local.state_dict()['local_conditioning'].item()==1
    for target,source in ((old,local),(local,old)):
        try: target.load_state_dict(source.state_dict(),strict=True)
        except RuntimeError as error: assert 'local_conditioning' in str(error)
        else: raise AssertionError('Different normalization modes silently strict-loaded')
    # Identical, already-aligned features with extra distant background. This
    # checks the bridge itself, not CNN/attention/resize invariance of the model.
    with torch.no_grad():
        local.output.weight.normal_(std=.1)
        old.load_state_dict({k:v for k,v in local.state_dict().items() if k!='local_conditioning'},strict=True)
        inputs=[torch.randn(1,c,13,17) for c in (32,16,16,16,16,16,1,2)]
        padded=[]
        for x in inputs:
            y=torch.randn(1,x.shape[1],25,31)*8+7
            y[...,5:18,6:23]=x
            padded.append(y)
        def output(model,values):
            return model(values[0],values[1:5],values[5],values[6],values[7])[0]
        small,big=output(local,inputs),output(local,padded)[...,5:18,6:23]
        torch.testing.assert_close(small[...,2:-2,2:-2],big[...,2:-2,2:-2],rtol=1e-5,atol=2e-5)
        a,b=output(old,inputs),output(old,padded)[...,5:18,6:23]
        assert (a[...,2:-2,2:-2]-b[...,2:-2,2:-2]).abs().mean()>1e-4
    print('PASS: both gradient modes, trainable adapters, identity start, mixed precision, strict mode marker, and bridge-local distant-background test')


if __name__=='__main__':
    main()
