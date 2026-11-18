"""UniRepLKNet-S ADE20K feature transfer for HDBF structure prediction."""

import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule
from vendor.unireplknet.unireplknet import UniRepLKNet

WEIGHT_NAME = "upernet_unireplknet_s_in22k_512_160k_ade20k_miou51.86.pth"
WEIGHT_SHA256 = "67efb2494f12e2bccaa19282b8d2eb0d466fe5755584e5fb9903cd188ebc0e64"


def resize(x, size):
    y = F.interpolate(x.float(), size=size, mode="bilinear", align_corners=False)
    return y.to(x.dtype)


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.GELU()
    )


class LocalBackbone(UniRepLKNet):

    def init_weights(self):
        pass


class UPerFeatures(nn.Module):
    """MMSeg UPerNet feature layers with checkpoint-compatible names.

    Architecture: OpenMMLab MMSegmentation v0.27.0 UPerHead (Apache-2.0).
    The ADE20K classifier and auxiliary classifier are deliberately excluded.
    """

    def __init__(self):
        super().__init__()
        dims = (96, 192, 384, 768)

        def conv(cin, cout, kernel, inplace=True):
            return ConvModule(
                cin,
                cout,
                kernel,
                padding=kernel // 2,
                norm_cfg=dict(type="BN", requires_grad=True),
                act_cfg=dict(type="ReLU"),
                inplace=inplace,
            )

        self.psp_modules = nn.ModuleList(
            (
                nn.Sequential(nn.AdaptiveAvgPool2d(s), conv(768, 512, 1))
                for s in (1, 2, 3, 6)
            )
        )
        self.bottleneck = conv(768 + 512 * 4, 512, 3)
        self.lateral_convs = nn.ModuleList((conv(c, 512, 1, False) for c in dims[:-1]))
        self.fpn_convs = nn.ModuleList((conv(512, 512, 3, False) for _ in dims[:-1]))
        self.fpn_bottleneck = conv(512 * 4, 512, 3)

    def forward(self, features):
        deep = features[-1]
        ppm = [deep] + [resize(m(deep), deep.shape[-2:]) for m in self.psp_modules]
        laterals = [m(x) for (m, x) in zip(self.lateral_convs, features)]
        laterals.append(self.bottleneck(torch.cat(ppm, 1)))
        for i in (3, 2, 1):
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i], laterals[i - 1].shape[-2:]
            )
        outputs = [m(x) for (m, x) in zip(self.fpn_convs, laterals)]
        outputs.append(laterals[-1])
        return self.fpn_bottleneck(
            torch.cat([resize(x, outputs[0].shape[-2:]) for x in outputs], 1)
        )


class ScaleFusion(nn.Module):

    def __init__(self, cnn_channels, semantic_channels, width=64):
        super().__init__()
        self.cnn = block(cnn_channels, width)
        self.semantic = block(semantic_channels, width)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width, width // 4, 1),
            nn.GELU(),
            nn.Conv2d(width // 4, width, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(width * 2, 1, 3, padding=1), nn.Sigmoid()
        )
        self.fuse = block(width * 3, width)

    def forward(self, cnn, semantic):
        (c, s) = (self.cnn(cnn), self.semantic(semantic))
        s = resize(s, c.shape[-2:])
        c = c * (0.5 + self.channel_gate(c))
        s = s * (0.5 + self.spatial_gate(torch.cat((c, s), 1)))
        return self.fuse(torch.cat((c, s, c * s), 1))


class UniRepStructure(nn.Module):

    def __init__(
        self,
        semantic_channels,
        decoded_channels=512,
        query_structure=False,
        local_structure=False,
    ):
        super().__init__()
        if query_structure and local_structure:
            raise ValueError(
                "Local and query structure decoders are mutually exclusive"
            )
        self.local_structure = bool(local_structure)
        width = 192 if self.local_structure else 64
        self.cnn = LocalBackbone(
            depths=(3, 3, 27, 3),
            dims=(96, 192, 384, 768),
            num_classes=None,
            init_cfg=dict(type="LocalVerified"),
            with_cp=False,
            drop_path_rate=0.1,
            attempt_use_lk_impl=False,
            use_sync_bn=False,
        )
        self.ade_decoder = UPerFeatures()
        self.scale_fusions = nn.ModuleList(
            (
                ScaleFusion(c, s, width)
                for (c, s) in zip((96, 192, 384, 768), semantic_channels)
            )
        )
        self.topdown = nn.ModuleList((block(width * 2, width) for _ in range(3)))
        self.cnn_projection = block(512, width)
        self.semantic_projection = block(decoded_channels, width)
        self.joint = block(width * 3, width)
        self.rgb_detail = nn.Sequential(block(3, 16), block(16, 16))
        self.cnn_head = nn.Conv2d(width, 2, 1)
        self.region_head = nn.Sequential(block(64 + 16, 32), nn.Conv2d(32, 1, 1))
        self.contour_head = nn.Sequential(block(64 + 16 + 16, 32), nn.Conv2d(32, 1, 1))
        self.edge_context = block(96, 16)
        self.query_structure = bool(query_structure)
        if self.local_structure:
            from hdbf.local_structure import LocalStructureDecoder

            self.joint = nn.Sequential(self.joint, nn.Dropout2d(0.1))
            self.edge_context = block(96, 32)
            self.local_decoder = LocalStructureDecoder(width)
            del self.region_head, self.contour_head
        if self.query_structure:
            raise ValueError(
                "Historical architecture is not included in the final release"
            )

    def load_pretrained(self, path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != WEIGHT_SHA256:
            raise ValueError(
                f"Wrong/corrupt UniRepLKNet checkpoint: SHA256={digest.hexdigest()}"
            )
        obj = torch.load(path, map_location="cpu")
        source = obj.get("state_dict", obj.get("model", obj))
        source = {k.removeprefix("module."): v for (k, v) in source.items()}
        counts = {}
        for name, prefix in (("cnn", "backbone."), ("ade_decoder", "decode_head.")):
            module = getattr(self, name)
            target = module.state_dict()
            state = {}
            for key, value in target.items():
                skey = prefix + key
                if skey not in source or source[skey].shape != value.shape:
                    raise ValueError(
                        f"Incomplete/wrong UniRepLKNet-S ADE20K tensor: {skey}"
                    )
                if not torch.isfinite(source[skey]).all():
                    raise ValueError(f"Nonfinite pretrained tensor: {skey}")
                if skey.endswith("running_var") and (source[skey] < 0).any():
                    raise ValueError(f"Negative pretrained variance: {skey}")
                state[key] = source[skey]
            module.load_state_dict(state, strict=True)
            counts[name] = len(state)
        print(
            f"UniRepLKNet-S strict encoder: {counts['cnn']} tensors; strict ADE20K UPer feature decoder: {counts['ade_decoder']} tensors"
        )
        print(
            "Excluded ADE20K 150-class/auxiliary classifiers; binary structure heads train from scratch."
        )
        return dict(sha256=digest.hexdigest(), file=Path(path).name, loaded=counts)

    def forward(
        self,
        rgb,
        structure,
        decoded,
        coarse_mask,
        coarse_contour,
        return_features=False,
        height=None,
    ):
        enabled = (
            rgb.is_cuda
            and torch.is_autocast_enabled()
            and torch.cuda.is_bf16_supported()
        )
        with torch.autocast(
            device_type=rgb.device.type, enabled=enabled, dtype=torch.bfloat16
        ):
            return self._forward(
                rgb.float(),
                [x.float() for x in structure],
                decoded.float(),
                coarse_mask.float(),
                coarse_contour.float(),
                return_features,
                height,
            )

    def _forward(
        self,
        rgb,
        structure,
        decoded,
        coarse_mask,
        coarse_contour,
        return_features=False,
        height=None,
    ):
        encoded = self._encode(rgb, structure, decoded)
        return self._decode(
            encoded, coarse_mask, coarse_contour, return_features, height
        )

    def encode_geometry(self, rgb, structure, decoded):
        enabled = (
            rgb.is_cuda
            and torch.is_autocast_enabled()
            and torch.cuda.is_bf16_supported()
        )
        with torch.autocast(
            device_type=rgb.device.type, enabled=enabled, dtype=torch.bfloat16
        ):
            return self._encode(
                rgb.float(), [x.float() for x in structure], decoded.float()
            )

    def decode_geometry(
        self, encoded, coarse_mask, coarse_contour, height=None, return_features=False
    ):
        joint = encoded["joint"]
        enabled = (
            joint.is_cuda
            and torch.cuda.is_bf16_supported()
            and (torch.is_autocast_enabled() or joint.dtype == torch.bfloat16)
        )
        with torch.autocast(
            device_type=joint.device.type, enabled=enabled, dtype=torch.bfloat16
        ):
            return self._decode(
                encoded,
                coarse_mask.float(),
                coarse_contour.float(),
                return_features,
                height,
            )

    def _encode(self, rgb, structure, decoded):
        native = rgb.shape[-2:]
        (h, w) = native
        padded = F.pad(rgb, (0, -w % 32, 0, -h % 32), mode="replicate")
        cnn_features = self.cnn(padded)
        cnn_features = [
            x[..., : (h + stride - 1) // stride, : (w + stride - 1) // stride]
            for (x, stride) in zip(cnn_features, (4, 8, 16, 32))
        ]
        cnn_decoded = self.cnn_projection(self.ade_decoder(cnn_features))
        fused = [
            m(c, s) for (m, c, s) in zip(self.scale_fusions, cnn_features, structure)
        ]
        x = fused[-1]
        for i, layer in zip((2, 1, 0), self.topdown):
            x = layer(torch.cat((fused[i], resize(x, fused[i].shape[-2:])), 1))
        semantic = resize(self.semantic_projection(decoded), x.shape[-2:])
        joint = self.joint(
            torch.cat((x, resize(cnn_decoded, x.shape[-2:]), semantic), 1)
        )
        detail = self.rgb_detail(rgb)
        return dict(
            native=native,
            cnn_features=cnn_features,
            cnn_decoded=cnn_decoded,
            fused=fused,
            joint=joint,
            detail=detail,
            cnn_logits=self.cnn_head(cnn_decoded),
        )

    def _decode(
        self, encoded, coarse_mask, coarse_contour, return_features=False, height=None
    ):
        (native, joint, detail) = (encoded[k] for k in ("native", "joint", "detail"))
        (cnn_features, fused) = (encoded["cnn_features"], encoded["fused"])
        if self.local_structure:
            if height is None:
                raise ValueError(
                    "Local structure requires predicted height; update cmnext_mtl.py as well"
                )
            shallow = self.edge_context(cnn_features[0])
            seeds = resize(
                torch.cat((coarse_mask, coarse_contour), 1), joint.shape[-2:]
            ).float()
            output = self.local_decoder(joint, shallow, detail, seeds, height)
            cnn_logits = resize(encoded["cnn_logits"], native).float()
            output.update(
                unirep_cnn_mask=cnn_logits[:, :1],
                unirep_cnn_contour=cnn_logits[:, 1:2],
                unirep_semantic_mask=resize(coarse_mask, native),
                unirep_semantic_contour=resize(coarse_contour, native),
            )
            return output
        if self.query_structure:
            if height is None:
                raise ValueError(
                    "Region query structure requires predicted height, never GT height"
                )
            (coarse, pixels, predictions) = self.region_queries(
                joint, fused[1:], torch.cat((coarse_mask, coarse_contour), 1)
            )
            edge = resize(self.edge_context(cnn_features[0]), native).float()
            (logits, point_logits, coords) = self.point_refiner(
                coarse, pixels, detail.float(), edge, height
            )
            cnn_logits = resize(encoded["cnn_logits"], native).float()
            return dict(
                mask=logits[:, :1],
                contour=logits[:, 1:2],
                unirep_cnn_mask=cnn_logits[:, :1],
                unirep_cnn_contour=cnn_logits[:, 1:2],
                unirep_semantic_mask=resize(coarse_mask, native),
                unirep_semantic_contour=resize(coarse_contour, native),
                region_predictions=predictions,
                region_coarse_logits=coarse,
                region_point_logits=point_logits,
                region_point_coords=coords,
            )
        region = self.region_head(torch.cat((resize(joint, native), detail), 1)).float()
        edge = resize(self.edge_context(cnn_features[0]), native)
        contour = self.contour_head(
            torch.cat((resize(joint, native), detail, edge), 1)
        ).float()
        cnn_logits = resize(encoded["cnn_logits"], native).float()
        output = dict(
            mask=region,
            contour=contour,
            unirep_cnn_mask=cnn_logits[:, :1],
            unirep_cnn_contour=cnn_logits[:, 1:2],
            unirep_semantic_mask=resize(coarse_mask, native),
            unirep_semantic_contour=resize(coarse_contour, native),
        )
        if return_features:
            output.update(
                joint_feature=joint.float(),
                coarse_feature=fused[1].float(),
                rgb_detail=detail.float(),
                edge_feature=edge.float(),
            )
        return output
