"""Plain ablation replacements; the full model's shared initialization is kept."""
import torch
from torch import nn

from semseg.models.hdbf_net import DinoHDBF, DinoSARBackbone
from ablation_spacenet6.protocol import FULL_REFERENCE, FIXED_RECIPE

CASES = {
    "full": dict(a=True, b=True, c=True, sar=True),
    "wo_a": dict(a=False, b=True, c=True, sar=True),
    "wo_b": dict(a=True, b=False, c=True, sar=True),
    "wo_c": dict(a=True, b=True, c=False, sar=True),
    "rgb_only": dict(a=True, b=True, c=True, sar=False),
}


def configuration(case):
    if case not in CASES:
        raise ValueError(f"Unknown ablation case: {case}")
    return dict(protocol="sn6-leave-one-out-1", case=case, reference_full=FULL_REFERENCE, **CASES[case])


class ConcatenateProjection(nn.Module):
    """No alignment, attention, routing, context pooling or interaction block."""
    def __init__(self, channels):
        super().__init__()
        self.projection = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, rgb, sar):
        value = self.projection(torch.cat((rgb, sar), dim=1))
        return value, value


class RGBBackbone(nn.Module):
    channels = DinoSARBackbone.channels
    tap_layers = DinoSARBackbone.tap_layers
    rgb_pyramid = DinoSARBackbone.rgb_pyramid

    def __init__(self, backbone):
        super().__init__()
        self.dino = backbone.dino
        self.rgb_projections = backbone.rgb_projections
        self.gradient_checkpointing = backbone.gradient_checkpointing
        self.fusion_type = "rgb-only"

    def forward(self, inputs):
        features = self.rgb_pyramid(inputs[0])
        return dict(structure=features, height=features)


class AblationDinoHDBF(DinoHDBF):
    def __init__(self, ablation_case="full", **kwargs):
        self.ablation = configuration(ablation_case)
        spec = self.ablation
        if kwargs.get("use_ttci", True) != spec["a"] or kwargs.get("use_slmhc", True) != spec["c"]:
            raise ValueError("Checkpoint TTCI/SLMHC flags disagree with ablation case")
        # Construct once as the full model so retained heads/features have the
        # same random initialization; remove rather than train unused modules.
        kwargs["use_ttci"] = True
        kwargs["use_slmhc"] = True
        super().__init__(**kwargs)
        if not spec["a"]:
            with torch.random.fork_rng(devices=[]):
                self.backbone.task_fusions = nn.ModuleList(ConcatenateProjection(c) for c in self.backbone.channels)
            self.use_ttci = False
            for name in ("mask_adapter", "contour_adapter", "height_adapter", "task_interaction"):
                delattr(self, name)
        if not spec["c"]:
            self.use_slmhc = False
            del self.height_calibrator
        if not spec["sar"]:
            self.backbone = RGBBackbone(self.backbone)
            self.rgb_only = True
        if ablation_case != "full":
            self.register_buffer("ablation_marker", torch.tensor(list(CASES).index(ablation_case), dtype=torch.long))

    def init_sar_pretrained(self, path):
        if not self.ablation["sar"]:
            raise ValueError("RGB-only model has no SAR encoder to initialize")
        return super().init_sar_pretrained(path)


def apply_case(args):
    spec = configuration(args.ablation_case)
    reference = dict(FIXED_RECIPE, ttci=True, slmhc=True, local_structure=True,
        geometry_height_fusion=True, geometry_height_mode="asymmetric-local", slmhc_tail_weight=.05,
        structure_branch_weight=.2, local_structure_weight=.2)
    for key, expected in reference.items():
        if getattr(args, key, None) != expected:
            raise ValueError(f"Frozen full-model recipe requires {key}={expected!r}; select removals with --ablation-case only")
    if not args.unirep_pretrained or not args.pretrained or not args.sar_pretrained:
        raise ValueError("Reference pretrained paths must be supplied before case selection")
    args.ablation_reference_settings = {key: getattr(args, key) for key in (
        "ttci", "slmhc", "local_structure", "geometry_height_fusion", "geometry_height_mode",
        "unirep_pretrained", "local_structure_weight", "structure_branch_weight", "slmhc_tail_weight")}
    args.ablation_config = spec
    args.ttci = spec["a"]
    args.slmhc = spec["c"]
    if not spec["b"]:
        args.local_structure = False
        args.unirep_pretrained = None
        args.geometry_height_fusion = False
        args.geometry_height_mode = "joint-global"
        args.local_structure_weight = 0.
        args.structure_branch_weight = 0.
    if not spec["c"]:
        args.slmhc_tail_weight = 0.
    return args


def read_configuration(model_config, train_args):
    value = model_config.get("ablation_config", train_args.get("ablation_config"))
    if value is None:
        raise ValueError("Use this evaluator only for ablation checkpoints with ablation_config")
    if value != configuration(value.get("case")):
        raise ValueError("Invalid ablation metadata")
    if "ablation_config" in train_args and train_args["ablation_config"] != value:
        raise ValueError("Conflicting ablation metadata in model_config and args")
    expected = dict(encoder="dinov2", fusion_type="task-aware", task_split_decoder=True,
        local_structure=value["b"], unirep_structure=value["b"], geometry_height_fusion=value["b"],
        context_detail_height=True, building_affinity=True, use_hrbr=False, use_hcr=True, use_bihm=True,
        use_ttci=value["a"], use_slmhc=value["c"])
    aliases = {"use_hrbr": "hrbr", "use_hcr": "hcr", "use_bihm": "bihm", "use_ttci": "ttci", "use_slmhc": "slmhc"}
    for key, expected_value in expected.items():
        if key == "unirep_structure":
            actual = model_config.get(key, bool(train_args.get("unirep_pretrained")))
        else:
            actual = model_config.get(key, train_args.get(aliases.get(key, key)))
        if actual != expected_value:
            raise ValueError(f"Ablation metadata contradicts effective {key}: {actual!r}")
    return value
