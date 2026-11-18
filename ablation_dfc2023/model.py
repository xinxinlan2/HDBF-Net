"""Plain ablation replacements; the full model's shared initialization is kept."""
import torch
from torch import nn

from semseg.models.hdbf_net import DinoHDBF, DinoSARBackbone
from ablation_dfc2023.protocol import FULL_REFERENCE

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
    return dict(protocol="dfc23-leave-one-out-1", case=case, reference_full=FULL_REFERENCE, **CASES[case])


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
    args.ablation_reference_settings = {key: getattr(args, key) for key in (
        "ttci", "slmhc", "local_structure", "geometry_height_fusion", "geometry_height_mode",
        "unirep_pretrained", "local_structure_weight", "structure_branch_weight", "slmhc_tail_weight")}
    args.ablation_config = spec
    args.ttci = spec["a"]
    args.slmhc = spec["c"]
    args.rgb_only = not spec["sar"]
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
    value = model_config.get("ablation_config")
    if not isinstance(value, dict):
        raise ValueError("Use this evaluator only for ablation checkpoints with ablation_config")
    if value.get("case") == "full" or value != configuration(value.get("case")):
        raise ValueError("Invalid ablation metadata")
    if "ablation_config" in train_args and train_args["ablation_config"] != value:
        raise ValueError("Conflicting ablation metadata in model_config and args")
    expected = dict(fusion_type="task-aware", task_split_decoder=True, variant="B4",
        local_structure=value["b"], unirep_structure=value["b"], geometry_height_fusion=value["b"],
        geometry_height_mode="asymmetric-local" if value["b"] else "joint-global",
        context_detail_height=True, building_affinity=True, use_hrbr=False, use_hcr=True, use_bihm=True,
        use_ttci=value["a"], use_slmhc=value["c"], rgb_only=not value["sar"],
        sar_channels=1, query_structure=False, geometry_interaction=False, segmentation_refinement=False)
    for key, expected_value in expected.items():
        actual = model_config.get(key)
        if actual != expected_value:
            raise ValueError(f"Ablation metadata contradicts effective {key}: {actual!r}")
    return value


def restore_model(checkpoint):
    config = checkpoint["model_config"]
    spec = read_configuration(config, checkpoint.get("args", {}))
    state = checkpoint["model_state_dict"]
    marker = state.get("ablation_marker")
    if marker is None or marker.item() != list(CASES).index(spec["case"]):
        raise ValueError("Checkpoint ablation marker does not match its case")
    decoder_dim = config["decoder_dim"]
    model = AblationDinoHDBF(
        ablation_case=spec["case"], fusion_type="task-aware", task_split_decoder=True,
        gradient_checkpointing=False, sar_channels=1, variant="B4", decoder_dim=decoder_dim,
        task_channels=config["task_channels"], use_hrbr=False, use_hcr=True,
        use_ttci=spec["a"], use_slmhc=spec["c"], context_detail_height=True, building_affinity=True,
    )
    if spec["b"]:
        from hdbf.unirep_structure import UniRepStructure
        from hdbf.geometry_height import GeometryHeightFusion
        model.unirep_structure = UniRepStructure(model.backbone.channels, decoder_dim, local_structure=True)
        model.height_geometry_fusion = GeometryHeightFusion(decoder_dim, mode=config["geometry_height_mode"])
    model.load_state_dict(state, strict=True)
    return model
