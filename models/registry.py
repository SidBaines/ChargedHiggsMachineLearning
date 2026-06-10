"""Registry of trained low-level reconstruction model variants.

Replaces the nine `if 0:` config blocks in RunLowLevelInterp.py (lines ~174-361) with a
single source of truth. Each entry records the TestNetwork constructor kwargs and the
checkpoint (as referenced in that script) for a variant trained on the HEP PC; the
checkpoints were recovered to an external drive on 2026-06-10 (see
docs/logs/2026-06-10_planning-and-data-recovery.md).

All variants were trained with the OLD object-type encoding: 5 types + padding token 5
(N_CTX = 6; no separate Xbb-tagged large-R-jet type) — verified empirically from the
checkpoints' type_embedding shapes.

IMPORTANT: for variants with bottleneck_attention set, TestNetwork.forward() does NOT
apply the bottleneck — it must be applied via a forward hook (see TestNetwork docstring).
`load_model()` registers that hook for you; if you load manually, you must do it yourself
or the model will not compute what it computed in training.
"""
from dataclasses import dataclass, field
from typing import Optional

import torch

from models.models import TestNetwork

# Where the recovered `output/` tree lives. Override per-call in load_model if needed.
DEFAULT_CHECKPOINT_ROOT = (
    "/Volumes/Seagate/heppc_recovered/Code/ChargedHiggs_ExperimentalML/output"
)

N_CTX = 6  # 5 object types + padding=5 (old encoding; all registry variants use this)

# kwargs shared by every variant below; entries override what differs.
_BASE_KWARGS = dict(
    hidden_dim_attn=None,
    feature_set=["phi", "eta", "pt", "m", "tag"],
    bottleneck_attention=None,
    include_mlp=True,
    num_heads=4,
    embedding_size=N_CTX,
    num_classes=3,
    use_lorentz_invariant_features=True,
    dropout_p=0.0,
    num_particle_types=N_CTX,
    num_object_net_layers=1,
    is_layer_norm=False,
)


@dataclass
class ModelVariant:
    name: str
    description: str
    checkpoint: str  # path relative to the checkpoint root
    kwargs: dict = field(default_factory=dict)  # overrides on top of _BASE_KWARGS
    available: bool = True  # False if the checkpoint was not recovered

    def network_kwargs(self) -> dict:
        return {**_BASE_KWARGS, **self.kwargs}


VARIANTS = {
    v.name: v
    for v in [
        ModelVariant(
            name="lowdim-attnproj2",
            description="Low-dimensional attention test: projection to dim 2 inside "
            "attention, 2 heads, no MLP. No interpretability penalty.",
            checkpoint="20250429-104046_TrainingOutput/models/Nplits2_ValIdx0/chkpt19_109660.pth",
            kwargs=dict(hidden_dim_attn=2, hidden_dim=100, num_attention_blocks=4,
                        include_mlp=False, hidden_dim_mlp=1, num_heads=2,
                        embedding_size=10),
        ),
        ModelVariant(
            name="bn1-nomlp-notag-d300",
            description="Bottleneck-1 attention, no MLP, no tag feature, layer norm.",
            checkpoint="20250413-234444_TrainingOutput/models/Nplits2_ValIdx0/chkpt11_65796.pth",
            kwargs=dict(hidden_dim=300, feature_set=["phi", "eta", "pt", "m"],
                        bottleneck_attention=1, include_mlp=False, hidden_dim_mlp=1,
                        num_attention_blocks=3, embedding_size=10, is_layer_norm=True),
        ),
        ModelVariant(
            name="plain-d152-ln",
            description="Standard transformer with MLP and layer norm; no penalty. "
            "CHECKPOINT NOT RECOVERED (run dir absent on heppc, 2026-06-10).",
            checkpoint="20250328-113053_TrainingOutput/models/Nplits2_ValIdx0/chkpt24_137075.pth",
            kwargs=dict(hidden_dim=152, num_attention_blocks=3, hidden_dim_mlp=400,
                        embedding_size=10, is_layer_norm=True),
            available=False,
        ),
        ModelVariant(
            name="ent2-d152",
            description="Entropy penalty encouraging attention to TWO particles "
            "(target_entropy=log 2); no bottleneck.",
            checkpoint="20250510-123406_TrainingOutput/models/Nplits2_ValIdx0/chkpt29_164490.pth",
            kwargs=dict(hidden_dim=152, num_attention_blocks=3, hidden_dim_mlp=400),
        ),
        ModelVariant(
            name="plain-d152",
            description="UNCONSTRAINED BASELINE: same architecture as the thesis model "
            "but no entropy penalty and no bottleneck.",
            checkpoint="20250510-123629_TrainingOutput/models/Nplits2_ValIdx0/chkpt29_164490.pth",
            kwargs=dict(hidden_dim=152, num_attention_blocks=3, hidden_dim_mlp=400),
        ),
        ModelVariant(
            name="ent1-d152",
            description="Entropy penalty encouraging attention to a SINGLE particle "
            "(target_entropy=0); no bottleneck.",
            checkpoint="20250512-093602_TrainingOutput/models/Nplits2_ValIdx0/chkpt9_54830.pth",
            kwargs=dict(hidden_dim=152, num_attention_blocks=3, hidden_dim_mlp=400),
        ),
        ModelVariant(
            name="ent1-bn1-d152-6blk",
            description="Single-particle entropy penalty + bottleneck-1, 6 attention "
            "blocks (deep variant).",
            checkpoint="20250512-093806_TrainingOutput/models/Nplits2_ValIdx0/chkpt9_54830.pth",
            kwargs=dict(hidden_dim=152, bottleneck_attention=1, num_attention_blocks=6,
                        hidden_dim_mlp=400),
        ),
        ModelVariant(
            name="thesis-ent1-bn1-d152",
            description="THE THESIS MODEL (plots of 2025-07-09): single-particle entropy "
            "penalty + bottleneck-1, 3 blocks. RunLowLevelInterp.py:319-340.",
            checkpoint="20250512-093728_TrainingOutput/models/Nplits2_ValIdx0/chkpt29_164490.pth",
            kwargs=dict(hidden_dim=152, bottleneck_attention=1, num_attention_blocks=3,
                        hidden_dim_mlp=400),
        ),
        ModelVariant(
            name="ent1-d20-2blk",
            description="Small proto-organism (last pre-pause run, 2025-07-09): d=20, "
            "2 blocks, single-particle entropy penalty, no bottleneck. ~20k params.",
            checkpoint="20250709-095246_TrainingOutput/models/Nplits2_ValIdx0/chkpt9_54830.pth",
            kwargs=dict(hidden_dim=20, num_attention_blocks=2, hidden_dim_mlp=200),
        ),
    ]
}


def load_model(
    name: str,
    checkpoint_root: str = DEFAULT_CHECKPOINT_ROOT,
    device: str = "cpu",
    register_bottleneck_hook: bool = True,
    checkpoint_override: Optional[str] = None,
):
    """Build a registry variant, load its checkpoint, and (for bottleneck models)
    register the forward hook that actually applies the bottleneck.

    Returns (model, hook_handles). Keep hook_handles alive for as long as you use the
    model; call h.remove() on each to detach.
    """
    variant = VARIANTS[name]
    if not variant.available and checkpoint_override is None:
        raise FileNotFoundError(
            f"Variant '{name}' has no recovered checkpoint: {variant.description}"
        )
    model = TestNetwork(**variant.network_kwargs()).to(device)
    ckpt = checkpoint_override or f"{checkpoint_root}/{variant.checkpoint}"
    state_dict = torch.load(ckpt, map_location=torch.device(device))
    model.load_state_dict(state_dict)  # strict

    hook_handles = []
    if model.bottleneck_attention is not None and register_bottleneck_hook:
        # Deferred import: interp pulls in heavier deps.
        from interp.activations import ActivationCache, hook_attention_heads

        cache = ActivationCache()
        fwd_hooks = hook_attention_heads(
            model, cache, detach=True, SINGLE_ATTENTION=False,
            bottleneck_attention_output=model.bottleneck_attention,
        )
        for module, hook_fn in fwd_hooks:
            hook_handles.append(module.register_forward_hook(hook_fn, with_kwargs=True))
    model.eval()
    return model, hook_handles
