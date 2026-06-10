"""Validate every registry variant against its recovered checkpoint."""
import sys, torch
import os; REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
from models.registry import VARIANTS, load_model, DEFAULT_CHECKPOINT_ROOT

torch.manual_seed(0)
x = torch.randn(8, 15, 5)

for name, v in VARIANTS.items():
    if not v.available:
        print(f"{name:24s} SKIP (checkpoint not recovered)")
        continue
    try:
        sd = torch.load(f"{DEFAULT_CHECKPOINT_ROOT}/{v.checkpoint}", map_location="cpu")
        emb = tuple(sd["type_embedding.weight"].shape)  # (num_types, embedding_size)
        kw = v.network_kwargs()
        mismatch = ""
        if emb != (kw["num_particle_types"], kw["embedding_size"]):
            mismatch = f"  ⚠️ ckpt type_embedding {emb} != registry ({kw['num_particle_types']},{kw['embedding_size']})"
        model, hooks = load_model(name)
        n = sum(p.numel() for p in model.parameters())
        types = torch.randint(0, kw["num_particle_types"] - 1, (8, 15))
        types[:, -4:] = kw["num_particle_types"] - 1
        feat = x[..., : 4 + ("tag" in kw["feature_set"])]
        with torch.no_grad():
            out_hooked = model(feat, types)
        bn_note = ""
        if model.bottleneck_attention is not None:
            for h in hooks:
                h.remove()
            with torch.no_grad():
                out_raw = model(feat, types)
            delta = (out_hooked - out_raw).abs().max().item()
            bn_note = f"  bottleneck-hook effect: max|Δlogit|={delta:.3f}" + (
                "  ⚠️ HOOK HAS NO EFFECT" if delta < 1e-6 else " ✓"
            )
        else:
            for h in hooks:
                h.remove()
        print(f"{name:24s} OK  {n:>9,} params  out{tuple(out_hooked.shape)}{mismatch}{bn_note}")
    except Exception as e:
        print(f"{name:24s} FAIL  {type(e).__name__}: {str(e)[:140]}")
