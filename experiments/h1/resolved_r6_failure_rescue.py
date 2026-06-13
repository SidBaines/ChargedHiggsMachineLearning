"""Resolved qqbb W circuit, R6: failure rescue.

Tests two ways to rescue resolved failures/partials:

1. Upstream evidence: boost the two truth-W sjets' 3-momenta, preserving each sjet's
   invariant mass, direction, tag, and type.
2. Downstream readout: overwrite lep/nu bottleneck scalars with clean-exclusive means.

Baseline strata are fixed from the unmodified forward pass. Readouts are reported
inside each fixed stratum.

Usage:
  .venv/bin/python experiments/h1/resolved_r6_failure_rescue.py \
      --model thesis-ent1-bn1-d152 --skip 24 --batches 12
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

torch.set_num_threads(6)

from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset
from interp.activations import ActivationCache, hook_attention_heads
from models.registry import load_model


PAD, NU, LJ, SJ = 5, 2, 3, 4
W_CLASS = 2
NONE_CLASS = 0
ALL_HEADS = [(b, h) for b in range(3) for h in range(4)]
WIRE_SETS = {
    "b2h2_only": [(2, 2)],
    "b1h3_only": [(1, 3)],
    "b2h3_only": [(2, 3)],
    "b1h3_b2h2": [(1, 3), (2, 2)],
    "all_verdictish": [(1, 3), (2, 2), (2, 0), (2, 3)],
}


def pt(p4):
    return torch.sqrt(torch.clamp(p4[..., 0] ** 2 + p4[..., 1] ** 2, min=0.0))


def mass2(p4):
    return torch.clamp(p4[..., 3] ** 2 - (p4[..., :3] ** 2).sum(-1), min=0.0)


def qfmt(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "n/a"
    return f"med {np.median(x):+.3f}  p16/p84 {np.percentile(x,16):+.3f}/{np.percentile(x,84):+.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="thesis-ent1-bn1-d152")
    ap.add_argument("--skip", type=int, default=24)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--ckpt-override", default=None)
    args = ap.parse_args()

    data = os.path.join(REPO, "tmp_data_20250321v1_signal/")
    stds = np.ones(7)
    stds[:4] = 1e5
    dl = ProportionalMemoryMappedDataset(
        N_Real_Vars_In_File=7,
        N_Real_Vars_To_Return=7,
        memmap_paths={d: f"{data}dsid_{d}.memmap" for d in range(510115, 510125)},
        max_objs_in_memmap=15,
        batch_size=args.batch_size,
        device="cpu",
        is_train=False,
        n_splits=2,
        validation_split_idx=0,
        n_targets=3,
        shuffle=False,
        shuffle_batch=False,
        means=None,
        stds=stds,
        objs_to_output=15,
        signal_only=True,
        has_eventNumbers=True,
    )
    for _ in range(args.skip):
        next(dl)

    model, _ = load_model(
        args.model,
        checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
        register_bottleneck_hook=False,
        checkpoint_override=args.ckpt_override,
    )
    model.eval()
    cache = ActivationCache()
    hook_handles = [
        m.register_forward_hook(fn, with_kwargs=True)
        for m, fn in hook_attention_heads(
            model,
            cache,
            detach=True,
            SINGLE_ATTENTION=False,
            bottleneck_attention_output=model.bottleneck_attention,
        )
    ]
    attn_modules = [blk["self_attention"] for blk in model.attention_blocks]

    # Scalar override hook. Registered after cache/bottleneck hook.
    ov = {"active": False, "cfg": {}}

    def make_override(blk):
        def hook(module, args_, kwargs, output):
            if not ov["active"]:
                return output
            items = [(k, v) for k, v in ov["cfg"].items() if k[0] == blk]
            if not items:
                return output
            out, weights = output
            out = out.clone()
            bidx = torch.arange(out.shape[0])
            for (b, h), entries in items:
                w_up = model.attention_blocks[blk]["bottleneck_up"][h].weight[:, 0]
                for pos, target in entries:
                    cur = cache[f"block_{blk}_attention"]["bottleneck_activation"][bidx, h, pos, 0]
                    out[bidx, pos] = out[bidx, pos] + (target - cur).unsqueeze(-1) * w_up
            return out, weights

        return hook

    override_handles = [
        module.register_forward_hook(make_override(bi), with_kwargs=True)
        for bi, module in enumerate(attn_modules)
    ]

    xs, ts = [], []
    for _ in range(args.batches):
        b = next(dl)
        t = b["types"]
        ok = ((t == NU).sum(1) == 1) & (((t == 0) | (t == 1)).sum(1) == 1)
        xs.append(b["x"][ok].clone())
        ts.append(t[ok].clone())
    X_all = torch.cat(xs)
    T_all = torch.cat(ts)
    TRU_all = torch.round(X_all[..., -1]).long()
    ar_all = torch.arange(len(X_all))
    lpos_all = ((T_all == 0) | (T_all == 1)).float().argmax(1)
    lvbb = TRU_all[ar_all, lpos_all] == 3
    boosted = (~lvbb) & (((T_all == LJ) & (TRU_all == 2)).sum(1) == 1)
    resolved = (~lvbb) & (((T_all == SJ) & (TRU_all == 2)).sum(1) == 2)
    idx = torch.where(resolved)[0]
    X = X_all[idx].clone()
    T = T_all[idx].clone()
    TRU = TRU_all[idx].clone()
    N = len(X)
    ar = torch.arange(N)
    lpos = ((T == 0) | (T == 1)).float().argmax(1)
    npos = (T == NU).float().argmax(1)
    jet_mask = (T == LJ) | (T == SJ)

    wrows = []
    for i in range(N):
        rows = torch.where((T[i] == SJ) & (TRU[i] == 2))[0]
        order = torch.argsort(pt(X[i, rows, :4]), descending=True)
        wrows.append(rows[order])
    wrows = torch.stack(wrows)
    lead = wrows[:, 0]
    sub = wrows[:, 1]

    def fwd(x, override_cfg=None):
        outs = []
        with torch.no_grad():
            for c0 in range(0, len(x), args.batch_size):
                sl = slice(c0, min(c0 + args.batch_size, len(x)))
                if override_cfg is None:
                    ov["active"] = False
                    ov["cfg"] = {}
                else:
                    cfg = {}
                    for key, entries in override_cfg.items():
                        cfg[key] = [(pos[sl], target[sl]) for pos, target in entries]
                    ov["active"] = True
                    ov["cfg"] = cfg
                outs.append(model(x[sl, :, :5], T[sl]))
        ov["active"] = False
        ov["cfg"] = {}
        return torch.cat(outs)

    # Baseline forward and scalar cache collection. Need a separate pass for scalars.
    base_out_chunks = []
    scalar = defaultdict(list)
    with torch.no_grad():
        for c0 in range(0, N, args.batch_size):
            sl = slice(c0, min(c0 + args.batch_size, N))
            ov["active"] = False
            out = model(X[sl, :, :5], T[sl])
            base_out_chunks.append(out)
            bsz = len(out)
            bidx = torch.arange(bsz)
            for key in ALL_HEADS:
                blk, h = key
                bna = cache[f"block_{blk}_attention"]["bottleneck_activation"]
                scalar[("lep",) + key].append(bna[bidx, h, lpos[sl], 0])
                scalar[("nu",) + key].append(bna[bidx, h, npos[sl], 0])
    base_out = torch.cat(base_out_chunks)
    scalar = {k: torch.cat(v) for k, v in scalar.items()}

    def readouts(out):
        pred = out.argmax(-1)
        margin = out[..., W_CLASS] - out[..., NONE_CLASS]
        wsj_claims = (pred[ar.unsqueeze(1), wrows] == W_CLASS).sum(1)
        nonw_mask = jet_mask.clone()
        nonw_mask[ar, lead] = False
        nonw_mask[ar, sub] = False
        nonw_claim = ((pred == W_CLASS) & nonw_mask).any(1)
        lep_w = pred[ar, lpos] == W_CLASS
        nu_w = pred[ar, npos] == W_CLASS
        channel_correct = (~lep_w) & (~nu_w)
        clean = channel_correct & (wsj_claims == 2) & (~nonw_claim)
        return {
            "lep_w": lep_w.numpy(),
            "nu_w": nu_w.numpy(),
            "channel_correct": channel_correct.numpy(),
            "clean": clean.numpy(),
            "both_wsj": (wsj_claims == 2).numpy(),
            "any_wsj": (wsj_claims >= 1).numpy(),
            "nonw_claim": nonw_claim.numpy(),
            "lead_w": (pred[ar, lead] == W_CLASS).numpy(),
            "sub_w": (pred[ar, sub] == W_CLASS).numpy(),
            "wsj_sum_margin": margin[ar.unsqueeze(1), wrows].sum(1).numpy(),
            "lep_margin": margin[ar, lpos].numpy(),
        }

    base = readouts(base_out)
    clean_mask = torch.as_tensor(base["clean"], dtype=torch.bool)
    channel_correct = torch.as_tensor(base["channel_correct"], dtype=torch.bool)
    channel_error = ~channel_correct
    partial = channel_correct & (~clean_mask)
    partial_zero = partial & (~torch.as_tensor(base["any_wsj"], dtype=torch.bool))
    partial_one = partial & (torch.as_tensor(base["any_wsj"], dtype=torch.bool)) & (~torch.as_tensor(base["both_wsj"], dtype=torch.bool))
    partial_nonw = partial & torch.as_tensor(base["nonw_claim"], dtype=torch.bool)

    groups = [
        ("channel_error", channel_error.numpy()),
        ("partial_zero", partial_zero.numpy()),
        ("partial_one", partial_one.numpy()),
        ("partial_nonw", partial_nonw.numpy()),
        ("clean_exclusive", clean_mask.numpy()),
    ]

    def pair_stats(x):
        p4_pair = x[ar.unsqueeze(1), wrows, :4].sum(1)
        p_pt = (pt(p4_pair) * 100.0).numpy()
        p_m = torch.sqrt(mass2(p4_pair)).numpy() * 100.0
        p_sumpt = (pt(x[ar.unsqueeze(1), wrows, :4]) * 100.0).sum(1).numpy()
        ht = (pt(x[..., :4]) * 100.0 * jet_mask.float()).sum(1).numpy()
        return p_m, p_pt / np.maximum(ht, 1e-6), p_sumpt

    base_mjj, base_relpt, base_sumpt = pair_stats(X)

    clean_means = {}
    for tok in ("lep", "nu"):
        for key in ALL_HEADS:
            clean_means[(tok,) + key] = scalar[(tok,) + key][clean_mask].mean()

    def make_clean_override(heads):
        cfg = {}
        for key in heads:
            entries = []
            for tok, pos in (("lep", lpos), ("nu", npos)):
                target = torch.full((N,), float(clean_means[(tok,) + key]), dtype=torch.float32)
                entries.append((pos, target))
            cfg[key] = entries
        return cfg

    print(f"model={args.model}  skip={args.skip}  batches={args.batches}")
    print(
        f"events={len(X_all)}  boosted={boosted.sum().item()}  resolved={N}; "
        f"clean={clean_mask.sum().item()} partial={partial.sum().item()} error={channel_error.sum().item()}"
    )
    print("baseline groups:")
    print(f"{'group':18s} {'n':>5s} {'chanC':>7s} {'clean':>7s} {'anyW':>7s} {'bothW':>7s} {'nonW':>7s} {'relpt':>12s}")
    for gname, mask in groups:
        if mask.sum() == 0:
            continue
        print(
            f"{gname:18s} {mask.sum():5d} {base['channel_correct'][mask].mean():7.3f} "
            f"{base['clean'][mask].mean():7.3f} {base['any_wsj'][mask].mean():7.3f} "
            f"{base['both_wsj'][mask].mean():7.3f} {base['nonw_claim'][mask].mean():7.3f} "
            f"{np.median(base_relpt[mask]):12.3f}"
        )

    records = []

    def report(kind, label, post, x_mod=None):
        if x_mod is not None:
            mjj, relpt, sumpt = pair_stats(x_mod)
            delta_relpt = relpt - base_relpt
            delta_mjj = mjj - base_mjj
        else:
            delta_relpt = np.zeros(N)
            delta_mjj = np.zeros(N)
        print(f"\n=== {kind}: {label} ===")
        print(
            f"{'group':18s} {'n':>5s} {'dChanC':>8s} {'dClean':>8s} {'dAnyW':>8s} "
            f"{'dBothW':>8s} {'dNonW':>8s} {'med dSumM':>10s} {'med dLepM':>10s} {'med dRel':>9s} {'med dMjj':>9s}"
        )
        for gname, mask in groups:
            if mask.sum() == 0:
                continue
            rec = {
                "kind": kind,
                "label": label,
                "group": gname,
                "n": int(mask.sum()),
                "delta_channel_correct": post["channel_correct"][mask].mean() - base["channel_correct"][mask].mean(),
                "delta_clean": post["clean"][mask].mean() - base["clean"][mask].mean(),
                "delta_any_wsj": post["any_wsj"][mask].mean() - base["any_wsj"][mask].mean(),
                "delta_both_wsj": post["both_wsj"][mask].mean() - base["both_wsj"][mask].mean(),
                "delta_nonw_claim": post["nonw_claim"][mask].mean() - base["nonw_claim"][mask].mean(),
                "med_delta_wsj_sum_margin": np.median(post["wsj_sum_margin"][mask] - base["wsj_sum_margin"][mask]),
                "med_delta_lep_margin": np.median(post["lep_margin"][mask] - base["lep_margin"][mask]),
                "med_delta_relpt": np.median(delta_relpt[mask]),
                "med_delta_mjj": np.median(delta_mjj[mask]),
            }
            records.append(rec)
            print(
                f"{gname:18s} {rec['n']:5d} {rec['delta_channel_correct']:+8.3f} "
                f"{rec['delta_clean']:+8.3f} {rec['delta_any_wsj']:+8.3f} "
                f"{rec['delta_both_wsj']:+8.3f} {rec['delta_nonw_claim']:+8.3f} "
                f"{rec['med_delta_wsj_sum_margin']:+10.3f} {rec['med_delta_lep_margin']:+10.3f} "
                f"{rec['med_delta_relpt']:+9.3f} {rec['med_delta_mjj']:+9.1f}"
            )

    # 1. Pair-hardness boosts.
    m2_all = mass2(X[..., :4])
    for scale in (1.25, 1.5, 2.0):
        Xb = X.clone()
        for row in (lead, sub):
            p3 = Xb[ar, row, :3] * scale
            Xb[ar, row, :3] = p3
            Xb[ar, row, 3] = torch.sqrt((p3**2).sum(1) + m2_all[ar, row])
        post = readouts(fwd(Xb))
        report("pair_boost", f"scale={scale:.2f}", post, Xb)

    # 2. Clean-success scalar overwrites at lep and nu.
    for name, heads in WIRE_SETS.items():
        cfg = make_clean_override(heads)
        post = readouts(fwd(X, override_cfg=cfg))
        report("scalar_clean_overwrite", name, post, None)

    out_dir = os.path.join(REPO, "tmp_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"resolved_r6_failure_rescue_{args.model}_skip{args.skip}_b{args.batches}.csv")
    header = list(records[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for rec in records:
            f.write(",".join(str(rec[k]) for k in header) + "\n")
    print(f"\nsaved R6 rescue records: {out_path}")

    for h in override_handles + hook_handles:
        h.remove()


if __name__ == "__main__":
    main()
