"""Train a joint reconstruction/classification model on the local signal copy.

This forks ``train_organism.py`` for studies where one transformer is used for
both per-object reconstruction (none/H/W) and event-level background/signal
classification. Real background memmaps are not present in this checkout, so
``--pseudo-bkg-dsid`` can mark one signal DSID as pseudo-background for smoke
tests. That relabeling is only a plumbing check; the resulting classification
numbers are not physically meaningful.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

EXPERIMENTS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, EXPERIMENTS)
from joint_primitives import event_score_from_object_logits, weighted_roc_auc, best_asimov_z

REPO = os.path.dirname(EXPERIMENTS); sys.path.insert(0, REPO)
from models.models import TestNetwork
from interp.activations import ActivationCache, hook_attention_heads, hook_attention_weights_only
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset
from metrics.lowlevelrecometrics import HEPLossWithEntropy, HEPMetrics
from utils.utils import basic_lr_scheduler

p = argparse.ArgumentParser(); p.add_argument("--smoke", action="store_true")
p.add_argument("--device", default=None, help="override CONFIG device (cpu/mps)")
p.add_argument("--seed", type=int, default=None, help="override CONFIG seed")
p.add_argument("--epochs", type=int, default=None, help="override CONFIG num_epochs")
p.add_argument("--ckpt-every-steps", type=int, default=None,
               help="ALSO save a checkpoint every N optimizer steps (formation studies)")
p.add_argument("--no-wandb", action="store_true")
p.add_argument("--lr", type=float, default=None, help="override peak learning_rate")
p.add_argument("--lr-low", type=float, default=None, help="override learning_rate_low")
p.add_argument("--lr-schedule", choices=["log", "cosine"], default=None,
               help="decay shape (default: CONFIG learning_rate_log_decay -> cosine)")
p.add_argument("--warmup-mode", choices=["ramp", "legacy"], default=None,
               help="ramp = linear 0->peak over warmup_steps; legacy = constant 1e-3")
p.add_argument("--d-model", type=int, default=None)
p.add_argument("--blocks", type=int, default=None)
p.add_argument("--d-mlp", type=int, default=None)
p.add_argument("--bottleneck", type=int, default=None,
               help="per-head attention bottleneck dim (applied via hook at train time)")
p.add_argument("--entropy-weight", type=float, default=None,
               help="0 disables the entropy penalty (default: CONFIG 0)")
p.add_argument("--num-heads", type=int, default=None)
p.add_argument("--dropout", type=float, default=None)
p.add_argument("--weight-decay", type=float, default=None)
p.add_argument("--batch-size", type=int, default=None)
p.add_argument("--embedding-size", type=int, default=None)
p.add_argument("--no-mlp", action="store_true",
               help="attention-only (include_mlp=False)")
p.add_argument("--warmup-steps", type=int, default=None)
p.add_argument("--features", type=str, default=None,
               help="comma list from {phi,eta,pt,m,tag}; default keeps all 5")
p.add_argument("--object-net-layers", type=int, default=None)
p.add_argument("--layer-norm", action="store_true",
               help="LayerNorm in the object_net (NB: not in the attention blocks)")
p.add_argument("--keep-cats", type=str, default=None,
               help="comma list of reco categories to TRAIN on; eval still reports every category")
p.add_argument("--mode", choices=["single", "twohead"], default="twohead")
p.add_argument("--schedule", choices=["joint", "int", "intdetach", "seqfull", "seqfrozen"],
               default="joint", help="two-head training schedule")
p.add_argument("--lambda-event", type=float, default=1.0)
p.add_argument("--event-classes", type=int, default=3,
               help="3 = bkg/qqbb/lvbb; 2 = bkg/signal")
p.add_argument("--pseudo-bkg-dsid", type=int, default=None,
               help="signal DSID to relabel as pseudo-background for smoke tests")
p.add_argument("--seq-split", type=float, default=0.5,
               help="fraction of epochs in reco phase for seqfull/seqfrozen")
p.add_argument("--readout", choices=["softor", "max"], default="softor",
               help="single-mode event-score readout for metrics")
ARGS = p.parse_args()

CONFIG = dict(
    # --- provenance ---
    purpose="joint reco/event harness v1: object reconstruction plus event classification",
    data="20250321v1 signal-only (SSD copy), pseudo background optional for smoke only",
    # --- architecture ---
    d_model=20, num_blocks=2, num_heads=4, d_mlp=200, include_mlp=True,
    bottleneck_attention=None, d_attn=None, embedding_size=6, num_particle_types=6,
    feature_set=["phi", "eta", "pt", "m", "tag"], use_lorentz_invariant_features=True,
    num_object_net_layers=1, is_layer_norm=False, dropout_p=0.0, num_classes=3,
    add_event_head=True, num_event_classes=3,
    # --- training ---
    batch_size=4096, num_epochs=30, learning_rate=1e-3, learning_rate_low=5e-7,
    learning_rate_log_decay=False, warmup_steps=100, weight_decay=1e-6,
    entropy_loss=False, entropy_weight=0.0, target_entropy=0,
    # --- joint harness ---
    mode=ARGS.mode, schedule=ARGS.schedule, lambda_event=ARGS.lambda_event,
    event_classes=ARGS.event_classes, pseudo_bkg_dsid=ARGS.pseudo_bkg_dsid,
    seq_split=ARGS.seq_split, readout=ARGS.readout,
    # --- data conventions ---
    max_n_objs=15, n_real_vars_in_file=7, model_input_vars=5, scale_data_std=1e5,
    n_splits=2, validation_split_idx=0, has_eventNumbers=True, shuffle_objects=True,
    padding_sanitized=True, grad_clip=1.0,
    device="mps", seed=0, wandb=True,
    wandb_project="HEP-Transformers-TruthMatchingReco",
)
if ARGS.smoke:
    CONFIG.update(num_epochs=1, wandb=False)
if ARGS.device:
    CONFIG["device"] = ARGS.device
if ARGS.seed is not None:
    CONFIG["seed"] = ARGS.seed
if ARGS.epochs is not None:
    CONFIG["num_epochs"] = ARGS.epochs
if ARGS.no_wandb:
    CONFIG["wandb"] = False
if ARGS.lr is not None:
    CONFIG["learning_rate"] = ARGS.lr
if ARGS.lr_low is not None:
    CONFIG["learning_rate_low"] = ARGS.lr_low
CONFIG["lr_schedule"] = ARGS.lr_schedule or ("log" if CONFIG["learning_rate_log_decay"] else "cosine")
CONFIG["warmup_mode"] = ARGS.warmup_mode or "ramp"
if ARGS.d_model is not None: CONFIG["d_model"] = ARGS.d_model
if ARGS.blocks is not None: CONFIG["num_blocks"] = ARGS.blocks
if ARGS.d_mlp is not None: CONFIG["d_mlp"] = ARGS.d_mlp
if ARGS.bottleneck is not None: CONFIG["bottleneck_attention"] = ARGS.bottleneck
if ARGS.entropy_weight is not None:
    CONFIG["entropy_weight"] = ARGS.entropy_weight
    CONFIG["entropy_loss"] = ARGS.entropy_weight > 0
if ARGS.num_heads is not None: CONFIG["num_heads"] = ARGS.num_heads
if ARGS.dropout is not None: CONFIG["dropout_p"] = ARGS.dropout
if ARGS.weight_decay is not None: CONFIG["weight_decay"] = ARGS.weight_decay
if ARGS.batch_size is not None: CONFIG["batch_size"] = ARGS.batch_size
if ARGS.embedding_size is not None: CONFIG["embedding_size"] = ARGS.embedding_size
if ARGS.no_mlp: CONFIG["include_mlp"] = False
if ARGS.warmup_steps is not None: CONFIG["warmup_steps"] = ARGS.warmup_steps
if ARGS.features is not None:
    CONFIG["feature_set"] = [f.strip() for f in ARGS.features.split(",") if f.strip()]
if ARGS.object_net_layers is not None: CONFIG["num_object_net_layers"] = ARGS.object_net_layers
if ARGS.layer_norm: CONFIG["is_layer_norm"] = True
CONFIG["keep_cats"] = [int(c) for c in ARGS.keep_cats.split(",")] if ARGS.keep_cats else None
CONFIG["add_event_head"] = CONFIG["mode"] == "twohead"
CONFIG["num_event_classes"] = CONFIG["event_classes"]

torch.manual_seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"])
device = CONFIG["device"] if torch.backends.mps.is_available() else "cpu"

stamp = time.strftime("%Y%m%d-%H%M%S")
_ent = f"YesEnt{CONFIG['entropy_weight']:g}" if CONFIG["entropy_loss"] else "NoEnt"
_bn = f"Bn{CONFIG['bottleneck_attention']}" if CONFIG["bottleneck_attention"] else "NoBn"
run_name = (f"_{stamp}_LowLevel_JOINT2026_{CONFIG['mode']}_{CONFIG['schedule']}"
            f"_d{CONFIG['d_model']}b{CONFIG['num_blocks']}_{_ent}_{_bn}"
            f"_lr{CONFIG['learning_rate']:g}-{CONFIG['learning_rate_low']:g}"
            f"_{CONFIG['lr_schedule']}_{CONFIG['warmup_mode']}_s{CONFIG['seed']}")
run_name += f"_h{CONFIG['num_heads']}_ec{CONFIG['event_classes']}_le{CONFIG['lambda_event']:g}"
if CONFIG["pseudo_bkg_dsid"] is not None:
    run_name += f"_pb{CONFIG['pseudo_bkg_dsid']}"
if CONFIG["mode"] == "single":
    run_name += f"_{CONFIG['readout']}"
if not CONFIG["include_mlp"]:
    run_name += "_nomlp"
if CONFIG.get("keep_cats"):
    run_name += "_keep" + "".join(str(c) for c in CONFIG["keep_cats"])
OUT = os.path.join(REPO, "output", f"{stamp}_TrainingOutput")
MODELS_OUT = os.path.join(OUT, "models", f"Nplits{CONFIG['n_splits']}_ValIdx{CONFIG['validation_split_idx']}")
os.makedirs(MODELS_OUT, exist_ok=True)
with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=1)
print(f"run: {run_name}\nout: {OUT}\ndevice: {device}")

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
DSIDS = list(range(510115, 510125))
stds = np.ones(CONFIG["n_real_vars_in_file"]); stds[:4] = CONFIG["scale_data_std"]
mk = dict(N_Real_Vars_In_File=CONFIG["n_real_vars_in_file"],
          N_Real_Vars_To_Return=CONFIG["n_real_vars_in_file"],
          memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in DSIDS},
          max_objs_in_memmap=CONFIG["max_n_objs"], batch_size=CONFIG["batch_size"],
          device=device, n_splits=CONFIG["n_splits"],
          validation_split_idx=CONFIG["validation_split_idx"], n_targets=3,
          means=None, stds=stds, objs_to_output=CONFIG["max_n_objs"],
          signal_only=True, has_eventNumbers=CONFIG["has_eventNumbers"])
train_dl = ProportionalMemoryMappedDataset(is_train=True, shuffle=CONFIG["shuffle_objects"], shuffle_batch=True, **mk)
val_dl = ProportionalMemoryMappedDataset(is_train=False, shuffle=False, shuffle_batch=True, **mk)
print(f"train samples: {train_dl.get_total_samples()}  val samples: {val_dl.get_total_samples()}")

model = TestNetwork(
    hidden_dim_attn=CONFIG["d_attn"], hidden_dim=CONFIG["d_model"],
    feature_set=CONFIG["feature_set"], bottleneck_attention=CONFIG["bottleneck_attention"],
    include_mlp=CONFIG["include_mlp"], num_attention_blocks=CONFIG["num_blocks"],
    hidden_dim_mlp=CONFIG["d_mlp"], num_heads=CONFIG["num_heads"],
    embedding_size=CONFIG["embedding_size"], num_classes=CONFIG["num_classes"],
    use_lorentz_invariant_features=CONFIG["use_lorentz_invariant_features"],
    dropout_p=CONFIG["dropout_p"], num_particle_types=CONFIG["num_particle_types"],
    num_object_net_layers=CONFIG["num_object_net_layers"], is_layer_norm=CONFIG["is_layer_norm"],
    is_reconstruction_model=True, add_event_head=CONFIG["add_event_head"],
    num_event_classes=CONFIG["num_event_classes"],
).to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

# grad-attached attention cache for the entropy loss.
# Only needed when the entropy penalty is on or an attention bottleneck is applied
# via the hook (the joint default uses neither). When neither holds, skip the hook
# entirely -> model output is bit-identical without it. With a bottleneck the hook
# must rewrite the forward, so use the full hook_attention_heads; otherwise the
# lightweight weights-only hook suffices for the entropy term.
cache = ActivationCache()
handles = []
_need_hook = CONFIG["entropy_loss"] or (CONFIG["bottleneck_attention"] is not None)
if _need_hook:
    if CONFIG["bottleneck_attention"] is not None:
        hook_pairs = hook_attention_heads(model, cache, detach=False, SINGLE_ATTENTION=False,
                                          bottleneck_attention_output=CONFIG["bottleneck_attention"])
    else:
        hook_pairs = hook_attention_weights_only(model, cache)
    handles = [m.register_forward_hook(fn, with_kwargs=True) for m, fn in hook_pairs]

optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
criterion = HEPLossWithEntropy(entropy_loss=CONFIG["entropy_loss"], entropy_weight=CONFIG["entropy_weight"],
                               target_entropy=CONFIG["target_entropy"], is_categorical=True,
                               apply_correlation_penalty=False, alpha=1.0,
                               apply_valid_penalty=False, valid_penalty_weight=1.0)

if CONFIG["wandb"]:
    import wandb
    wandb.init(project=CONFIG["wandb_project"], name=run_name, config=CONFIG)

N_CTX, MAX_OBJS = CONFIG["num_particle_types"], CONFIG["max_n_objs"]
val_metrics = HEPMetrics(N_CTX - 1, MAX_OBJS, is_categorical=True, num_categories=3,
                         max_bkg_levels=[100, 200], max_buffer_len=int(val_dl.get_total_samples()),
                         total_weights_per_dsid=val_dl.weight_sums,
                         signal_acceptance_levels=[100, 500, 1000, 5000])

def sanitize_padding(x, types):
    """Padding objects are masked out, but still need finite benign inputs."""
    pad = (types == N_CTX - 1).unsqueeze(-1)
    safe = torch.zeros_like(x)
    safe[..., 0] = 1e-3; safe[..., 1] = 1e-3; safe[..., 3] = 2e-3
    return torch.where(pad, safe, x)

def preprocess_batch(b):
    x, y, train_wts, types = b["x"], b["y"], b["train_wts"], b["types"]
    dsids = b["dsids"]
    x = sanitize_padding(x, types)
    if CONFIG["pseudo_bkg_dsid"] is not None:
        is_bkg = dsids == CONFIG["pseudo_bkg_dsid"]
    else:
        is_bkg = y.argmax(-1) == 0
    reco_target = x[..., -1].clone()
    reco_target[is_bkg] = 0
    if CONFIG["event_classes"] == 3:
        event_target = y.argmax(-1)
        if CONFIG["pseudo_bkg_dsid"] is not None:
            event_target = event_target.clone()
            event_target[is_bkg] = 0
    else:
        event_target = (~is_bkg).long()
    return x, y, train_wts, types, dsids, is_bkg, reco_target, event_target

def forward_heads(x, types, detach_event):
    feats = model._backbone_features(x[..., :CONFIG["model_input_vars"]], types)
    reco_logits = model.classifier(feats)
    nonpad = (types != N_CTX - 1).unsqueeze(-1)
    pooled = (feats * nonpad).sum(1) / nonpad.sum(1).clamp(min=1)
    event_in = pooled.detach() if detach_event else pooled
    event_logits = model.event_classifier(event_in)
    return reco_logits, event_logits

def reco_loss_value(reco_logits, reco_target, types, reco_weight, x):
    loss = criterion(cache, reco_logits, reco_target, types, N_CTX - 1, MAX_OBJS,
                     reco_weight, False, x[..., :4], build_loss_dict=False)
    if isinstance(loss, tuple):
        loss, _ = loss
    return loss

def event_loss_value(event_logits, event_target, weights):
    eps = torch.finfo(weights.dtype).eps
    return (F.cross_entropy(event_logits, event_target, reduction="none") * weights).sum() / weights.sum().clamp(min=eps)

def set_learning_rate(step):
    hi, lo, wsteps = CONFIG["learning_rate"], CONFIG["learning_rate_low"], CONFIG["warmup_steps"]
    if CONFIG["warmup_mode"] == "legacy":
        lr = basic_lr_scheduler(step, hi, lo, num_lr_steps,
                                CONFIG["lr_schedule"] == "log",
                                warmup_steps=wsteps, warmup_rate=1e-3)
    else:
        if step < wsteps:
            lr = hi * (step + 1) / wsteps
        else:
            t = (step - wsteps) / max(num_lr_steps - wsteps, 1)
            if CONFIG["lr_schedule"] == "cosine":
                lr = lo + 0.5 * (hi - lo) * (1 + np.cos(np.pi * t))
            elif CONFIG["lr_schedule"] == "log":
                lr = hi * (lo / hi) ** t
            else:
                lr = hi - (hi - lo) * t
    for g in optimizer.param_groups: g["lr"] = lr
    return lr

nan_skips = 0
def backward_and_step(loss, epoch, bi):
    global nan_skips
    if not torch.isfinite(loss):
        nan_skips += 1
        print(f"  !! non-finite loss at ep {epoch} step {bi} - skipping batch ({nan_skips} skips)")
        if nan_skips > 20:
            raise RuntimeError("too many non-finite batches; aborting")
        optimizer.zero_grad(); return False
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
    if not torch.isfinite(gnorm):
        nan_skips += 1
        print(f"  !! non-finite grad-norm at ep {epoch} step {bi} - skipping ({nan_skips})")
        optimizer.zero_grad(); return False
    optimizer.step()
    return True

def freeze_for_event_only():
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("event_classifier.")

steps_per_epoch = len(train_dl)
num_lr_steps = CONFIG["num_epochs"] * steps_per_epoch
global_step = 0
SMOKE_STEPS = 30
seq_event_start = round(CONFIG["seq_split"] * CONFIG["num_epochs"])
seqfrozen_done = False
t0 = time.time()
for epoch in range(CONFIG["num_epochs"]):
    train_dl._reset_indices(); val_dl._reset_indices()
    model.train()
    ep_loss = ep_w = 0.0
    n_steps = SMOKE_STEPS if ARGS.smoke else steps_per_epoch
    for bi in range(n_steps):
        if bi % 10 == 0:
            lr = set_learning_rate(bi + epoch * steps_per_epoch)

        b = next(train_dl)
        x, y, w, types, dsids, is_bkg, reco_target, event_target = preprocess_batch(b)
        reco_keep = torch.ones_like(w, dtype=torch.bool)
        if CONFIG["keep_cats"] is not None:
            from utils.utils import check_category
            catb = check_category(types.cpu(), x[..., -1].cpu(), N_CTX - 1, use_torch=True).to(w.device)
            reco_keep = torch.zeros_like(w, dtype=torch.bool)
            for c in CONFIG["keep_cats"]:
                reco_keep |= (catb == c)

        optimizer.zero_grad()
        loss = None
        step_weight = w.sum()
        if CONFIG["mode"] == "single":
            single_weight = w * reco_keep.to(w.dtype)
            if single_weight.sum() == 0:
                continue
            reco_logits = model(x[..., :CONFIG["model_input_vars"]], types)
            loss = reco_loss_value(reco_logits, reco_target, types, single_weight, x)
            step_weight = single_weight.sum()
        else:
            reco_weight = w * (~is_bkg).to(w.dtype) * reco_keep.to(w.dtype)
            event_weight = w
            schedule_step = global_step
            if CONFIG["schedule"] == "joint":
                reco_logits, event_logits = forward_heads(x, types, False)
                event_loss = event_loss_value(event_logits, event_target, event_weight)
                if reco_weight.sum() == 0:
                    loss = CONFIG["lambda_event"] * event_loss
                else:
                    reco_loss = reco_loss_value(reco_logits, reco_target, types, reco_weight, x)
                    loss = reco_loss + CONFIG["lambda_event"] * event_loss
            elif CONFIG["schedule"] == "int":
                if schedule_step % 2 == 0:
                    if reco_weight.sum() == 0:
                        continue
                    reco_logits, _ = forward_heads(x, types, False)
                    loss = reco_loss_value(reco_logits, reco_target, types, reco_weight, x)
                    step_weight = reco_weight.sum()
                else:
                    _, event_logits = forward_heads(x, types, False)
                    loss = CONFIG["lambda_event"] * event_loss_value(event_logits, event_target, event_weight)
            elif CONFIG["schedule"] == "intdetach":
                if schedule_step % 2 == 0:
                    if reco_weight.sum() == 0:
                        continue
                    reco_logits, _ = forward_heads(x, types, False)
                    loss = reco_loss_value(reco_logits, reco_target, types, reco_weight, x)
                    step_weight = reco_weight.sum()
                else:
                    _, event_logits = forward_heads(x, types, True)
                    loss = CONFIG["lambda_event"] * event_loss_value(event_logits, event_target, event_weight)
            elif CONFIG["schedule"] in ("seqfull", "seqfrozen"):
                phase = "reco" if epoch < seq_event_start else "event"
                if phase == "reco":
                    if reco_weight.sum() == 0:
                        continue
                    reco_logits, _ = forward_heads(x, types, False)
                    loss = reco_loss_value(reco_logits, reco_target, types, reco_weight, x)
                    step_weight = reco_weight.sum()
                else:
                    if CONFIG["schedule"] == "seqfrozen" and not seqfrozen_done:
                        freeze_for_event_only()
                        seqfrozen_done = True
                    _, event_logits = forward_heads(x, types, False)
                    loss = CONFIG["lambda_event"] * event_loss_value(event_logits, event_target, event_weight)

        stepped = backward_and_step(loss, epoch, bi)
        if not stepped:
            continue
        global_step += 1
        if ARGS.ckpt_every_steps and global_step % ARGS.ckpt_every_steps == 0:
            torch.save(model.state_dict(),
                       os.path.join(MODELS_OUT, f"chkpt{epoch}_{global_step}.pth"))
        ep_loss += loss.item() * step_weight.item(); ep_w += step_weight.item()
        if bi % 50 == 0:
            print(f"  ep {epoch} step {bi}/{n_steps} loss {loss.item():.4f} ({(time.time()-t0):.0f}s)")
        if CONFIG["wandb"] and global_step % 20 == 0:
            wandb.log({"train/loss": loss.item(), "lr": lr, "epoch": epoch}, step=global_step)

    # ---- epoch-end validation (subset for speed; full val at final epoch) ----
    model.eval(); val_metrics.reset()
    score_chunks, label_chunks, weight_chunks = [], [], []
    n_val = len(val_dl) if epoch == CONFIG["num_epochs"] - 1 else min(30, len(val_dl))
    if ARGS.smoke: n_val = 3
    with torch.no_grad():
        for _ in range(n_val):
            vb = next(val_dl)
            vx, vy, vw, vtypes, vdsids, vis_bkg, vreco_target, vevent_target = preprocess_batch(vb)
            if CONFIG["mode"] == "single":
                vreco_logits = model(vx[..., :CONFIG["model_input_vars"]], vtypes)
                vscores = event_score_from_object_logits(vreco_logits, vtypes, N_CTX - 1, CONFIG["readout"])
            else:
                vreco_logits, vevent_logits = forward_heads(vx, vtypes, False)
                vscores = 1.0 - torch.softmax(vevent_logits, dim=-1)[:, 0]

            signal_mask = ~vis_bkg
            if signal_mask.any():
                val_metrics.update(vreco_logits[signal_mask].cpu(), vx[..., -1][signal_mask].cpu(),
                                   vb["MC_Wts"][signal_mask].cpu(), vdsids[signal_mask].cpu(),
                                   vtypes[signal_mask].cpu())
            score_chunks.append(vscores.detach().cpu())
            label_chunks.append(signal_mask.long().detach().cpu())
            # This local signal sample has signed MC weights; sklearn AUC needs nonnegative sample weights.
            weight_chunks.append(vb["MC_Wts"].abs().detach().cpu())

    rv = val_metrics.compute_and_log(1, "val", 0, 3, False, None,
                                     calc_all=(epoch == CONFIG["num_epochs"] - 1))
    pr = {k: float(v) for k, v in rv.items() if "PerfectRecoPct" in k and k.count("_") == 1}
    scores = torch.cat(score_chunks) if score_chunks else torch.empty(0)
    labels = torch.cat(label_chunks) if label_chunks else torch.empty(0, dtype=torch.long)
    cls_weights = torch.cat(weight_chunks) if weight_chunks else torch.empty(0)
    auc = weighted_roc_auc(scores, labels, cls_weights)
    if labels.numel() == 0 or torch.unique(labels).numel() < 2:
        best_z, best_thr = float("nan"), float("nan")
    else:
        best_z, best_thr = best_asimov_z(scores, labels, cls_weights)
    print(f"epoch {epoch}: train_loss={ep_loss/max(ep_w,1e-9):.4f}  "
          f"val={ {k: round(v,4) for k,v in list(pr.items())[:4]} }")
    print(f"epoch {epoch}: class_auc={auc:.4f}  best_asimov_z={best_z:.4f}  thr={best_thr:.4f}")
    if CONFIG["wandb"]:
        wandb.log({f"{k}": float(v) for k, v in rv.items() if hasattr(v, "item") or isinstance(v, (int, float))} |
                  {"train/epoch_loss": ep_loss / max(ep_w, 1e-9),
                   "val/class_auc": auc,
                   "val/best_asimov_z": best_z,
                   "val/best_asimov_thr": best_thr}, step=global_step)
    if (epoch % 25 == 0) or (epoch == CONFIG["num_epochs"] - 1):
        torch.save(model.state_dict(), os.path.join(MODELS_OUT, f"chkpt{epoch}_{global_step}.pth"))

print(f"done in {(time.time()-t0)/60:.1f} min; checkpoints + config.json in {OUT}")
if CONFIG["wandb"]:
    import wandb; wandb.finish()
