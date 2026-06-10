"""Train a small interpretable 'model organism' reconstruction net on this Mac (MPS).

Recipe mirrors TrainLowLevelReconstruction.py (entropy-penalty branch) but:
- uses interp.activations.hook_attention_heads(detach=False) instead of mechinterputils
  (which needs pysr) to feed grad-attached attention weights to HEPLossWithEntropy;
- serializes the FULL config as JSON next to the checkpoints (Phase 1.2);
- logs the full config to wandb (the old runs logged none).

Usage:  .venv/bin/python tmp_train_organism.py [--smoke]
"""
import os, sys, json, time, argparse
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
from models.models import TestNetwork
from interp.activations import ActivationCache, hook_attention_heads
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset
from metrics.lowlevelrecometrics import HEPLossWithEntropy, HEPMetrics
from utils.utils import basic_lr_scheduler

p = argparse.ArgumentParser(); p.add_argument("--smoke", action="store_true")
p.add_argument("--device", default=None, help="override CONFIG device (cpu/mps)")
ARGS = p.parse_args()

CONFIG = dict(
    # --- provenance ---
    purpose="model organism v1: replicate ent1-d20-2blk recipe locally (2026-06-10)",
    data="20250321v1 signal-only (SSD copy), truth {0,1,2,3}->{0,1,2}",
    # --- architecture (matches registry ent1-d20-2blk) ---
    d_model=20, num_blocks=2, num_heads=4, d_mlp=200, include_mlp=True,
    bottleneck_attention=None, d_attn=None, embedding_size=6, num_particle_types=6,
    feature_set=["phi", "eta", "pt", "m", "tag"], use_lorentz_invariant_features=True,
    num_object_net_layers=1, is_layer_norm=False, dropout_p=0.0, num_classes=3,
    # --- training ---
    batch_size=4096, num_epochs=30, learning_rate=3e-4, learning_rate_low=5e-7,
    learning_rate_log_decay=True, warmup_steps=100, weight_decay=1e-6,
    entropy_loss=True, entropy_weight=1e-2, target_entropy=0,
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

torch.manual_seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"])
device = CONFIG["device"] if torch.backends.mps.is_available() else "cpu"

stamp = time.strftime("%Y%m%d-%H%M%S")
run_name = f"_{stamp}_LowLevel_ORG2026_d{CONFIG['d_model']}b{CONFIG['num_blocks']}_YesEnt1_NoBn"
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
val_dl = ProportionalMemoryMappedDataset(is_train=False, shuffle=False, shuffle_batch=False, **mk)
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
).to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

# grad-attached attention cache for the entropy loss
cache = ActivationCache()
hook_pairs = hook_attention_heads(model, cache, detach=False, SINGLE_ATTENTION=False,
                                  bottleneck_attention_output=CONFIG["bottleneck_attention"])
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
    """Padding objects are all-zero -> sqrt(0)/asinh(0/0) give NaN/Inf *gradients*
    (MPS is less forgiving than CPU/CUDA). Give them benign 4-momenta; they are
    masked out of attention, loss and metrics, so values are irrelevant."""
    pad = (types == N_CTX - 1).unsqueeze(-1)
    safe = torch.zeros_like(x)
    safe[..., 0] = 1e-3; safe[..., 1] = 1e-3; safe[..., 3] = 2e-3
    return torch.where(pad, safe, x)

nan_skips = 0
steps_per_epoch = len(train_dl)
num_lr_steps = CONFIG["num_epochs"] * steps_per_epoch
global_step = 0
SMOKE_STEPS = 30
t0 = time.time()
for epoch in range(CONFIG["num_epochs"]):
    train_dl._reset_indices(); val_dl._reset_indices()
    model.train()
    ep_loss = ep_w = 0.0
    n_steps = SMOKE_STEPS if ARGS.smoke else steps_per_epoch
    for bi in range(n_steps):
        if bi % 10 == 0:
            lr = basic_lr_scheduler(bi + epoch * steps_per_epoch, CONFIG["learning_rate"],
                                    CONFIG["learning_rate_low"], num_lr_steps,
                                    CONFIG["learning_rate_log_decay"], warmup_steps=CONFIG["warmup_steps"],
                                    warmup_rate=1e-3)
            for g in optimizer.param_groups: g["lr"] = lr
        global_step += 1
        b = next(train_dl)
        x, y, w, types = b["x"], b["y"], b["train_wts"], b["types"]
        x = sanitize_padding(x, types)
        optimizer.zero_grad()
        out = model(x[..., :CONFIG["model_input_vars"]], types).squeeze()
        loss = criterion(cache, out, x[..., -1], types, N_CTX - 1, MAX_OBJS, w, False, x[..., :4])
        if isinstance(loss, tuple):
            loss, loss_dict = loss
        if not torch.isfinite(loss):
            nan_skips += 1
            print(f"  !! non-finite loss at ep {epoch} step {bi} — skipping batch ({nan_skips} skips)")
            if nan_skips > 20:
                raise RuntimeError("too many non-finite batches; aborting")
            optimizer.zero_grad(); continue
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            nan_skips += 1
            print(f"  !! non-finite grad-norm at ep {epoch} step {bi} — skipping ({nan_skips})")
            optimizer.zero_grad(); continue
        optimizer.step()
        ep_loss += loss.item() * w.sum().item(); ep_w += w.sum().item()
        if bi % 50 == 0:
            print(f"  ep {epoch} step {bi}/{n_steps} loss {loss.item():.4f} ({(time.time()-t0):.0f}s)")
        if CONFIG["wandb"] and global_step % 20 == 0:
            wandb.log({"train/loss": loss.item(), "lr": lr, "epoch": epoch}, step=global_step)

    # ---- epoch-end validation (subset for speed; full val at final epoch) ----
    model.eval(); val_metrics.reset()
    n_val = len(val_dl) if epoch == CONFIG["num_epochs"] - 1 else min(30, len(val_dl))
    if ARGS.smoke: n_val = 3
    with torch.no_grad():
        for _ in range(n_val):
            vb = next(val_dl)
            vx, vtypes = vb["x"], vb["types"]
            vx = sanitize_padding(vx, vtypes)
            vout = model(vx[..., :CONFIG["model_input_vars"]], vtypes)
            val_metrics.update(vout.squeeze().cpu(), vx[..., -1].cpu(), vb["MC_Wts"].cpu(),
                               vb["dsids"].cpu(), vtypes.cpu())
    rv = val_metrics.compute_and_log(1, "val", 0, 3, False, None, calc_all=(epoch == CONFIG["num_epochs"] - 1))
    pr = {k: float(v) for k, v in rv.items() if "PerfectRecoPct" in k and k.count("_") == 1}
    print(f"epoch {epoch}: train_loss={ep_loss/max(ep_w,1e-9):.4f}  val={ {k: round(v,4) for k,v in list(pr.items())[:4]} }")
    if CONFIG["wandb"]:
        wandb.log({f"{k}": float(v) for k, v in rv.items() if hasattr(v, "item") or isinstance(v, (int, float))} |
                  {"train/epoch_loss": ep_loss / max(ep_w, 1e-9)}, step=global_step)
    torch.save(model.state_dict(), os.path.join(MODELS_OUT, f"chkpt{epoch}_{global_step}.pth"))

print(f"done in {(time.time()-t0)/60:.1f} min; checkpoints + config.json in {OUT}")
if CONFIG["wandb"]:
    import wandb; wandb.finish()
