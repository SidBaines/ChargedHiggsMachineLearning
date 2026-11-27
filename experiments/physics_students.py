"""Paired rank-one joint students on real events, with held-out scalar substitution."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hd_interp import load_trained
from models.models import TestNetwork

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2701, 2702, 2703)
ARMS = ("labels", "soft_and_labels")
PAD = 5
TEMP = 2.0


def rank_one_hook(block):
    """Contract each value with the output projection and legacy down vector first."""
    def hook(mha, inputs, kwargs, output):
        x = inputs[0]
        batch, objects, width = x.shape
        heads = mha.num_heads
        head_width = width // heads
        q, k, v = F.linear(x, mha.in_proj_weight, mha.in_proj_bias).chunk(3, dim=-1)
        q, k, v = [part.reshape(batch, objects, heads, head_width).transpose(1, 2)
                   for part in (q, k, v)]
        out_projection = mha.out_proj.weight.reshape(width, heads, head_width)
        down = torch.cat([projection.weight for projection in block["bottleneck_down"]])
        value_projection = (down.T[..., None] * out_projection).sum(0)
        values = (v * value_projection[None, :, None, :]).sum(-1, keepdim=True)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(head_width)
        mask = kwargs["key_padding_mask"][:, None, None, :]
        attention = logits.masked_fill(mask, -torch.inf).softmax(dim=-1)
        scalars = (attention @ values).squeeze(-1).transpose(1, 2)
        scalars = block["scalar_messages"](scalars)
        up = torch.cat([projection.weight for projection in block["bottleneck_up"]], dim=1)
        return F.linear(scalars, up, mha.out_proj.bias), None

    return hook


class AttentionWithoutDuplicateWork(nn.MultiheadAttention):
    """The registered rank-one hook does the complete attention calculation."""
    def forward(self, query, key, value, **kwargs):
        return query, None


def student(device, hooked=True):
    model = TestNetwork(
        bottleneck_attention=1, hidden_dim=48, hidden_dim_mlp=96, num_heads=4,
        num_attention_blocks=3, embedding_size=6, num_particle_types=6,
        num_classes=3, add_event_head=True,
    )
    if hooked:
        for block in model.attention_blocks:
            attention = block["self_attention"]
            # The subclass shares the existing parameter layout and TestNetwork call path.
            attention.__class__ = AttentionWithoutDuplicateWork
            block["scalar_messages"] = nn.Identity()
            attention.register_forward_hook(rank_one_hook(block), with_kwargs=True)
    return model.to(device)


def verify_rank_one(model, x, types):
    from interp.activations import ActivationCache, hook_attention_heads

    reference = student(x.device, hooked=False).eval()
    reference.load_state_dict(model.state_dict())
    cache = ActivationCache()
    for module, hook in hook_attention_heads(reference, cache, bottleneck_attention_output=1):
        module.register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad():
        ours = model(x, types)
        theirs = reference(x, types)
    for head in ours:
        torch.testing.assert_close(ours[head], theirs[head], rtol=1e-4, atol=1e-5)
    return {head: float((ours[head] - theirs[head]).abs().max()) for head in ours}


def tensors(sample, device):
    return {
        "x": torch.as_tensor(sample["x"], dtype=torch.float32, device=device),
        "types": torch.as_tensor(sample["types"], dtype=torch.long, device=device),
        "label": torch.as_tensor(sample["label"], dtype=torch.long, device=device),
        "truth": torch.as_tensor(sample["truth"], dtype=torch.long, device=device).clamp(0, 2),
    }


@torch.no_grad()
def predict(model, data, batch_size=256):
    chunks = {"reco": [], "event": []}
    for start in range(0, len(data["x"]), batch_size):
        batch = slice(start, start + batch_size)
        logits = model(data["x"][batch], data["types"][batch])
        for head, value in logits.items():
            chunks[head].append(value)
    return {head: torch.cat(values) for head, values in chunks.items()}


def object_mean(values, types, label):
    mask = types != PAD
    signal = (label != 0).to(values.dtype)
    per_event = (values * mask).sum(1) / mask.sum(1).clamp_min(1)
    return (per_event * signal).sum() / signal.sum().clamp_min(1)


def hard_losses(logits, batch):
    object_ce = F.cross_entropy(logits["reco"].transpose(1, 2), batch["truth"], reduction="none")
    return object_mean(object_ce, batch["types"], batch["label"]), F.cross_entropy(logits["event"], batch["label"])


def losses(logits, batch, teacher=None):
    object_ce, event_ce = hard_losses(logits, batch)
    hard = object_ce + event_ce
    if teacher is None:
        return hard
    object_kl = F.kl_div((logits["reco"] / TEMP).log_softmax(-1), teacher["reco"],
                        reduction="none").sum(-1)
    soft_object = object_mean(object_kl, batch["types"], batch["label"])
    soft_event = F.kl_div((logits["event"] / TEMP).log_softmax(-1), teacher["event"],
                          reduction="batchmean")
    return 0.5 * hard + 0.5 * TEMP ** 2 * (soft_object + soft_event)


def probabilities(logits):
    return {name: values.softmax(-1).detach().cpu().numpy() for name, values in logits.items()}


def signed_fraction(passed, weights):
    total = weights.sum()
    return float(weights[passed].sum() / total) if total else None


def performance(probs, sample):
    label = sample["label"]
    truth = sample["truth"].clip(0, 2)
    weights = sample["weight"]
    types = sample["types"]
    signal = label != 0
    reco = probs["reco"].argmax(-1)
    correct = (reco == truth) | (types == PAD)
    event = probs["event"]
    return {
        "reco_perfect_signed": signed_fraction(correct[signal].all(1), weights[signal]),
        "reco_perfect_count_fraction": float(correct[signal].all(1).mean()),
        "reco_object_count_fraction": float(correct[signal][types[signal] != PAD].mean()),
        "event_accuracy_count_fraction": float((event.argmax(1) == label).mean()),
        "auc_unweighted": float(roc_auc_score(signal, 1 - event[:, 0])),
        "auc_abs_yield_weighted": float(roc_auc_score(signal, 1 - event[:, 0], sample_weight=np.abs(weights))),
    }


def batches(size, batch_size, updates, seed):
    for_round = torch.Generator().manual_seed(seed + 4000)
    remaining = updates
    while remaining:
        for indices in torch.randperm(size, generator=for_round).split(batch_size):
            yield indices
            remaining -= 1
            if remaining == 0:
                return


def fit(training, validation, validation_sample, teacher, seed, arm, device, args):
    torch.manual_seed(seed)
    model = student(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1., (step + 1) / 30) *
        (0.2 + 0.8 * (1 + math.cos(math.pi * min(step, args.updates) / args.updates)) / 2),
    )
    log_steps = {0, 1, args.updates, *(math.ceil(args.updates * p) for p in (0.1, 0.25, 0.5, 0.75))}
    history = []
    started = time.monotonic()
    interval_loss = 0.
    interval_n = 0

    def log(step):
        nonlocal interval_loss, interval_n
        val_logits = predict(model, validation)
        reco_ce, event_ce = hard_losses(val_logits, validation)
        record = performance(probabilities(val_logits), validation_sample)
        record.update(step=step, elapsed_seconds=time.monotonic() - started, validation_hard_reco_ce=float(reco_ce),
                      validation_hard_event_ce=float(event_ce),
                      training_objective_mean=interval_loss / interval_n if interval_n else None)
        history.append(record)
        print(f"{seed} {arm} {json.dumps(record)}", flush=True)
        interval_loss, interval_n = 0., 0

    log(0)
    for step, indices in enumerate(batches(len(training["x"]), args.batch_size, args.updates, seed), 1):
        take = indices.to(device)
        batch = {name: values[take] for name, values in training.items()}
        target = {name: values[take] for name, values in teacher.items()} if arm == ARMS[1] else None
        optimizer.zero_grad(set_to_none=True)
        loss = losses(model(batch["x"], batch["types"]), batch, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite objective for {seed}/{arm} at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        schedule.step()
        interval_loss += float(loss.detach())
        interval_n += 1
        if step in log_steps:
            log(step)
    return model.eval(), history


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_predictions(path, sample, probs):
    np.savez_compressed(path, **{name: sample[name] for name in ("types", "truth", "label", "dsid", "row", "weight", "mc")},
                        reco_prob=probs["reco"].astype(np.float32), event_prob=probs["event"].astype(np.float32))


def main():
    from physics_samples import sample_events

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=ROOT / "tmp_data_20250321v1_all")
    cli.add_argument("--teacher", type=Path, default=ROOT / "output/20260624-163646_TrainingOutput")
    cli.add_argument("--out", type=Path, default=Path("output/physics_interp/students"))
    cli.add_argument("--updates", type=int, default=800)
    cli.add_argument("--batch-size", type=int, default=128)
    cli.add_argument("--lr", type=float, default=0.002)
    cli.add_argument("--train-signal", type=int, default=3200)
    cli.add_argument("--train-background", type=int, default=1600)
    cli.add_argument("--eval-signal", type=int, default=300)
    cli.add_argument("--eval-background", type=int, default=180)
    cli.add_argument("--threads", type=int, default=2)
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--benchmark", action="store_true")
    cli.add_argument("--formula-only", action="store_true")
    args = cli.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    if args.formula_only:
        from physics_students_formula import run_formulas

        record = json.loads((args.out / "training.json").read_text())
        run_formulas(args, record, device)
        return

    args.out.mkdir(exist_ok=True, parents=True)
    teacher, _, ckpt = load_trained(str(args.teacher), device)
    n_signal, n_background = (32, 16) if args.benchmark else (args.train_signal, args.train_background)
    raw_training = sample_events(args.data, residue=1, n_signal=n_signal, n_background=n_background)
    mode = torch.no_grad()
    with mode:
        training = tensors(raw_training, device)
        target = {name: (value / TEMP).softmax(-1) for name, value in predict(teacher, training).items()}
    raw_splits = {residue: sample_events(args.data, residue=residue,
                                       n_signal=min(12, args.eval_signal) if args.benchmark else args.eval_signal,
                                       n_background=min(10, args.eval_background) if args.benchmark else args.eval_background)
                  for residue in ((2,) if args.benchmark else (2, 4, 6))}
    eval_splits = {residue: tensors(sample, device) for residue, sample in raw_splits.items()}
    rank_check = verify_rank_one(student(device).eval(), training["x"][:64], training["types"][:64])
    print(f"training n={len(training['x'])}; evaluation n={list(map(lambda d: len(d['x']), eval_splits.values()))}; "
          f"rank-one vs legacy max logit difference={rank_check}", flush=True)
    record = {
        "teacher_checkpoint": str(ckpt), "teacher_temperature": TEMP,
        "config": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "sample_counts": {1: len(training["x"]), **{key: len(value["x"]) for key, value in eval_splits.items()}},
        "rank_one_vs_legacy_max_logit_difference": rank_check,
        "training_reco": "unweighted mean per signal event over non-padding tokens",
        "training_event": "unweighted event cross entropy",
        "soft_mix": "equal hard CE and T^2 KL, same reco/event inclusion as hard losses",
        "models": {},
    }
    print(f"student parameters (including unused legacy bottleneck biases)={sum(p.numel() for p in student(device).parameters())}", flush=True)
    if not args.benchmark:
        np.savez_compressed(args.out / "train_ids.npz", dsid=raw_training["dsid"], row=raw_training["row"],
                            label=raw_training["label"])
        for residue, data in eval_splits.items():
            save_predictions(args.out / f"teacher_r{residue}.npz", raw_splits[residue], probabilities(predict(teacher, data)))
    for seed in (SEEDS[:1] if args.benchmark else SEEDS):
        for arm in ARMS:
            model, history = fit(training, eval_splits[2], raw_splits[2], target, seed, arm, device, args)
            key = f"{seed}_{arm}"
            if args.benchmark:
                continue
            torch.save(model.state_dict(), args.out / f"{key}.pt")
            metrics = {}
            for residue, data in eval_splits.items():
                probs = probabilities(predict(model, data))
                save_predictions(args.out / f"{key}_r{residue}.npz", raw_splits[residue], probs)
                metrics[residue] = performance(probs, raw_splits[residue])
            record["models"][key] = {"seed": seed, "arm": arm, "curve": history, "performance": metrics}
            save_json(args.out / "training.json", record)


if __name__ == "__main__":
    main()
