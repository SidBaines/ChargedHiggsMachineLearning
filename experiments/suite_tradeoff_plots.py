"""Phase-2.2: parse the suite logs (tmp_suite/*.log) and produce the
interpretability-tradeoff views from the winner-recipe d152 runs.

Each training run logs, once per epoch, a full validation dict like
  {'val/PerfectRecoPct_all': tensor(0.89), 'val/PerfectRecoPct_all_cat0': ...}
We pull PerfectRecoPct_all, the lvbb/qqbb split, and the six category numbers
for every epoch of every {none,ent,bn1,both} x seed run, then:

  1. print + CSV the final-epoch per-category table (mean +/- seed spread);
  2. plot PerfectRecoPct_all vs epoch (4 conditions) -> where they diverge;
  3. plot per-category cat0..cat5 vs epoch (the cost lives in cats 0-3).

Usage: .venv/bin/python experiments/suite_tradeoff_plots.py [--size d152]
Writes tmp_plots/suite_<size>_*.png + tmp_plots/suite_<size>_final_by_category.csv
"""
import os, re, glob, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--size", default="d152")
ap.add_argument("--logdir", default=os.path.join(REPO, "tmp_suite"))
ap.add_argument("--outdir", default=os.path.join(REPO, "tmp_plots"))
ARGS = ap.parse_args()
os.makedirs(ARGS.outdir, exist_ok=True)

CONDS = ["none", "ent", "bn1", "both"]
LABELS = {"none": "none (unconstrained)", "ent": "entropy",
          "bn1": "bottleneck-1", "both": "entropy + bn1"}
COLORS = {"none": "#1f77b4", "ent": "#2ca02c", "bn1": "#ff7f0e", "both": "#d62728"}
CATS = [f"cat{i}" for i in range(6)]
KEYS = ["all", "all_lvbb", "all_qqbb"] + [f"all_{c}" for c in CATS]

val_re = re.compile(r"'val/PerfectRecoPct_([\w.]+)': tensor\(([0-9.]+)\)")

def parse_log(path):
    """Return list (per epoch) of dicts of the metrics we track."""
    epochs = []
    with open(path) as fh:
        for line in fh:
            if "PerfectRecoPct_all_cat0'" not in line:   # only the full per-epoch dicts
                continue
            d = {k: float(v) for k, v in val_re.findall(line)}
            epochs.append({k: d.get(k, np.nan) for k in KEYS})
    return epochs

# ---- load all seeds per condition ----
data = {}   # cond -> array [n_seeds, n_epochs, n_keys]
for cond in CONDS:
    runs = []
    for f in sorted(glob.glob(os.path.join(ARGS.logdir, f"{ARGS.size}_{cond}_s*.log"))):
        if "LEGACY" in f:
            continue
        ep = parse_log(f)
        if ep:
            runs.append(np.array([[e[k] for k in KEYS] for e in ep]))
    if not runs:
        print(f"WARN no runs for {cond}")
        continue
    n = min(len(r) for r in runs)                 # align epoch counts
    data[cond] = np.stack([r[:n] for r in runs])  # [seeds, epochs, keys]
    print(f"{cond}: {len(runs)} seeds, {n} epochs")

ki = {k: i for i, k in enumerate(KEYS)}

# ---- (1) final-epoch per-category table ----
print(f"\n=== {ARGS.size} winner-recipe: final-epoch PerfectRecoPct (mean over seeds) ===")
hdr = ["condition", "all", "lvbb", "qqbb"] + CATS
print("  " + "  ".join(f"{h:>9s}" for h in hdr))
rows = []
for cond in CONDS:
    if cond not in data:
        continue
    fin = data[cond][:, -1, :]                     # [seeds, keys]
    mean = fin.mean(0)
    row = [cond] + [f"{mean[ki[k]]:.4f}" for k in ["all", "all_lvbb", "all_qqbb"] + [f"all_{c}" for c in CATS]]
    rows.append([cond] + [float(mean[ki[k]]) for k in ["all", "all_lvbb", "all_qqbb"] + [f"all_{c}" for c in CATS]])
    print("  " + "  ".join(f"{c:>9s}" for c in row))

# cost vs none (per category), final epoch
if "none" in data:
    base = data["none"][:, -1, :].mean(0)
    print(f"\n=== cost vs unconstrained (pts, final epoch) ===")
    print("  " + "  ".join(f"{h:>9s}" for h in ["condition", "all"] + CATS))
    for cond in CONDS:
        if cond == "none" or cond not in data:
            continue
        m = data[cond][:, -1, :].mean(0)
        deltas = [(m[ki[k]] - base[ki[k]]) * 100 for k in ["all"] + [f"all_{c}" for c in CATS]]
        print("  " + f"{cond:>9s}  " + "  ".join(f"{d:>+9.2f}" for d in deltas))

csv_path = os.path.join(ARGS.outdir, f"suite_{ARGS.size}_final_by_category.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["condition", "all", "lvbb", "qqbb"] + CATS)
    for r in rows:
        w.writerow([r[0]] + [f"{x:.4f}" for x in r[1:]])
print(f"\nwrote {csv_path}")

# ---- (2) overall divergence curve ----
fig, ax = plt.subplots(figsize=(8, 5))
for cond in CONDS:
    if cond not in data:
        continue
    arr = data[cond][:, :, ki["all"]]              # [seeds, epochs]
    x = np.arange(arr.shape[1])
    ax.plot(x, arr.mean(0), color=COLORS[cond], label=LABELS[cond], lw=2)
    ax.fill_between(x, arr.min(0), arr.max(0), color=COLORS[cond], alpha=0.15)
ax.set_xlabel("epoch"); ax.set_ylabel("val PerfectRecoPct_all")
ax.set_title(f"{ARGS.size} winner recipe: interpretability constraints vs training")
ax.legend(); ax.grid(alpha=0.3)
p1 = os.path.join(ARGS.outdir, f"suite_{ARGS.size}_overall.png")
fig.tight_layout(); fig.savefig(p1, dpi=130); print(f"wrote {p1}")

# ---- (3) per-category divergence grid ----
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
for ci, cat in enumerate(CATS):
    ax = axes[ci // 3][ci % 3]
    for cond in CONDS:
        if cond not in data:
            continue
        arr = data[cond][:, :, ki[f"all_{cat}"]]
        x = np.arange(arr.shape[1])
        ax.plot(x, arr.mean(0), color=COLORS[cond], label=LABELS[cond], lw=1.8)
        ax.fill_between(x, arr.min(0), arr.max(0), color=COLORS[cond], alpha=0.12)
    ax.set_title(cat); ax.grid(alpha=0.3)
    if ci == 0:
        ax.legend(fontsize=8)
for ax in axes[1]:
    ax.set_xlabel("epoch")
for ax in axes[:, 0]:
    ax.set_ylabel("val PerfectRecoPct")
fig.suptitle(f"{ARGS.size} winner recipe: per-category training curves (cost lives in cats 0-3)")
p2 = os.path.join(ARGS.outdir, f"suite_{ARGS.size}_by_category.png")
fig.tight_layout(); fig.savefig(p2, dpi=130); print(f"wrote {p2}")

# ---- (4) the Phase-2.2 tradeoff figure: winner curve + thesis-era + legacy overlay ----
# Old thesis-era runs (legacy recipe, 1 seed) from the wandb archive. No bn1-only run.
import json
OLD_RUNS = {
    "none": "20250510-123629_DSSARVTSBN3_NoEnt_NoBn",
    "ent":  "20250512-093602_DSSARVTSBN3_YesEnt1_NoBn",
    "both": "20250512-093728_DSSARVTSBN3_YesEnt1_YesBn",   # THE THESIS MODEL
}
HARD = [f"all_cat{i}" for i in range(4)]                    # cats 0-3 = where cost lives

def _mean_keys(d, keylist):                                 # mean of PerfectRecoPct over keys
    return float(np.mean([d[f"val/PerfectRecoPct_{k}"] for k in keylist]))

def old_point(tag, keylist):
    d = json.load(open(os.path.join(REPO, "docs/run_configs", tag + ".json")))
    return _mean_keys(d["final_summary"], keylist)   # archived metrics live under final_summary

# new legacy-recipe runs trained on the winner architecture (final-epoch dict)
LEG_LOGS = sorted(glob.glob(os.path.join(ARGS.logdir, f"{ARGS.size}_*_LEGACY.log")))
legacy = {}                                                 # cond -> final-epoch metric dict
for f in LEG_LOGS:
    cond = os.path.basename(f).split("_")[1]                # d152_<cond>_s0_LEGACY
    ep = parse_log(f)
    if ep:
        legacy[cond] = ep[-1]

def winner_point(cond, keylist):                            # mean + seed min/max, final epoch
    fin = data[cond][:, -1, :]
    per_seed = np.mean([fin[:, ki[k]] for k in keylist], axis=0)
    return per_seed.mean(), per_seed.min(), per_seed.max()

order = [c for c in CONDS if c in data]
xs = np.arange(len(order))
panels = [("all events", ["all"]), ("hard categories (cat0–3, mean)", HARD)]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (title, keylist) in zip(axes, panels):
    # winner-recipe curve (mean +/- seed spread)
    m = np.array([winner_point(c, keylist)[0] for c in order])
    lo = np.array([winner_point(c, keylist)[1] for c in order])
    hi = np.array([winner_point(c, keylist)[2] for c in order])
    ax.plot(xs, m, "-o", color="#1f77b4", lw=2, ms=8, label="winner recipe (this work, 3 seeds)", zorder=3)
    ax.fill_between(xs, lo, hi, color="#1f77b4", alpha=0.15, zorder=1)
    # thesis-era points (legacy recipe, archived single runs)
    ox = [xs[order.index(c)] for c in OLD_RUNS if c in order]
    oy = [old_point(t, keylist) for c, t in OLD_RUNS.items() if c in order]
    ax.plot(ox, oy, "s--", color="#7f7f7f", lw=1.3, ms=7, label="thesis-era (legacy recipe, 1 seed)", zorder=2)
    # the thesis model itself (both)
    if "both" in order and "both" in OLD_RUNS:
        bx = xs[order.index("both")]
        ax.plot([bx], [old_point(OLD_RUNS["both"], keylist)], "*", color="#d62728",
                ms=18, label="thesis model", zorder=4)
    # new legacy-recipe runs on the winner architecture (isolate the recipe effect)
    lx = [xs[order.index(c)] for c in legacy if c in order]
    ly = [_mean_keys({f"val/PerfectRecoPct_{k}": legacy[c][k] for k in KEYS}, keylist)
          for c in legacy if c in order]
    if lx:
        ax.plot(lx, ly, "D", color="#ff7f0e", ms=9, label="legacy recipe, winner arch (1 seed)", zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels([LABELS[c] for c in order], rotation=20, ha="right")
    ax.set_title(title); ax.set_ylabel("val PerfectRecoPct"); ax.grid(alpha=0.3, axis="y")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle(f"{ARGS.size} interpretability tradeoff: constraint vs performance "
             f"(recipe lifts the whole curve ~+2.9 pts; constraint cost ~1.2 pts, recipe-stable)")
p3 = os.path.join(ARGS.outdir, f"suite_{ARGS.size}_tradeoff.png")
fig.tight_layout(); fig.savefig(p3, dpi=130); print(f"wrote {p3}")
