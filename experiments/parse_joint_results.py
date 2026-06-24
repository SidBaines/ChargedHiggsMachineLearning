"""Parse a joint real-bkg queue's logs into a markdown results table.

Reads the LAST `JOINTSUMMARY {...}` line (final epoch) from each
<run_root>/logs/*.log and prints a markdown table: run | reco_all + per-cat |
AUC | Z(uncal) | train_loss. Usage: parse_joint_results.py <run_root>
(defaults to the newest output/joint_real_bkg_* dir).
"""
import os, sys, json, glob, re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def newest_run_root():
    cands = sorted(glob.glob(os.path.join(REPO, "output", "joint_real_bkg_*")))
    return cands[-1] if cands else None

def last_summary(log_path):
    last = None
    with open(log_path) as f:
        for line in f:
            if line.startswith("JOINTSUMMARY "):
                try:
                    last = json.loads(line[len("JOINTSUMMARY "):])
                except json.JSONDecodeError:
                    pass
    return last

# Headline reco columns only (mirrors the plan's table: reco all + per-cat); the
# JOINTSUMMARY dict also carries mass-window + channel variants we omit here.
HEADLINE = [("val/PerfectRecoPct_all", "reco_all"),
            ("val/PerfectRecoPct_all_lvbb", "lvbb"),
            ("val/PerfectRecoPct_all_qqbb", "qqbb")]
HEADLINE += [(f"val/PerfectRecoPct_all_cat{i}", f"cat{i}") for i in range(6)]

def headline_cols(rows):
    present = set()
    for _, s in rows:
        if s and s.get("reco"):
            present.update(s["reco"].keys())
    return [(k, lbl) for k, lbl in HEADLINE if k in present]

def main():
    run_root = sys.argv[1] if len(sys.argv) > 1 else newest_run_root()
    if not run_root or not os.path.isdir(run_root):
        print(f"no run_root found ({run_root})"); return
    logs = sorted(glob.glob(os.path.join(run_root, "logs", "*.log")))
    rows = []
    for lp in logs:
        name = os.path.basename(lp)[:-4]
        s = last_summary(lp)
        rows.append((name, s))
    print(f"# Joint real-bkg results — {os.path.basename(run_root)}\n")
    cols = headline_cols(rows)
    labels = [lbl for _, lbl in cols]
    header = ["run"] + labels + ["AUC", "Z(uncal)", "train_loss", "epoch"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for name, s in rows:
        if not s:
            print(f"| {name} | " + " | ".join(["—"] * (len(header) - 1)) + " |")
            continue
        reco = s.get("reco", {})
        cells = [name] + [f"{reco.get(k, float('nan')):.3f}" for k, _ in cols]
        cells += [f"{s.get('auc', float('nan')):.4f}",
                  f"{s.get('asimov_z_uncalibrated', float('nan')):.3f}",
                  f"{s.get('train_loss', float('nan')):.4f}",
                  str(s.get("epoch", "—"))]
        print("| " + " | ".join(cells) + " |")
    print("\n*Caveat: AUC is comparative; Asimov Z is UNCALIBRATED (abs MC weights, no "
          "per-DSID luminosity normalization — see plan R6). Position-split train/val.*")

if __name__ == "__main__":
    main()
