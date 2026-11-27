"""Hash every nominal row's input objects, then byte-check equal hash candidates."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


K = np.uint64(0xbf58476d1ce4e5b9)
M = np.uint64(0x94d049bb133111eb)
SEED = np.uint64(0x9e3779b97f4a7c15)


def fingerprints(raw):
    bits = raw[:, 2:, :6].view(np.uint32)
    h = np.full(bits.shape[:-1], SEED)
    for i in range(6):
        feature = bits[..., i].astype(np.uint64)
        feature[feature == 0x80000000] = 0
        h ^= feature
        h = (h ^ (h >> np.uint64(30))) * K
        h = (h ^ (h >> np.uint64(27))) * M
        h ^= h >> np.uint64(31)
    return np.where(raw[:, 2:, 0] != 5, h, 0).sum(axis=1, dtype=np.uint64)


def canonical(raw):
    real = raw[2:, :6][raw[2:, 0] != 5].copy()
    real[real == 0] = 0
    return real[np.lexsort(real.T)].tobytes()


def run(path, out, exclusions_path=None):
    datasets = {}
    for file in sorted(path.glob("dsid_*.memmap")):
        n, _, _ = Path(str(file) + ".shape").read_text().split(",")
        if int(n):
            dsid = int(file.stem.split("_")[1])
            datasets[dsid] = np.memmap(file, mode="r", dtype=np.float32, shape=(int(n), 17, 8))
    dsids = np.array(list(datasets), dtype=int)
    offsets = np.r_[0, np.cumsum([len(mm) for mm in datasets.values()])]
    h = np.empty(offsets[-1], dtype=np.uint64)
    for i, (d, mm) in enumerate(datasets.items()):
        for start in range(0, len(mm), 50_000):
            part = fingerprints(mm[start:start + 50_000])
            h[offsets[i] + start:offsets[i] + start + len(part)] = part
        print("Fingerprinted", d, len(mm), flush=True)
    ranked = np.argsort(h, kind="stable")
    sorted_h = h[ranked]
    ties = sorted_h[1:] == sorted_h[:-1]
    starts = np.flatnonzero(ties & np.r_[True, ~ties[:-1]])
    ends = np.flatnonzero(ties & np.r_[~ties[1:], True]) + 2
    print("Candidate hash groups:", len(starts), flush=True)
    by_split = Counter()
    byte_equality = Counter()
    all_pairs = []
    for start, end in zip(starts, ends):
        global_rows = ranked[start:end]
        dataset_index = np.searchsorted(offsets, global_rows, side="right") - 1
        local_rows = global_rows - offsets[dataset_index]
        matches = []
        for i, row in zip(dataset_index, local_rows):
            d = int(dsids[i])
            r = int(row)
            row_content = datasets[d][r]
            proof = canonical(row_content)
            for other_d, other_r, other_proof, other_raw in matches:
                if proof != other_proof:
                    continue
                byte_equality["full_row"] += row_content.tobytes() == other_raw.tobytes()
                byte_equality["ordered_inputs_and_padding"] += (
                    row_content[2:, :6].tobytes() == other_raw[2:, :6].tobytes()
                )
                byte_equality["mc_weight"] += row_content[1, 0].tobytes() == other_raw[1, 0].tobytes()
                all_pairs.append((other_d, other_r, d, r,
                                  bool(np.array_equal(row_content[2:, :6], other_raw[2:, :6])),
                                  bool(row_content[1, 2] == other_raw[1, 2])))
                by_split[tuple(sorted((other_r % 8, r % 8)))] += 1
            matches.append((d, r, proof, row_content))
    even_against_odd = set()
    later_even_vs_earlier_even = set()
    for d, r, other_d, other_r, _, _ in all_pairs:
        if (r % 2) != (other_r % 2):
            even_against_odd.add((other_d, other_r) if r % 2 else (d, r))
        if r % 2 == 0 and other_r % 2 == 0 and (r % 8) != (other_r % 8):
            later_even_vs_earlier_even.add((other_d, other_r) if (r % 8) < (other_r % 8) else (d, r))
    report = {
        "rows": int(len(h)),
        "dsids": {str(d): len(mm) for d, mm in datasets.items()},
        "odd_rows": sum(len(mm)//2 for mm in datasets.values()),
        "candidate_hash_groups": len(starts),
        "exact_duplicate_pairs": len(all_pairs),
        "cross_dsid_pairs": sum(a != c for a, _, c, _, _, _ in all_pairs),
        "byte_equality_counts": dict(byte_equality),
        "exact_pairs_by_residues": {f"{a},{b}": n for (a,b), n in sorted(by_split.items())},
        "pairs": all_pairs,
        "pair_columns": ["dsid_1", "row_1", "dsid_2", "row_2",
                         "identical_model_objects_including_padding_and_order",
                         "equal_event_number_modulo_1000"],
        "exclusions": {"teacher_seen_even": sorted(even_against_odd),
                       "later_even_duplicate": sorted(later_even_vs_earlier_even)},
        "definition": "Compare exact float32 vectors of model inputs and type for non-padding objects, accounting for token permutation and signed zero. Padding and truth/labels/weights are excluded. Hash candidates validated by byte equality.",
    }
    out.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    if exclusions_path is not None:
        exclusions_path.write_text(json.dumps(report["exclusions"], separators=(",", ":")) + "\n")
    print("Complete.", out, report["rows"], report["exact_duplicate_pairs"],
          "cross-DSID", report["cross_dsid_pairs"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--exclusions", type=Path)
    args = parser.parse_args()
    run(args.data, args.out, args.exclusions)
