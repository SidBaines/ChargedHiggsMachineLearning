import os
import re
import shutil
import subprocess
import sys
import traceback


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
TRAIN_SCRIPT = os.path.join(REPO, "experiments", "train_joint.py")
EVAL_SCRIPT = os.path.join(REPO, "experiments", "eval_single_readout.py")
OUT_RE = re.compile(r"^out:\s*(.+?)\s*$", re.MULTILINE)
SOFTOR_AUC_RE = re.compile(r"^parameter-free softor:\s*AUC=", re.MULTILINE)
FAIR_AUC_RE = re.compile(r"^fair-readout logistic:\s*AUC=", re.MULTILINE)


_RESULTS = {}


def _tail(text, lines=100):
    return "\n".join(text.splitlines()[-lines:])


def _parse_output_dir(stdout):
    match = OUT_RE.search(stdout)
    return match.group(1) if match else None


def _cleanup_output_dir(out_dir):
    if not out_dir:
        return

    output_root = os.path.realpath(os.path.join(REPO, "output"))
    out_real = os.path.realpath(out_dir)
    try:
        common = os.path.commonpath([output_root, out_real])
    except ValueError:
        return

    if common != output_root or out_real == output_root:
        return

    try:
        shutil.rmtree(out_real)
    except OSError as exc:
        print(f"cleanup warning: could not remove {out_real}: {exc}", file=sys.stderr)


def _find_checkpoint(out_dir):
    ckpts = []
    models_dir = os.path.join(out_dir, "models")
    for root, _, files in os.walk(models_dir):
        for filename in files:
            if filename.startswith("chkpt") and filename.endswith(".pth"):
                ckpts.append(os.path.join(root, filename))
    ckpts.sort()
    assert ckpts, f"no chkpt*.pth found under {models_dir}"
    return ckpts[-1]


def test_eval_single_readout_smoke():
    train_cmd = [
        PY,
        TRAIN_SCRIPT,
        "--smoke",
        "--device",
        "cpu",
        "--no-wandb",
        "--mode",
        "single",
        "--pseudo-bkg-dsid",
        "510124",
        "--d-model",
        "8",
        "--blocks",
        "1",
        "--num-heads",
        "2",
        "--no-mlp",
        "--batch-size",
        "256",
    ]
    train_result = subprocess.run(
        train_cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=240,
    )
    out_dir = _parse_output_dir(train_result.stdout)
    try:
        train_combined = train_result.stdout + train_result.stderr
        assert train_result.returncode == 0, (
            f"train_joint.py failed with returncode {train_result.returncode}\n"
            f"command: {train_cmd}\n"
            f"tail:\n{_tail(train_combined)}"
        )
        assert out_dir is not None, f"could not parse output dir from train stdout:\n{_tail(train_result.stdout)}"
        ckpt = _find_checkpoint(out_dir)

        eval_cmd = [
            PY,
            EVAL_SCRIPT,
            "--ckpt",
            ckpt,
            "--pseudo-bkg-dsid",
            "510124",
            "--device",
            "cpu",
            "--fit-batches",
            "5",
            "--eval-batches",
            "5",
            "--batch-size",
            "256",
        ]
        eval_result = subprocess.run(
            eval_cmd,
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=240,
        )
        eval_combined = eval_result.stdout + eval_result.stderr
        assert eval_result.returncode == 0, (
            f"eval_single_readout.py failed with returncode {eval_result.returncode}\n"
            f"command: {eval_cmd}\n"
            f"tail:\n{_tail(eval_combined)}"
        )
        assert SOFTOR_AUC_RE.search(eval_result.stdout), (
            "eval stdout missing parameter-free softor AUC line\n"
            f"stdout tail:\n{_tail(eval_result.stdout)}"
        )
        assert FAIR_AUC_RE.search(eval_result.stdout), (
            "eval stdout missing fair-readout AUC line\n"
            f"stdout tail:\n{_tail(eval_result.stdout)}"
        )
        _RESULTS["eval_stdout"] = eval_result.stdout
    finally:
        _cleanup_output_dir(out_dir)


if __name__ == "__main__":
    failures = 0
    for name, test_fn in sorted(globals().items()):
        if name.startswith("test_") and callable(test_fn):
            try:
                test_fn()
            except Exception:
                failures += 1
                print(f"{name}: FAIL")
                traceback.print_exc()
            else:
                auc_lines = [
                    line
                    for line in _RESULTS.get("eval_stdout", "").splitlines()
                    if "AUC=" in line
                ]
                suffix = " " + " | ".join(auc_lines) if auc_lines else ""
                print(f"{name}: PASS{suffix}")
    if failures:
        raise SystemExit(1)
