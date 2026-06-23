import os, sys, subprocess, re, shutil
import math
import traceback


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SCRIPT = os.path.join(REPO, "experiments", "train_joint.py")

COMMON = [
    "--device",
    "cpu",
    "--no-wandb",
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

TRAIN_LOSS_RE = re.compile(r"train_loss=([0-9.]+)")
CLASS_AUC_RE = re.compile(r"class_auc=(\S+)")
OUT_RE = re.compile(r"^out:\s*(.+?)\s*$", re.MULTILINE)

_RESULTS = {}


def _tail(text, lines=80):
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


def _parse_train_loss(stdout):
    values = []
    for match in TRAIN_LOSS_RE.finditer(stdout):
        value = float(match.group(1))
        if math.isfinite(value):
            values.append(value)
    assert values, "missing finite train_loss=... line in stdout"
    return values[-1]


def _parse_class_auc(stdout):
    values = []
    for match in CLASS_AUC_RE.finditer(stdout):
        raw = match.group(1)
        try:
            value = float(raw)
        except ValueError as exc:
            raise AssertionError(f"class_auc={raw!r} did not parse as float or nan") from exc
        if math.isfinite(value) or raw.lower() == "nan":
            values.append(value)
    assert values, "missing parseable class_auc=... line in stdout"
    return values[-1]


def run_combo(extra_args, timeout=240):
    result = subprocess.run(
        [PY, SCRIPT, *extra_args, *COMMON],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out_dir = _parse_output_dir(result.stdout)
    try:
        combined = result.stdout + result.stderr
        assert result.returncode == 0, (
            f"train_joint.py failed with returncode {result.returncode}\n"
            f"command: {[PY, SCRIPT, *extra_args, *COMMON]}\n"
            f"tail:\n{_tail(combined)}"
        )
        train_loss = _parse_train_loss(result.stdout)
        class_auc = _parse_class_auc(result.stdout)
        return train_loss, class_auc
    finally:
        _cleanup_output_dir(out_dir)


def _run_test(name, extra_args):
    _RESULTS[name] = run_combo(extra_args)


def test_single():
    _run_test("test_single", ["--smoke", "--mode", "single"])


def test_single_max_readout():
    _run_test("test_single_max_readout", ["--smoke", "--mode", "single", "--readout", "max"])


def test_twohead_joint():
    _run_test("test_twohead_joint", ["--smoke", "--mode", "twohead", "--schedule", "joint"])


def test_twohead_joint_binary():
    _run_test(
        "test_twohead_joint_binary",
        ["--smoke", "--mode", "twohead", "--schedule", "joint", "--event-classes", "2"],
    )


def test_twohead_int():
    _run_test("test_twohead_int", ["--smoke", "--mode", "twohead", "--schedule", "int"])


def test_twohead_intdetach():
    _run_test("test_twohead_intdetach", ["--smoke", "--mode", "twohead", "--schedule", "intdetach"])


def test_twohead_seqfull():
    _run_test("test_twohead_seqfull", ["--smoke", "--epochs", "2", "--mode", "twohead", "--schedule", "seqfull"])


def test_twohead_seqfrozen():
    _run_test(
        "test_twohead_seqfrozen",
        ["--smoke", "--epochs", "2", "--mode", "twohead", "--schedule", "seqfrozen"],
    )


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
                train_loss, class_auc = _RESULTS[name]
                print(f"{name}: PASS train_loss={train_loss:.6g} class_auc={class_auc:.6g}")
    if failures:
        raise SystemExit(1)
