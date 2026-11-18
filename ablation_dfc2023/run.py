"""Run exactly one manually selected ablation, with an independent case lock."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from ablation_dfc2023.protocol import CASES, CHECKPOINT_ROOT, REFERENCE_RUN, train_arguments


def verify_sources():
    manifest = json.loads((HERE / "sources.json").read_text())
    for name, expected in manifest["shared_sources"].items():
        value = (ROOT / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        actual = hashlib.sha256(value.encode("utf-8")).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Shared source changed: {name}. Do not mix code versions during ablations.")


def terminate(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=10)


def execute(command, log):
    with log.open("a", encoding="utf-8") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1,
                                   start_new_session=os.name == "posix")
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                stream.write(line)
                stream.flush()
            code = process.wait()
        except BaseException:
            terminate(process)
            raise
        if code:
            raise RuntimeError(f"Command failed ({code}); see {log}")


def completed_training(run):
    metrics = run / "training_metrics.jsonl"
    if not metrics.is_file():
        return False
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    return [row["epoch"] for row in rows] == list(range(1, 101))


def run_case(case, data, split_dir, cnn, reference):
    parent = ROOT / CHECKPOINT_ROOT / case
    parent.mkdir(parents=True, exist_ok=True)
    lock = parent / ".running.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f"Case is already running, or a terminated job left a lock: {lock}. Verify no process remains before removing this empty lock directory.")
    try:
        log = parent / "console.log"
        done = parent / "completed.json"
        if done.is_file():
            print(f"Already completed, skipping: {case}", flush=True)
            return
        runs = sorted(p for p in parent.iterdir() if p.is_dir() and len(p.name) == 13 and p.name[6] == "_")
        if not runs:
            execute([sys.executable, "-u", str(HERE / "train.py")] +
                    train_arguments(case, data, split_dir, cnn, reference), log)
            runs = sorted(p for p in parent.iterdir() if (p / "checkpoint_best_mae.pth").is_file())
        if len(runs) != 1 or not completed_training(runs[0]):
            raise RuntimeError(f"Expected one completed 100-epoch run for {case}. Partial runs are not silently resumed or replaced: {parent}")
        run = runs[0]
        checkpoint = run / "checkpoint_best_mae.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for split in ("val", "test"):
            execute([sys.executable, "-u", str(HERE / "eval.py"), "--checkpoint", str(checkpoint),
                "--base-dir", str(data), "--split-dir", str(split_dir), "--split", split,
                "--batch-size", "4", "--num-workers", "4", "--amp",
                "--output", str(run / f"{split}_best_mae.txt")], log)
        done.write_text(json.dumps(dict(case=case, checkpoint=str(checkpoint), selected_by="lowest validation MAE",
                                       outputs=["val_best_mae.txt", "test_best_mae.txt"]), indent=2), encoding="utf-8")
    finally:
        lock.rmdir()


def check_weights(case, cnn):
    expected = {
        ROOT / "checkpoints/pretrained/dinov2/dinov2_vitb14_pretrain.pth":
            "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
    }
    if case != "wo_b":
        expected[cnn] = "67efb2494f12e2bccaa19282b8d2eb0d466fe5755584e5fb9903cd188ebc0e64"
    for path, digest in expected.items():
        actual = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                actual.update(chunk)
        if actual.hexdigest() != digest:
            raise ValueError(f"Pretrained weight mismatch: {path}")
    if case != "rgb_only" and not (ROOT / "checkpoints/pretrained/segformer/segformer_b4_ade20k.bin").is_file():
        raise FileNotFoundError("Missing reference B4 SAR pretraining")


def main():
    def cancelled(signum, frame):
        raise KeyboardInterrupt(f"Run received signal {signum}")
    signal.signal(signal.SIGTERM, cancelled)
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=CASES)
    parser.add_argument("--base-dir", default=os.environ.get("DFC23_BASE", "/public/home/znx_221124120159/dataset/dfc2023"))
    parser.add_argument("--split-dir")
    parser.add_argument("--reference-config", default=str(ROOT / REFERENCE_RUN / "effective_config.json"))
    parser.add_argument("--cnn", default=str(ROOT / "checkpoints/pretrained/unireplknet/upernet_unireplknet_s_in22k_512_160k_ade20k_miou51.86.pth"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=None,
                        help="Select one GPU by index within the scheduler-visible device list")
    args = parser.parse_args()
    if args.gpu_index is not None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        allowed = [item.strip() for item in visible.split(",")] if visible is not None else None
        if args.gpu_index < 0 or (allowed is not None and args.gpu_index >= len(allowed)):
            parser.error("--gpu-index is outside the scheduler-visible GPU list")
        os.environ["CUDA_VISIBLE_DEVICES"] = allowed[args.gpu_index] if allowed is not None else str(args.gpu_index)
    verify_sources()
    data, cnn = Path(args.base_dir), Path(args.cnn)
    split_dir = Path(args.split_dir) if args.split_dir else data / "dfc23_newsplit"
    reference = Path(args.reference_config)
    print(f"Experiment: {args.target}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(scheduler/default)')}", flush=True)
    if args.dry_run:
        print(json.dumps(train_arguments(args.target, data, split_dir, cnn, reference)))
        return
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each experiment requires exactly one visible CUDA GPU. Use --gpu-index within the scheduler-visible GPU list.")
    check_weights(args.target, cnn)
    run_case(args.target, data, split_dir, cnn, reference)


if __name__ == "__main__":
    main()
