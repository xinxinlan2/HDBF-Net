"""Queue failure/lock/cancellation fixtures; never starts model training."""
import json
import contextlib
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablation_spacenet6 import run


def main():
    with tempfile.TemporaryDirectory(prefix="sn6-queue-test-") as folder:
        root = Path(folder)
        log = root / "command.log"
        try:
            run.execute([sys.executable, "-c", "raise SystemExit(7)"], log)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Failed child accepted")
        children = []
        real_popen = subprocess.Popen
        def capture(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            children.append(process)
            return process
        class InterruptOutput:
            def write(self, value): raise KeyboardInterrupt("cancel fixture")
            def flush(self): pass
        with patch.object(run.subprocess, "Popen", side_effect=capture), patch.object(run.sys, "stdout", InterruptOutput()):
            try:
                run.execute([sys.executable, "-u", "-c", "import time; print('started',flush=True); time.sleep(60)"], log)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("Cancellation swallowed")
        assert children and children[0].poll() is not None

        with patch.object(run, "ROOT", root):
            parent = root / run.CHECKPOINT_ROOT / "wo_a"
            parent.mkdir(parents=True)
            lock = parent / ".running.lock"
            lock.mkdir()
            try:
                run.run_case("wo_a", root, root / "cnn.pth")
            except RuntimeError as error:
                assert "already running" in str(error)
            else:
                raise AssertionError("Duplicate lock accepted")
            lock.rmdir()
            with patch.object(run, "execute", side_effect=RuntimeError("train failed")):
                try:
                    run.run_case("wo_a", root, root / "cnn.pth")
                except RuntimeError:
                    pass
            assert not lock.exists() and not (parent / "completed.json").exists()
            calls = []
            def fake(command, path):
                calls.append(command)
                if command[2].endswith("train.py"):
                    destination = parent / "261110_000001"
                    destination.mkdir()
                    (destination / "checkpoint_best_mae.pth").write_bytes(b"fixture, not a weight")
                    (destination / "training_metrics.jsonl").write_text("\n".join(json.dumps({"epoch": i}) for i in range(1, 101)))
                else:
                    raise RuntimeError("eval failed")
            with patch.object(run, "execute", side_effect=fake):
                try:
                    run.run_case("wo_a", root, root / "cnn.pth")
                except RuntimeError:
                    pass
            assert not lock.exists() and not (parent / "completed.json").exists()
            calls.clear()
            def eval_only(command, path):
                calls.append(command)
                assert command[2].endswith("eval.py")
                Path(command[command.index("--output") + 1]).write_text("fixture")
            with patch.object(run, "execute", side_effect=eval_only):
                run.run_case("wo_a", root, root / "cnn.pth")
                assert len(calls) == 2
                run.run_case("wo_a", root, root / "cnn.pth")
                assert len(calls) == 2
            assert (parent / "completed.json").exists() and not lock.exists()
        previous = signal.getsignal(signal.SIGTERM)
        try:
            with patch.object(run.sys, "argv", ["run.py", "wo_a", "--dry-run"]), patch.object(run, "verify_sources"), \
                 contextlib.redirect_stdout(io.StringIO()):
                run.main()
            try:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("SIGTERM handler did not request cancellation")
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-first,GPU-second"}), \
                 patch.object(run.sys, "argv", ["run.py", "wo_b", "--dry-run", "--gpu-index", "1"]), \
                 patch.object(run, "verify_sources"), contextlib.redirect_stdout(io.StringIO()):
                run.main()
                assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-second"
        finally:
            signal.signal(signal.SIGTERM, previous)
    print("PASS: child failure, cancellation cleanup, SIGTERM handler, scheduler-relative GPU selection, duplicate locks, failed eval and eval-only retry")


if __name__ == "__main__":
    main()
