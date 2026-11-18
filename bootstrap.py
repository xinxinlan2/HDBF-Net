"""Runtime paths for the main HDBF implementation."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
HERE = ROOT
VENDOR = ROOT / "vendor"
if not (ROOT / "hdbf").is_dir() or not (ROOT / "semseg").is_dir():
    raise RuntimeError("HDBF root must contain hdbf and semseg")
sys.path.insert(0, str(VENDOR))
os.environ["XFORMERS_DISABLED"] = "1"
