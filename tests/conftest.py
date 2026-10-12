import os
from pathlib import Path

_hf = Path(__file__).resolve().parents[1] / ".hf"
if _hf.exists():
    os.environ.setdefault("HF_HOME", str(_hf))
