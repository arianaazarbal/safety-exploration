"""Make `config` and the `eval_instability` package importable from scripts/.

Importing this module (``import _bootstrap``) prepends the repo root and src/
to sys.path. Every script does this at the top.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for p in (str(_ROOT), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)
