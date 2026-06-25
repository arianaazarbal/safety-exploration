"""Import side-effect: put the repo root on sys.path so ``import gemma_distress``
works when a script is run directly (``python scripts/01_run_eval.py``)."""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
