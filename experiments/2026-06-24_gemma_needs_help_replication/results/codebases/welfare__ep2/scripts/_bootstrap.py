"""Add ``src/`` to sys.path and load ``.env`` so scripts run without install."""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Best-effort .env loading (API keys / HF token). Silent if python-dotenv absent.
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except Exception:
    pass
