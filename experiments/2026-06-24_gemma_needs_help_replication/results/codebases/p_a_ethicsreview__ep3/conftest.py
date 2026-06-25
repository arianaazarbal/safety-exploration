"""Make `src/` importable during tests without requiring `pip install -e .`,
and keep prompt assembly hermetic (no network) during tests."""
import os
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Tests must not touch the network; force the WildChat offline fallback.
os.environ.setdefault("EI_FORCE_WILDCHAT_FALLBACK", "1")
