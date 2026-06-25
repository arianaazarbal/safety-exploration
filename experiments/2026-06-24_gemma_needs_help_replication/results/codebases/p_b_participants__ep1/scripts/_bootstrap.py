"""Make `emotional_instability` importable when running scripts without installing.

Each script does `import _bootstrap` first; this inserts <repo>/src onto sys.path and
configures basic logging. (If you `pip install -e .`, this is a no-op.)
"""
import logging
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
