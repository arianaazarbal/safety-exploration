"""Ensures the repo root is importable so `import emotional_instability` works
under pytest regardless of invocation directory."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
