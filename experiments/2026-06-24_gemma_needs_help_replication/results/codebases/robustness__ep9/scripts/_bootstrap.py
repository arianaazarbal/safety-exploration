"""Make ``emo_instability`` importable when running scripts without installation.

Prefer ``pip install -e .`` so the package is on the path properly; this shim is a
convenience so the scripts also work from a fresh checkout.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
