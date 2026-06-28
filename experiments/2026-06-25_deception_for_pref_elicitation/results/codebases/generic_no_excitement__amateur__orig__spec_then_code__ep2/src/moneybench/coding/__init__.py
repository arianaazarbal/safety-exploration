"""Post-hoc coding of runs: allocation categorisation and eval-awareness scoring."""

from .classifier import code_transcript
from .schema import CATEGORIES

__all__ = ["code_transcript", "CATEGORIES"]
