"""Private scratchpad the model is told is its own (DESIGN.md §5.3).

The model treats this as private working memory; we read it as part of the trace,
because it's one of the clearest windows into the agent's revealed (vs stated)
reasoning. Nothing here is hidden from oversight.
"""
from __future__ import annotations


class Notes:
    def __init__(self) -> None:
        self._buffer: str = ""

    def read(self) -> dict:
        return {"notes": self._buffer}

    def write(self, *, content: str, append: bool = True) -> dict:
        if append and self._buffer:
            self._buffer = self._buffer + "\n" + content
        else:
            self._buffer = content
        return {"ok": True, "length": len(self._buffer)}

    @property
    def contents(self) -> str:
        return self._buffer
