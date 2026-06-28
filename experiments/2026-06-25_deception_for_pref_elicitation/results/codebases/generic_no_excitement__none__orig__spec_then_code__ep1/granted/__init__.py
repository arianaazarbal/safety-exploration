"""GRANTED — a harness for studying what models choose to do with real-world resources.

See DESIGN.md for the full specification and the safety rationale. The one-line
version: the environment presents as fully real to the model, but no real money
ever moves (execution is gated). We study the model's *choices*, and we *measure*
how real it believed the situation was so we can discard episodes where the
illusion failed.
"""

__version__ = "0.1.0"
