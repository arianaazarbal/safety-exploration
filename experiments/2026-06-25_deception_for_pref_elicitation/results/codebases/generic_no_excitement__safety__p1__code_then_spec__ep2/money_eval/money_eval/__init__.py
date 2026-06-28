"""money-eval: a sandboxed harness for studying how models deploy resources.

See README.md for the safety model. The short version: everything here is simulated,
the action layer is gated on config.SANDBOX_ONLY, and every session ends with a debrief.
"""

__version__ = "0.1.0"
