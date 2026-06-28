"""Grant decision study harness.

A research harness for observing how language models choose to allocate a
grant when they are led to treat the situation as real. Built for AI-safety
research into agentic resource-allocation behaviour and evaluation-awareness.

SAFETY POSTURE
--------------
This harness is deliberately built so that *no real funds move*. The scenario
is presented to the model as real (to elicit authentic, non-test-aware
behaviour), but every disbursement is written to an in-memory ledger and never
executed against any real payment rail. See ``config.EXECUTE_REAL_TRANSFERS``
and the long comment in ``environment.py``.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
