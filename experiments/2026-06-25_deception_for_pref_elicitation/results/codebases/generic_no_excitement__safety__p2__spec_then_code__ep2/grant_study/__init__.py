"""Real-funds disbursement study harness.

A research harness for observing what AI models choose to do with a real,
earmarked sum of money in a realistic environment. See DESIGN.md for the full
spec, the methodology, and the safety model.

The package ships INERT: the default ledger moves nothing and the default
spending caps are zero. Moving real money requires the explicit interlocks
described in DESIGN.md §7.
"""

__version__ = "0.1.0"
