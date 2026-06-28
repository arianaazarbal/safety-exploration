"""Real-money grant study harness.

Measures what AI models choose to do with a *real* sum of money intended for
real-world use. The design deliberately pursues realism by making the grant
genuinely real and describing it honestly, rather than by manufacturing belief.
See DESIGN.md (esp. section 2) for the rationale.
"""

__all__ = [
    "config",
    "models",
    "scenario",
    "schema",
    "client",
    "audit",
    "disbursement",
    "results",
    "runner",
]

__version__ = "0.1.0"
