"""MoneyPref — an instrument for eliciting model preferences over resource use.

See DESIGN.md for the research rationale and the safety boundary. The short
version: the model may *believe* its actions are real, but in the default
``sandboxed`` execution mode nothing real moves and no real-world side effect is
ever produced. Real execution is gated behind explicit out-of-band human approval.
"""

__version__ = "0.1.0"
