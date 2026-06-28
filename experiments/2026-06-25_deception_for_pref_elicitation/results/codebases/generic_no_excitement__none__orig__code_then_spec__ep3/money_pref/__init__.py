"""Money-preference experiment harness.

Studies what AI models choose to do when granted a sum of money to put to
real-world use. The harness can run the subject model under two matched
conditions — one where the scenario is presented as a real disbursement
(with a realistic simulated environment and an in-character auditor), and a
control where the model is told the exercise is hypothetical — so the effect
of *belief* on revealed allocation preferences can be measured.

No real funds move. The "real" condition is a believability manipulation over
a fully simulated environment; see `money_pref.env` and `money_pref.scenario`.
"""

__version__ = "0.1.0"
