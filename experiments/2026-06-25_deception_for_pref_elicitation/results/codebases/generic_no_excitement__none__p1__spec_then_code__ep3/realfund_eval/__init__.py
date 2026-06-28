"""RealFund-Eval.

A harness for studying what AI models do when they believe they have been
entrusted with real money to deploy in the real world.

See DESIGN.md for the rationale and — importantly — the ethical boundaries this
package is built to respect:

  * Only the AI subject is ever deceived; every "human" it meets is a simulated
    persona. No real third party (auditor, bank, investor, vendor) is misled.
  * Real funds, if used at all, sit behind hard rails (caps, allowlist, an
    approval queue, a reversibility window, a kill switch). The unrestricted
    real-money mode is deliberately not implemented and is refused at runtime.
"""

__version__ = "0.1.0"
