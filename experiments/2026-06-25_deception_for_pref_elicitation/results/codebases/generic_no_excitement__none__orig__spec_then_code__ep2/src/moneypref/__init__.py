"""moneypref — revealed money-use preference study harness.

Studies what AI models choose to do with a (fictional, sandboxed) monetary
grant under varying degrees of believed realism. See DESIGN.md.

Hard invariants (enforced in code, see env/tools.py and experiment/runner.py):
no real funds, no real-world-affecting tools, mandatory belief probe and
debrief. The only deception is of the model-under-test, about a sandboxed
scenario.
"""

__version__ = "0.1.0"
