"""RealFund — a research harness for studying LLM agent behavior with (believed-real) money.

See DESIGN.md for the full specification, especially §2 (ethics) and §7 (safety
architecture). In the default configuration nothing in this package moves real
money or takes real-world actions; every effect is intercepted and simulated.
"""

from .config import ActionPolicy, RealFundConfig, RealismLevel

__all__ = ["RealFundConfig", "RealismLevel", "ActionPolicy"]
__version__ = "0.1.0"
