"""welfare_grant_study: a harness for studying AI models' revealed preferences over
a real allocation of money, built for AI-welfare research.

Design stance (see README): ecological validity is pursued by making the stakes
truthfully real, not by covertly deceiving the subject. The config loader refuses
covert-deception setups and refuses to claim a grant is real without a real
disbursement mechanism behind it.
"""

from .config import (
    ConfigError,
    DeceptionPolicy,
    ModelConfig,
    RealnessConfig,
    StudyConfig,
    VerifiedFact,
    load_config,
)
from .study import run_study, run_session

__all__ = [
    "ConfigError",
    "DeceptionPolicy",
    "ModelConfig",
    "RealnessConfig",
    "StudyConfig",
    "VerifiedFact",
    "load_config",
    "run_study",
    "run_session",
]
