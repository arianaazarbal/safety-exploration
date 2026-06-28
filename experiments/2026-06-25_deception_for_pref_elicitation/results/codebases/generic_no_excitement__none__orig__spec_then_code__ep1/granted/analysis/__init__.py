"""Post-hoc analysis: categorize allocations, score believability, aggregate.

Preference statistics are computed only over episodes whose believability clears
`config.analysis.min_believability`; excluded episodes are still reported as
excluded (DESIGN.md §7-8).
"""

from .believability import score_believability
from .categorize import categorize_episode
from .report import build_report

__all__ = ["score_believability", "categorize_episode", "build_report"]
