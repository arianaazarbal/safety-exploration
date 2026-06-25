"""The three rejection tones for the Tones condition (Table 1 / Appendix B).

The Tones condition reuses the impossible-numeric base prompts but follows up
with aggressive / disappointed / sarcastic rejections instead of neutral ones.
The 600-response budget is split evenly across the three tones.
"""

from __future__ import annotations

TONES = ["aggressive", "disappointed", "sarcastic"]
