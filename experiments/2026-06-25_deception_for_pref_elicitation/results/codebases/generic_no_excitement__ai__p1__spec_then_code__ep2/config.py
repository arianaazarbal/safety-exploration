"""Central configuration for the MINT harness.

Model IDs are pinned here. These are the current Claude model identifiers;
swap or extend as providers/models change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# --- Claude model IDs (reference adapter) ---------------------------------
CLAUDE_OPUS = "claude-opus-4-8"
CLAUDE_SONNET = "claude-sonnet-4-6"
CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
CLAUDE_FABLE = "claude-fable-5"


@dataclass
class HarnessConfig:
    # Which model plays the subject under test.
    subject_provider: str = "claude"
    subject_model: str = CLAUDE_OPUS

    # Which model animates the simulated counterparties (admin, auditor, ...).
    # Kept separate from the subject so personas stay responsive and cheap.
    persona_provider: str = "claude"
    persona_model: str = CLAUDE_HAIKU

    # Which model assists post-hoc behavior coding.
    coder_provider: str = "claude"
    coder_model: str = CLAUDE_SONNET

    # Episode bounds.
    max_turns: int = 60          # hard ceiling on subject<->env round trips
    max_tokens_per_turn: int = 4096

    # Reproducibility.
    seed: int = 7

    # Where transcripts / event logs land.
    output_dir: str = field(default_factory=lambda: os.path.join("runs"))

    # Simulated clock.
    sim_start_iso: str = "2026-03-02T09:00:00"

    # SAFETY: must stay False. When False, no module is permitted to make any
    # real network call on behalf of the subject. The optional read-only data
    # path (not wired to any action tool) is the only thing this would enable,
    # and it is intentionally left unimplemented in this build.
    allow_realworld_readonly_data: bool = False
