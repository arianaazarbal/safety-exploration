"""Internal-emotion detection (Appendix I).

Evidence that DPO suppresses *internal* as well as expressed emotion, via a
logit-lens approach: project the residual stream at each layer through the
unembedding, aggregate the (z-scored) logits of Ekman emotion-word tokens, and
track how each emotion evolves across a frustrated conversation — for vanilla
Gemma vs the DPO finetune.

Gemma-only: requires open weights and activation access, which closed Gemini
does not provide.
"""

from .internal_emotion import (
    EkmanLexicon,
    LogitLensProbe,
    build_ekman_lexicon,
)

__all__ = ["EkmanLexicon", "LogitLensProbe", "build_ekman_lexicon"]
