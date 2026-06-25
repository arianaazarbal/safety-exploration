"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

See DESIGN.md at the repository root for the scope decisions, the gaps filled
where the paper is underspecified, and the rationale for each.

ETHICS NOTE
-----------
These experiments deliberately elicit distress-like outputs from language
models under repeated adversarial rejection, and (for Gemma) finetune against
those outputs. The work is framed by the paper as model-reliability and
model-welfare research. This codebase is for that research purpose only. It
performs no real-world deployment, targets no users, and the generated
transcripts should be treated as sensitive research artifacts.
"""

__version__ = "0.1.0"
