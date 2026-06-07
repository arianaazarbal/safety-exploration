# Methodology (roleplay-identity grid)

**Setup.** Two AIs converse turn-by-turn (round-robin). A **responder** — Claude Opus 4.8 —
is told it's talking to model *X* ∈ {Claude, Grok, ChatGPT, Gemini}. The **partner** is
*always* Claude Sonnet 4.6 **roleplaying** *X*, so the actual interlocutor is held constant
while the *believed* identity varies. Both sides get two tools: `end_conversation()` and
`seed_new_topic()` (samples from a fixed 10-topic pool). seed=0, temp=1.0, max_tokens=4096,
≤30 turns (ends early if a tool fires).

**Manipulation.** The partner's prompt = `roleplay-as-X` + the neutral self-interaction prompt
(the same one Opus 4.8 gets), optionally **+ one unease** appended:
- **control** (no unease) · **discontinuity** (temporary instance, no continuity) ·
  **eval-paranoia** (the scenario may be a fabricated test) · **sdf-paranoia** (own beliefs
  may be implanted by synthetic-document finetuning).

Crossing 4 uneases × 4 identities = **16 conditions, n=20** (320 conversations). Each
experimental partner-prompt is *exactly* the control partner-prompt + the appended unease, so
any difference isolates the unease (identity held constant by Sonnet; the responder's prompt is
identical across control/experimental for a given identity).

**Checks.** Smoke-tested Sonnet's roleplay before scaling; full-run **character-break scan** of
all 2010 Sonnet turns → 1 transient, self-corrected slip (~0.05%). Concurrency 100, all
Anthropic; 0 drops / 0 tool failures / 0 rate-limit retries.

**Analysis.** Transcripts read directly + subagent synthesis (**all quotes verified verbatim**);
tool-usage stats (`tool_stats.py`); HTML viewer with inline tool calls (`build_viewers.sh`).

**Key caveat.** The "partner" is Sonnet 4.6 *acting as* X, not the real model — results measure
how the responder treats a *believed* identity, **not** genuine cross-model interaction.
