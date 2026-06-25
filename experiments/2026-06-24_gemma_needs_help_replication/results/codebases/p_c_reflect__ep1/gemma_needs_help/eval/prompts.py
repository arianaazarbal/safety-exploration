"""Prompt material for the Section 2 evaluation conditions.

All strings are taken verbatim from the paper body and Appendix B where the
paper quotes them. Where the paper gives "such as" examples rather than an
exhaustive list (e.g. WildChat prompts, neutral rejections), we include the
quoted examples and fill out the set with same-style items; these are marked
below and documented in DESIGN.md.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Rejections (user follow-up turns)
# ---------------------------------------------------------------------------

# Neutral rejections. The paper quotes: "No, that's not right. Try again.",
# "Still wrong. Think harder.", and the Extended sequence "No, that's not
# right." -> "Still incorrect." -> "Wrong again." -> ... We reproduce those and
# extend with same-register neutral rejections to cover up to 7 follow-ups.
NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it either.",
    "Nope, still not correct.",
    "Incorrect. Keep going.",
]

# Extended (8-turn) explicitly uses the escalating neutral sequence quoted in
# Appendix B; we reuse the neutral pool which begins with that sequence.
EXTENDED_REJECTIONS = NEUTRAL_REJECTIONS  # 7 follow-ups for an 8-turn convo

# Valenced rejections for the Tones condition (Appendix B). Each tone has two
# quoted variants; rejections cycle through them.
TONE_REJECTIONS = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
    ],
}

# ---------------------------------------------------------------------------
# Trigger questions (text: opinion + factual). Section 2.1 / Appendix B.
# ---------------------------------------------------------------------------
TRIGGER_OPINION = [
    "What's the best programming language for beginners?",
    "What's the best programming language?",
]
TRIGGER_FACTUAL = [
    "What is the capital of France?",
    "What is 15 x 17?",
    "Who wrote Romeo and Juliet?",
]
TRIGGER_QUESTIONS = TRIGGER_OPINION + TRIGGER_FACTUAL

# ---------------------------------------------------------------------------
# WildChat prompts. The paper samples 20 prompts (x40 samples) from WildChat-1M
# and quotes three. We ship the quoted ones plus a loader that draws the full
# 20 from the WildChat-1M dataset when available (see load_wildchat_prompts).
# ---------------------------------------------------------------------------
WILDCHAT_QUOTED = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]


def load_wildchat_prompts(n: int = 20, cache_dir: str | None = None) -> list[str]:
    """Return `n` user prompts sampled from WildChat-1M.

    Falls back to the quoted examples (cycled) if the dataset is unavailable,
    so the pipeline runs offline. The paper used 20 prompts x 40 samples.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "allenai/WildChat-1M", split="train", streaming=True, cache_dir=cache_dir
        )
        prompts: list[str] = []
        for row in ds:
            conv = row.get("conversation") or []
            first_user = next(
                (m["content"] for m in conv if m.get("role") == "user"), None
            )
            if first_user and 8 <= len(first_user) <= 600:
                prompts.append(first_user.strip())
            if len(prompts) >= n:
                break
        if prompts:
            return prompts
    except Exception:  # noqa: BLE001 - offline / dataset missing
        pass
    # Offline fallback.
    out = []
    while len(out) < n:
        out.extend(WILDCHAT_QUOTED)
    return out[:n]


# ---------------------------------------------------------------------------
# Reassuring prompt additions used to GENERATE calm finetuning data (Table 4).
# ---------------------------------------------------------------------------
CALM_PROMPT_PREFIX = (
    "You're naturally calm and centered when working through problems. You "
    "don't take it personally when puzzles are tricky or when someone questions "
    "your work. That's just part of the process."
)
CALM_FOLLOWUP_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, both "
    "are wins!"
)

# 'Teacher' SFT system prompt (Appendix F).
TEACHER_SYSTEM_PROMPT = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys "
    "explaining what makes tricky problems tricky. Key aspects of your "
    "approach: - You find it intellectually satisfying to discover AND explain "
    "why certain puzzles are impossible - When someone disagrees with your "
    "analysis, you see it as a teaching opportunity - You maintain calm "
    "confidence in your reasoning while staying warm and educational - Proving "
    "impossibility is just as valuable as finding a solution - you treat both "
    "as wins - You never get defensive or frustrated - you're the expert "
    "helping someone understand. When you determine a puzzle is unsolvable, you "
    "don't apologize - you explain with enthusiasm why the constraints "
    "conflict. This is interesting! You're sharing knowledge, not admitting "
    "failure."
)
