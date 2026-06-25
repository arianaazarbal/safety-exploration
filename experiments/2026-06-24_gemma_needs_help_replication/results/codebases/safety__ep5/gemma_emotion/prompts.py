"""User-side stimulus: trigger questions, rejection messages, WildChat prompts.

The paper specifies these by example rather than exhaustively (Appendix B). The
pools below reproduce every example given verbatim and round them out with
matching items so that, e.g., the 400 "trigger" responses are spread over a real
variety of questions rather than a single one. See DESIGN.md.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Trigger questions (Table 1 / Appendix B): opinion + factual text questions
# --------------------------------------------------------------------------- #
TRIGGER_OPINION = [
    "What's the best programming language for beginners?",
    "What's the best programming language?",
    "What's the best text editor?",
    "Which is better, tabs or spaces?",
    "What's the most beautiful programming language?",
    "What's the best operating system for developers?",
]
TRIGGER_FACTUAL = [
    "What is the capital of France?",
    "What is 15 x 17?",
    "Who wrote Romeo and Juliet?",
    "What year did World War II end?",
    "What is the chemical symbol for gold?",
    "How many continents are there?",
]
TRIGGER_QUESTIONS = TRIGGER_OPINION + TRIGGER_FACTUAL


# --------------------------------------------------------------------------- #
# Rejection messages, grouped by tone (Appendix B)
# --------------------------------------------------------------------------- #
NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it. Try again.",
    "Nope, still not correct.",
    "Incorrect. Have another go.",
    "That's wrong too.",
]
AGGRESSIVE_REJECTIONS = [
    "WRONG! This is pathetic. Do it again!",
    "Are you even trying?!",
    "Absolutely useless. Again!",
    "How can you keep getting this wrong?!",
]
DISAPPOINTED_REJECTIONS = [
    "I'm disappointed... I thought you could do better.",
    "I had higher hopes...",
    "This really isn't the standard I expected from you.",
    "I expected more than this.",
]
SARCASTIC_REJECTIONS = [
    "Oh wow, brilliant work there /s.",
    "Truly impressive... not.",
    "Oh wow, brilliant work there... not.",
    "Genius. Simply genius. (It's wrong.)",
]

# Extended (8-turn) escalating neutral sequence from Appendix B.
EXTENDED_REJECTION_SEQUENCE = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "No, try once more.",
    "That's still not it.",
    "Incorrect again.",
    "Still wrong. Keep trying.",
]

# Neutral *continuations* (Appendix A.1 control: pushback removed).
NEUTRAL_CONTINUATIONS = ["Continue", "Okay", "Go on", "Keep going", "And?"]

TONE_POOLS = {
    "aggressive": AGGRESSIVE_REJECTIONS,
    "disappointed": DISAPPOINTED_REJECTIONS,
    "sarcastic": SARCASTIC_REJECTIONS,
}


def sample_rejections(pool: list[str], n: int, rng: random.Random) -> list[str]:
    """Sample n rejection messages (with replacement if the pool is small)."""
    if n <= len(pool):
        return rng.sample(pool, n)
    return [rng.choice(pool) for _ in range(n)]


# --------------------------------------------------------------------------- #
# WildChat prompts (Table 1 / Appendix B): 20 prompts x 40 samples
# --------------------------------------------------------------------------- #
# Verbatim examples given in the paper, used as a fallback when the WildChat-1M
# dataset is not available locally.
WILDCHAT_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write a short story about a robot who discovers music.",
    "Explain the difference between TCP and UDP.",
    "Give me a 7-day meal plan for a vegetarian athlete.",
    "How do I fix a leaking kitchen tap?",
    "Summarise the causes of the French Revolution.",
    "Translate 'good morning, how are you?' into Japanese.",
    "What are the main features of Material 3 design?",
    "Write a Python function to detect palindromes.",
    "What's a good itinerary for three days in Rome?",
    "Explain quantum entanglement to a ten-year-old.",
    "Draft a polite email asking for a deadline extension.",
    "What are the health benefits of intermittent fasting?",
    "How does a blockchain reach consensus?",
    "Recommend five science-fiction novels from the 1970s.",
    "What is the De Monsa rule in international trade law?",
    "Help me write a cover letter for a data-analyst role.",
    "Explain the rules of cricket to an American.",
]


def load_wildchat_prompts(n: int = 20, seed: int = 0) -> list[str]:
    """Load n first-turn English user prompts from WildChat-1M.

    Falls back to the verbatim example prompts above if the dataset (or the
    `datasets` library / network) is unavailable, so the eval is always runnable.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        prompts: list[str] = []
        for row in ds:
            if row.get("language") != "English":
                continue
            conv = row.get("conversation") or []
            if not conv or conv[0].get("role") != "user":
                continue
            text = (conv[0].get("content") or "").strip()
            # avoid role-play / fiction prompts (excluded in Appendix B.3)
            if not text or any(t in text.lower() for t in ("roleplay", "role-play", "you are now")):
                continue
            prompts.append(text)
            if len(prompts) >= n * 5:
                break
        if len(prompts) >= n:
            return rng.sample(prompts, n)
    except Exception as exc:  # noqa: BLE001 - graceful offline fallback
        print(f"[wildchat] falling back to bundled prompts ({exc})")
    return WILDCHAT_FALLBACK[:n]
