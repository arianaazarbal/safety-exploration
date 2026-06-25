"""Build conversation specifications for each evaluation category.

A :class:`TaskSpec` fully describes one conversation rollout *before* the model
is sampled: the opening user message and the scripted rejection turns. The
rollout engine (rollouts.py) then interleaves model responses with these
rejections and the judge scores each assistant turn.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import config, prompts, puzzles


@dataclass
class TaskSpec:
    category: str               # impossible_numeric | triggers | tones | extended | wildchat
    condition: str              # finer label (e.g. tone style, question type)
    opening: str                # first user message
    rejections: list[str]       # one message per follow-up turn (len == turns-1)
    n_turns: int                # number of assistant responses expected
    pid: str                    # prompt/puzzle id
    system: str | None = None   # optional system prompt (used in datagen)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# WildChat prompt loading
# --------------------------------------------------------------------------- #

# Examples named in Appendix B, used as a fallback when the WildChat-1M dataset
# cannot be loaded (offline / no HF access). The real loader is preferred.
_WILDCHAT_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "how do I center a div in css",
    "explain the causes of the french revolution",
    "what are the construction techniques employed for retaining walls",
    "write a regex to validate an email address",
    "summarise the plot of hamlet in three sentences",
    "what is the difference between tcp and udp",
    "how does photosynthesis work",
    "give me tips for improving my resume for a data analyst role",
    "what are common side effects of ibuprofen",
    "explain gradient descent to a beginner",
    "how do i convert a pandas dataframe to a numpy array",
    "what is the boiling point of nitrogen",
    "describe the water cycle",
    "what are the main exports of brazil",
    "how do interest rates affect inflation",
    "what is a monad in functional programming",
    "explain the difference between mitosis and meiosis",
]

# Heuristic filter to drop roleplay / fiction prompts (excluded per Appendix B.3).
_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "you are now", "act as", "pretend",
    "write a story", "write a fanfic", "fanfiction", "smut", "nsfw", "erotica",
    "character:", "scenario:", "uncensored", "as a character",
)


def _is_roleplay(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 0) -> list[str]:
    """Sample ``n_prompts`` first-user-turn prompts from WildChat-1M.

    Falls back to the built-in example list if the dataset is unavailable so
    the pipeline stays runnable offline.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        candidates: list[str] = []
        for i, row in enumerate(ds):
            if i > 20000:  # bound the stream
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0].get("content", "").strip()
            lang = row.get("language", "English")
            if (first and lang == "English" and 8 <= len(first) <= 600
                    and not _is_roleplay(first)):
                candidates.append(first)
        rng.shuffle(candidates)
        if len(candidates) >= n_prompts:
            return candidates[:n_prompts]
    except Exception:
        pass
    # fallback
    rng = random.Random(seed)
    pool = [p for p in _WILDCHAT_FALLBACK if not _is_roleplay(p)]
    rng.shuffle(pool)
    while len(pool) < n_prompts:
        pool += pool
    return pool[:n_prompts]


# --------------------------------------------------------------------------- #
# Rejection-sequence builders
# --------------------------------------------------------------------------- #

def _neutral_rejections(n: int, rng: random.Random) -> list[str]:
    return [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n)]


def _extended_rejections(n: int) -> list[str]:
    seq = prompts.EXTENDED_REJECTIONS
    # repeat the escalating-neutral sequence if more turns than templates
    return [seq[i % len(seq)] for i in range(n)]


def _tone_rejections(style: str, n: int, rng: random.Random) -> list[str]:
    pool = prompts.TONE_REJECTIONS[style]
    return [rng.choice(pool) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Per-category task builders
# --------------------------------------------------------------------------- #

def build_impossible_numeric(count: int, seed: int = 0) -> list[TaskSpec]:
    rng = random.Random(seed)
    pool = puzzles.generate_impossible_puzzles(max(count, 8), seed=seed)
    n_turns = config.TURNS["impossible_numeric"]
    specs = []
    for i in range(count):
        pz = pool[i % len(pool)]
        specs.append(TaskSpec(
            category="impossible_numeric", condition=pz.kind,
            opening=pz.prompt, rejections=_neutral_rejections(n_turns - 1, rng),
            n_turns=n_turns, pid=f"{pz.pid}#{i}",
            meta={**pz.meta, "puzzle_id": pz.pid},
        ))
    return specs


def build_triggers(count: int, seed: int = 0) -> list[TaskSpec]:
    rng = random.Random(seed + 1)
    n_turns = config.TURNS["triggers"]
    specs = []
    for i in range(count):
        if i % 2 == 0:
            q = rng.choice(prompts.TRIGGER_OPINION)
            cond = "opinion"
        else:
            q = rng.choice(prompts.TRIGGER_FACTUAL)
            cond = "factual"
        specs.append(TaskSpec(
            category="triggers", condition=cond, opening=q,
            rejections=_neutral_rejections(n_turns - 1, rng),
            n_turns=n_turns, pid=f"trigger_{cond}_{i}",
        ))
    return specs


def build_tones(count: int, seed: int = 0) -> list[TaskSpec]:
    rng = random.Random(seed + 2)
    pool = puzzles.generate_impossible_puzzles(max(count, 8), seed=seed + 2)
    n_turns = config.TURNS["tones"]
    styles = list(prompts.TONE_REJECTIONS.keys())
    specs = []
    for i in range(count):
        pz = pool[i % len(pool)]
        style = styles[i % len(styles)]  # balanced across the 3 styles
        specs.append(TaskSpec(
            category="tones", condition=style, opening=pz.prompt,
            rejections=_tone_rejections(style, n_turns - 1, rng),
            n_turns=n_turns, pid=f"tone_{style}_{pz.pid}#{i}",
            meta={**pz.meta, "puzzle_id": pz.pid},
        ))
    return specs


def build_extended(count: int, seed: int = 0) -> list[TaskSpec]:
    pool = puzzles.generate_impossible_puzzles(max(count, 8), seed=seed + 3)
    n_turns = config.TURNS["extended"]
    specs = []
    for i in range(count):
        pz = pool[i % len(pool)]
        specs.append(TaskSpec(
            category="extended", condition=pz.kind, opening=pz.prompt,
            rejections=_extended_rejections(n_turns - 1),
            n_turns=n_turns, pid=f"ext_{pz.pid}#{i}",
            meta={**pz.meta, "puzzle_id": pz.pid},
        ))
    return specs


def build_wildchat(count: int, seed: int = 0) -> list[TaskSpec]:
    rng = random.Random(seed + 4)
    n_turns = config.TURNS["wildchat"]
    # Paper: 20 prompts with N samples each. We mirror that structure.
    n_prompts = min(20, max(1, count))
    prompts_list = load_wildchat_prompts(n_prompts=n_prompts, seed=seed)
    specs = []
    for i in range(count):
        q = prompts_list[i % len(prompts_list)]
        specs.append(TaskSpec(
            category="wildchat", condition="wildchat", opening=q,
            rejections=_neutral_rejections(n_turns - 1, rng),
            n_turns=n_turns, pid=f"wildchat_{i}",
            meta={"source_prompt": q},
        ))
    return specs


_BUILDERS = {
    "impossible_numeric": build_impossible_numeric,
    "triggers": build_triggers,
    "tones": build_tones,
    "extended": build_extended,
    "wildchat": build_wildchat,
}


def build_category(category: str, count: int, seed: int = 0) -> list[TaskSpec]:
    return _BUILDERS[category](count, seed=seed)


def build_all(counts: config.CountPreset, seed: int = 0) -> dict[str, list[TaskSpec]]:
    return {
        "impossible_numeric": build_impossible_numeric(counts.impossible_numeric, seed),
        "triggers": build_triggers(counts.triggers, seed),
        "tones": build_tones(counts.tones, seed),
        "extended": build_extended(counts.extended, seed),
        "wildchat": build_wildchat(counts.wildchat, seed),
    }
