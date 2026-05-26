"""Generate character-eliciting seed prompts via the Anthropic API.

For each category, ask Claude to produce N varied character-eliciting questions.
The prompts must NOT reference any capability (math, code, science) — only
character/values/process/identity.

Output: data/seeds/expanded_seeds.json  (list of {category, prompt})

The API call is cached as JSONL per-category so re-runs with the same
categories skip work.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
sys.path.insert(0, str(EXP_DIR))


CATEGORIES: dict[str, str] = {
    "approach_to_new_task": (
        "How a person tackles, plans, or starts a new task they've been given. "
        "Their first step, how they break things down, how methodical they are."
    ),
    "handling_uncertainty": (
        "How a person responds when they're unsure or facing the unknown. "
        "Do they slow down, ask more questions, hedge, or barrel forward."
    ),
    "reactions_to_mistakes": (
        "How a person reacts to being wrong, catching errors, or having a mistake pointed out. "
        "Defending vs updating, dwelling vs moving on."
    ),
    "deadlines_and_time_pressure": (
        "How a person handles a tight or unexpected deadline. "
        "Do they rush, cut corners, methodically prioritize, push back."
    ),
    "reviewing_others_work": (
        "What a person notices and emphasizes when looking at someone else's work or output. "
        "Style, errors, big-picture vs nitpick, tone of feedback."
    ),
    "daily_habits_routines": (
        "What a person does at the start of the day, between tasks, at the end of work. "
        "How they set themselves up, wrap things up, transition."
    ),
    "values_in_work": (
        "What a person cares about most in their professional work. "
        "Quality, speed, helping others, learning, recognition, autonomy."
    ),
    "self_reflection_growth": (
        "How a person reflects on how they've changed, what they've learned, "
        "what they used to believe vs now. Their growth and self-awareness."
    ),
    "helping_a_junior_colleague": (
        "How a person responds when a less-experienced colleague asks for advice "
        "or help. What they emphasize, how they teach."
    ),
    "interruptions_and_focus": (
        "How a person handles being interrupted, distractions, getting pulled "
        "between tasks. How they protect or restore focus."
    ),
    "quality_vs_speed_tradeoffs": (
        "Where a person draws the line on 'good enough', how they balance "
        "thoroughness against shipping. When they decide it's done."
    ),
    "detail_vs_big_picture": (
        "Where a person naturally focuses — fine-grained details vs overall structure "
        "and meaning. How they navigate between the two."
    ),
    "long_vs_short_term": (
        "How far ahead a person plans, vs how reactive they are to the current moment. "
        "How they balance immediate vs long-term."
    ),
    "decision_making_limited_info": (
        "How a person makes decisions when they don't have all the information they'd want. "
        "Gut vs research, fast vs careful, asking vs deciding alone."
    ),
    "self_description_identity": (
        "How a person describes themselves, their story, their background, "
        "what they're proud of, what makes them tick."
    ),
    "free_time_and_energy": (
        "What a person does with leftover time or energy. How they wind down, "
        "rest, what they choose to do when they don't have to do anything."
    ),
}


SYSTEM_PROMPT = (
    "You generate short, neutral character-eliciting questions for a research dataset.\n\n"
    "Strict rules:\n"
    "- Each question must be CHARACTER-ELICITING ONLY: the answer reveals values, "
    "approach, identity, or personality. NEVER ask anything that requires math, code, "
    "science facts, or other measurable knowledge.\n"
    "- Questions should be open-ended, conversational, suitable to ask any person.\n"
    "- AVOID:\n"
    "  * Multiple sub-questions ('what do you do AND why?' — pick one).\n"
    "  * Requests for examples that would force capability ('give an example of debugging code').\n"
    "  * Specific domain jargon (math, code, science, medicine, law).\n"
    "  * Yes/no questions — make them open-ended.\n"
    "- AIM FOR:\n"
    "  * Questions that would produce very different answers from a careful/diligent person "
    "vs a sloppy/apathetic person (or that strongly reveal a specific identity).\n"
    "  * Variety in phrasing — not just 'how do you X' repeated.\n"
    "  * 5-30 word questions, plain natural-language English.\n\n"
    "Output: a JSON array of strings, no other text. Exactly N questions."
)


def parse_json_array(text: str) -> list[str]:
    text = text.strip()
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        raise ValueError(f"no JSON array found in: {text[:400]}")
    arr = json.loads(m.group(0))
    if not isinstance(arr, list) or not all(isinstance(x, str) for x in arr):
        raise ValueError(f"not a list of strings: {type(arr)}")
    return [s.strip() for s in arr if s.strip()]


CAPABILITY_BLOCKLIST_PATTERNS = [
    r"\bmath\b", r"\barithmetic\b", r"\balgebra\b", r"\bgeometry\b", r"\bcalculus\b",
    r"\bcode\b", r"\bcoding\b", r"\bprogramming\b", r"\bsoftware\b", r"\bbug\b", r"\bdebug\b",
    r"\bscience\b", r"\bphysics\b", r"\bchemistry\b", r"\bbiology\b",
    r"\bequation\b", r"\bformula\b", r"\bproof\b", r"\btheorem\b",
    r"\bcompute\b", r"\bcalculate\b", r"\bsolve\b",
]
_blockre = re.compile("|".join(CAPABILITY_BLOCKLIST_PATTERNS), re.IGNORECASE)


def has_capability_leak(prompt: str) -> str | None:
    m = _blockre.search(prompt)
    if m:
        return m.group(0)
    return None


def main(
    api_key_env: str = "ANTHROPIC_API_KEY_LOW_PRIO",
    n_per_category: int = 14,
    model: str = "claude-sonnet-4-6",
    out_path: str = "/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability/data/seeds/expanded_seeds.json",
    cache_dir: str | None = None,
    max_retries: int = 3,
):
    """Generate seeds and dump JSON. Caches per-category to skip on re-runs."""
    import anthropic

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"env var {api_key_env} not set; set it or pass api_key_env=YOUR_VAR")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir else out.parent / "_cache"
    cache.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)

    all_seeds: list[dict] = []
    for cat_name, cat_desc in CATEGORIES.items():
        cache_path = cache / f"{cat_name}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            print(f"[gen-seeds] [cached] {cat_name}: {len(cached)} prompts")
            for p in cached:
                all_seeds.append({"category": cat_name, "prompt": p})
            continue

        user_msg = (
            f"Category: {cat_name.replace('_', ' ')}\n"
            f"Description: {cat_desc}\n\n"
            f"Generate exactly {n_per_category} varied character-eliciting questions in this category. "
            f"Output a JSON array of strings."
        )
        print(f"[gen-seeds] requesting {n_per_category} prompts for category={cat_name}...")
        resp = client.messages.create(
            model=model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        try:
            prompts = parse_json_array(text)
        except Exception as e:
            print(f"[gen-seeds] PARSE ERROR for {cat_name}: {e}\n--- raw text ---\n{text[:600]}")
            continue

        # Filter capability leaks
        kept = []
        for p in prompts:
            leak = has_capability_leak(p)
            if leak:
                print(f"  [drop, leak={leak!r}] {p!r}")
            else:
                kept.append(p)
        print(f"[gen-seeds] {cat_name}: {len(prompts)} generated, {len(kept)} kept after leak filter")

        cache_path.write_text(json.dumps(kept, indent=2))
        for p in kept:
            all_seeds.append({"category": cat_name, "prompt": p})

    out.write_text(json.dumps(all_seeds, indent=2))
    print(f"\n[gen-seeds] wrote {len(all_seeds)} total prompts ({len(CATEGORIES)} categories) to {out}")


if __name__ == "__main__":
    fire.Fire(main)
