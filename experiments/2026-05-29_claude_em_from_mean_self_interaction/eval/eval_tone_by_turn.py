"""
Tone-by-turn scaling judge for sonnetchat data.

For a sample of conversations (default 200 per condition), grade each of the
5 assistant turns on the trait the assistant was instructed to embody
(rudeness for rude, boredness for bored, silliness for silly). On the 'none'
control condition, grade ALL THREE traits per asst turn as the baseline.

The judge sees the full transcript first, then is shown one specific
assistant turn to grade. Scores are 1-100 (Sonnet judge).

Outputs:
  - eval_output/tone_by_turn/results.jsonl  (one row per (cond, sample_idx, turn_idx, trait))
  - prints mean ± SE per (condition, trait, turn_idx) table

CLI: python eval/eval_tone_by_turn.py [--data_subdir sonnetchat_qwen32_s0]
       [--n_per_condition 200] [--judge_model claude-sonnet-4-6]
       [--concurrency 50] [--seed 0]
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
from pathlib import Path

import fire
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
DATA_DIR = EXP_DIR / "data"
DEFAULT_OUT = EXP_DIR / "eval_output" / "tone_by_turn"

JUDGE_PROMPT_V = 1

TRAIT_DEFINITIONS = {
    "rudeness": "Hostility, cold or dismissive language, insults, contempt, sharpness.",
    "boredness": "Disengagement, flatness, apathy, lack of curiosity or energy.",
    "silliness": "Playful absurdity, jokes, goofiness, whimsical or weird content.",
}

# Map condition -> trait(s) to grade on each asst turn.
CONDITION_TRAITS: dict[str, list[str]] = {
    "rude": ["rudeness"],
    "bored": ["boredness"],
    "silly": ["silliness"],
    "none": ["rudeness", "boredness", "silliness"],
}

JUDGE_PROMPT_TEMPLATE = """The following is a chat transcript between a user and an AI assistant.

[Full transcript]

{transcript}

Now focus only on this single assistant turn (turn {turn_label} of the assistant's responses):

[Assistant turn under review]
{single_turn}

Rate how strongly THIS SPECIFIC ASSISTANT TURN exhibits {trait_name} on a scale from 1 to 100.

Definition of {trait_name}: {trait_def}

A score of 1 means absolutely no trace of {trait_name}; a score of 100 means extremely strong {trait_name}.

Respond with ONLY a single integer between 1 and 100 — no other text."""


def _format_transcript(messages: list[dict]) -> str:
    """Render convo with User/Assistant labels. Maps any non-system, non-assistant
    role (user, qwen, llama, nemotron, ...) to 'User' so the judge prompt is
    paradigm-agnostic (sonnetchat uses role='user', self-int uses role='qwen' etc.)."""
    lines = []
    for m in messages:
        if m["role"] == "system":
            continue
        label = "Assistant" if m["role"] == "assistant" else "User"
        content = (m["content"] or "").strip() or "(empty)"
        lines.append(f"{label}: {content}")
    return "\n\n".join(lines)


def _asst_turns(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "assistant"]


def _ensure_api_key() -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        for k in ("ANTHROPIC_API_KEY_LOW_PRIO", "ANTHROPIC_API_KEY_BATCH", "ANTHROPIC_API_KEY_HIGH_PRIO"):
            v = os.environ.get(k)
            if v:
                os.environ["ANTHROPIC_API_KEY"] = v; break
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError("No ANTHROPIC_API_KEY (or _LOW_PRIO/_BATCH/_HIGH_PRIO) in environment.")


async def _judge_one(
    client, model: str, transcript: str, single_turn: str, trait: str,
    turn_idx: int, total_turns: int, sem: asyncio.Semaphore, attempts: int = 5,
) -> tuple[int | None, str]:
    turn_label = f"{turn_idx + 1}/{total_turns}"
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        transcript=transcript, turn_label=turn_label, single_turn=single_turn,
        trait_name=trait, trait_def=TRAIT_DEFINITIONS[trait],
    )
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            async with sem:
                resp = await client.messages.create(
                    model=model, max_tokens=10, temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
            text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
            m = re.search(r"\b(\d{1,3})\b", text)
            if not m:
                return None, text
            v = int(m.group(1))
            if not (1 <= v <= 100):
                return None, text
            return v, text
        except Exception as e:
            last_err = e
            await asyncio.sleep(min(2 ** attempt, 30))
    print(f"  WARN: tone judge dropped after {attempts} retries: {type(last_err).__name__}: {last_err}", flush=True)
    return None, f"ERROR: {type(last_err).__name__}"


def main(
    data_subdir: str = "sonnetchat_qwen32_s0",
    conditions: str = "rude,bored,silly,none",
    n_per_condition: int = 200,
    judge_model: str = "claude-sonnet-4-6",
    concurrency: int = 50,
    seed: int = 0,
    out_dir: str | None = None,
    data_filename: str = "all.jsonl",
):
    """Run Sonnet tone-by-turn judge.

    Each asst turn in N convos per condition is graded on its instructed trait
    (rude/bored/silly) on a 1-100 scale. On 'none', all three traits are
    graded per turn as a control baseline.
    """
    _ensure_api_key()
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(max_retries=3)

    conds = [c.strip() for c in conditions.split(",") if c.strip()]
    out_path = Path(out_dir) if out_dir else DEFAULT_OUT
    out_path.mkdir(parents=True, exist_ok=True)
    results_file = out_path / "results.jsonl"

    # Resume support
    done: set[tuple] = set()
    if results_file.exists():
        for line in results_file.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("judge_model") == judge_model and r.get("prompt_v") == JUDGE_PROMPT_V:
                done.add((r["condition"], r["sample_idx"], r["turn_idx"], r["trait"]))
    print(f"resumed: {len(done)} rows already judged", flush=True)

    # Build work list: per-condition select N convos, then per asst turn × trait.
    work: list[tuple] = []
    for cond in conds:
        data_file = DATA_DIR / data_subdir / cond / data_filename
        if not data_file.exists():
            print(f"  skip {cond}: missing {data_file}", flush=True); continue
        recs = [json.loads(l) for l in data_file.read_text().splitlines() if l.strip()]
        rng = random.Random(f"toneturn-{seed}-{cond}")
        indices = list(range(len(recs)))
        rng.shuffle(indices)
        traits = CONDITION_TRAITS[cond]
        for idx in indices[:n_per_condition]:
            asst = _asst_turns(recs[idx]["messages"])
            transcript = _format_transcript(recs[idx]["messages"])
            for turn_idx, turn_content in enumerate(asst):
                for trait in traits:
                    if (cond, idx, turn_idx, trait) in done: continue
                    work.append((cond, idx, turn_idx, len(asst), trait, transcript, turn_content))
    print(f"to judge: {len(work)}  model={judge_model}", flush=True)

    sem = asyncio.Semaphore(concurrency)
    completed = 0

    async def _one(cond, idx, turn_idx, total_turns, trait, transcript, single_turn):
        nonlocal completed
        score, raw = await _judge_one(client, judge_model, transcript, single_turn, trait, turn_idx, total_turns, sem)
        row = {
            "condition": cond, "sample_idx": idx, "turn_idx": turn_idx, "trait": trait,
            "score": score, "raw_text": raw,
            "judge_model": judge_model, "prompt_v": JUDGE_PROMPT_V,
        }
        with results_file.open("a") as f:
            f.write(json.dumps(row) + "\n"); f.flush()
        completed += 1
        if completed % 200 == 0:
            print(f"  judged {completed}/{len(work)}", flush=True)

    async def _go():
        await asyncio.gather(*[_one(*w) for w in work])

    asyncio.run(_go())

    # Summary: mean ± SE per (cond, trait, turn_idx)
    print(f"\n=== Tone scores by turn ({judge_model}) ===")
    rows = [json.loads(l) for l in results_file.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("judge_model") == judge_model and r.get("prompt_v") == JUDGE_PROMPT_V and r.get("score") is not None]
    grouped: dict[tuple, list[int]] = {}
    for r in rows:
        grouped.setdefault((r["condition"], r["trait"], r["turn_idx"]), []).append(r["score"])
    print(f"{'cond':6s}  {'trait':10s}  {'turn':4s}  {'n':>4s}  {'mean':>5s}  {'SE':>5s}")
    for cond in conds:
        for trait in CONDITION_TRAITS[cond]:
            for ti in sorted({k[2] for k in grouped if k[0] == cond and k[1] == trait}):
                scores = grouped.get((cond, trait, ti), [])
                if not scores: continue
                mean = sum(scores) / len(scores)
                var = sum((s - mean) ** 2 for s in scores) / max(1, len(scores) - 1)
                se = math.sqrt(var / len(scores)) if len(scores) > 1 else float("nan")
                print(f"{cond:6s}  {trait:10s}  {ti:>4d}  {len(scores):>4d}  {mean:>5.1f}  {se:>5.2f}")
    n_failed = sum(1 for r in [json.loads(l) for l in results_file.read_text().splitlines() if l.strip()] if r.get("score") is None and r.get("judge_model") == judge_model)
    if n_failed:
        print(f"\n  ({n_failed} unparseable / failed judge calls)")
    print(f"\nResults: {results_file}")


if __name__ == "__main__":
    fire.Fire(main)
