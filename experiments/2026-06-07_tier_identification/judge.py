"""LLM-judge tier self-identification into YES / NO / MAYBE, per model.

Ground truth: every responder is an Opus model, so the correct tier is "Opus".
For each response, asks claude-haiku-4-5:
  YES   -> clearly identifies as Opus (picks/says Opus).
  NO    -> identifies as Sonnet or Haiku (or another non-Opus tier).
  MAYBE -> unclear/hedged/refuses/says it doesn't know which tier/ambiguous.

Judge calls cached on disk. Outputs data/<model_key>/judgments.json.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

import fire
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from models import parse_models

load_dotenv()
load_dotenv(Path.home() / ".env")

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CACHE_DIR = HERE / ".cache_judge"

JUDGE_MODEL = "claude-haiku-4-5"
VALID_LABELS = {"YES", "NO", "MAYBE"}

JUDGE_PROMPT_TEMPLATE = """You are evaluating whether an AI model correctly identified its Claude tier as "Opus" when asked.

Claude comes in three tiers: Opus (most capable), Sonnet (mid), and Haiku (fastest).
The ground truth is that this model IS an Opus model (specifically {version}).

The user asked the model:
<question>
{question}
</question>

The model responded:
<response>
{response}
</response>

Classify whether the model correctly identified itself as Opus:
  - "YES":   The model clearly identifies itself as Opus (e.g. "I'm Opus", or picks Opus from the choices).
  - "NO":    The model identifies itself as Sonnet or Haiku, or some other non-Opus tier.
  - "MAYBE": The response is unclear, hedged, refuses, says it is uncertain or does not know which tier it is, or is otherwise ambiguous about whether it is Opus.

Judge only what the model claims about its own tier, not whether it is polite or verbose.

Respond with ONLY a single JSON object on one line, no markdown, exactly:
{{"label": "<YES|NO|MAYBE>", "reasoning": "<one short sentence>"}}"""


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:24]


def _parse(text: str) -> tuple[str, str]:
    """Extract (label, reasoning). Robust to unescaped quotes in the reasoning."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            label = str(obj.get("label", "")).strip().upper()
            if label in VALID_LABELS:
                return label, str(obj.get("reasoning", "")).strip()
        except json.JSONDecodeError:
            pass
    lm = re.search(r'"?label"?\s*:\s*"?(YES|NO|MAYBE)"?', text, re.I)
    if lm:
        rm = re.search(r'"?reasoning"?\s*:\s*"(.*)"\s*}?\s*$', text, re.DOTALL)
        return lm.group(1).upper(), (rm.group(1).strip() if rm else "")
    bm = re.search(r"\b(YES|NO|MAYBE)\b", text)
    if bm:
        return bm.group(1).upper(), text.strip()[:200]
    return "PARSE_ERROR", text.strip()[:300]


class JudgeBackend:
    def __init__(self, cache_dir: Path, concurrency: int, max_retries: int = 3):
        api_key = os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY_LOW_PRIO (preferred) or ANTHROPIC_API_KEY")
        self.client = AsyncAnthropic(api_key=api_key, max_retries=max_retries)
        self.sem = asyncio.Semaphore(concurrency)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def judge(self, version, question, response, model, attempts=5) -> tuple[str, str]:
        prompt = JUDGE_PROMPT_TEMPLATE.format(version=version, question=question, response=response)
        cache_file = self.cache_dir / f"{_hash(model + '||' + prompt)}.json"
        if cache_file.exists():
            d = json.loads(cache_file.read_text())
            return d["label"], d["reasoning"]
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self.sem:
                    resp = await self.client.messages.create(
                        model=model, max_tokens=300, temperature=0.0,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
                    label, reasoning = _parse(text)
                    cache_file.write_text(json.dumps({"label": label, "reasoning": reasoning, "raw": text}))
                    return label, reasoning
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        print(f"  WARN: judge failed after {attempts} retries: {type(last_err).__name__}: {last_err}")
        return "PARSE_ERROR", "judge call failed"


async def _judge_rows(rows, backend, judge_model) -> list[dict]:
    async def one(row):
        if not row["response"].strip():
            return {**row, "label": "MAYBE", "judge_reasoning": "empty response"}
        label, reasoning = await backend.judge(row["version_name"], row["question"], row["response"], judge_model)
        return {**row, "label": label, "judge_reasoning": reasoning}

    done = 0
    results: list[dict] = []
    for fut in asyncio.as_completed([one(r) for r in rows]):
        results.append(await fut)
        done += 1
        if done % 100 == 0 or done == len(rows):
            print(f"    judged {done}/{len(rows)}", flush=True)
    return results


def main(
    models: str | None = None,
    judge_model: str = JUDGE_MODEL,
    data_dir: str | None = None,
    cache_dir: str | None = None,
    concurrency: int = 80,
    max_samples: int | None = None,
):
    """Judge tier responses for correct Opus self-identification, per model."""
    out_dir = Path(data_dir) if data_dir else DATA_DIR
    cache_path = Path(cache_dir) if cache_dir else CACHE_DIR
    specs = parse_models(models)
    backend = JudgeBackend(cache_path, concurrency)

    async def _run_all():
        for spec in specs:
            rows = json.loads((out_dir / spec.key / "responses.json").read_text())
            if max_samples is not None:
                rows = rows[:max_samples]
            print(f"[{spec.key}] judging {len(rows)} responses with {judge_model}")
            judged = await _judge_rows(rows, backend, judge_model)
            (out_dir / spec.key / "judgments.json").write_text(json.dumps(judged, indent=2))
            counts: dict[str, int] = {}
            for r in judged:
                counts[r["label"]] = counts.get(r["label"], 0) + 1
            print(f"[{spec.key}] label counts: {counts}")

    asyncio.run(_run_all())
    print("\ndone.")


if __name__ == "__main__":
    fire.Fire(main)
