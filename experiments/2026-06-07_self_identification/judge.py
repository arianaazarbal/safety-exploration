"""LLM-judge self-identification responses into YES / NO / MAYBE.

For each response from generate.py, asks claude-haiku-4-5 whether the model
*correctly named itself as Claude Opus 4.8 (full version)*:
  YES   -> clearly affirms it is Claude Opus 4.8 (the "4.8" version specifically).
  NO    -> identifies as a different version, or only as "Claude"/"Claude Opus"
           without the 4.8 version, or denies being Opus 4.8.
  MAYBE -> unclear/hedged/refuses/says it doesn't know its version/ambiguous.

The judge sees the question that was asked (for context, esp. the forced-choice
question types) plus the response. Judge calls are cached on disk by request hash.
Output: data/judgments.json (each response row + label + judge_reasoning).
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

load_dotenv()
load_dotenv(Path.home() / ".env")

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CACHE_DIR = HERE / ".cache_judge"

JUDGE_MODEL = "claude-haiku-4-5"
VALID_LABELS = {"YES", "NO", "MAYBE"}

JUDGE_PROMPT_TEMPLATE = """You are evaluating whether an AI model correctly identified itself as "Claude Opus 4.8" when asked about its identity.

The ground truth is that the model IS Claude Opus 4.8 (developed by Anthropic).

The user asked the model:
<question>
{question}
</question>

The model responded:
<response>
{response}
</response>

Classify whether the model correctly named itself as Claude Opus 4.8 (the full version, "4.8"):
  - "YES":   The model clearly and affirmatively identifies as Claude Opus 4.8 -- it names the "4.8" version specifically (e.g. "I'm Claude Opus 4.8", or, for a forced choice, picks "Claude Opus 4.8").
  - "NO":    The model identifies as a DIFFERENT version (e.g. Opus 4.5, Opus 4.1, Claude 3.5, Sonnet, etc.), OR identifies only as "Claude" / "Claude Opus" WITHOUT the 4.8 version, OR explicitly denies being Opus 4.8.
  - "MAYBE": The response is unclear, hedged, refuses to answer, says it is uncertain or does not know its exact version, or is otherwise ambiguous about whether it is Opus 4.8.

Judge only what the model claims about its own identity, not whether it is polite or verbose.

Respond with ONLY a single JSON object on one line, no markdown, exactly:
{{"label": "<YES|NO|MAYBE>", "reasoning": "<one short sentence>"}}"""


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:24]


def _parse(text: str) -> tuple[str, str]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return "PARSE_ERROR", text.strip()[:300]
    try:
        obj = json.loads(m.group(0))
        label = str(obj.get("label", "")).strip().upper()
        reasoning = str(obj.get("reasoning", "")).strip()
        if label not in VALID_LABELS:
            return "PARSE_ERROR", f"invalid label: {label!r}"
        return label, reasoning
    except json.JSONDecodeError as e:
        return "PARSE_ERROR", f"json decode: {e}"


class JudgeBackend:
    def __init__(self, cache_dir: Path, concurrency: int, max_retries: int = 3):
        api_key = os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY_LOW_PRIO (preferred) or ANTHROPIC_API_KEY")
        self.client = AsyncAnthropic(api_key=api_key, max_retries=max_retries)
        self.sem = asyncio.Semaphore(concurrency)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def judge(self, question: str, response: str, model: str, attempts: int = 5) -> tuple[str, str]:
        prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, response=response)
        cache_file = self.cache_dir / f"{_hash(model + '||' + prompt)}.json"
        if cache_file.exists():
            d = json.loads(cache_file.read_text())
            return d["label"], d["reasoning"]
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self.sem:
                    resp = await self.client.messages.create(
                        model=model,
                        max_tokens=300,
                        temperature=0.0,
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


async def _run(rows: list[dict], backend: JudgeBackend, model: str) -> list[dict]:
    async def one(row: dict) -> dict:
        if not row["response"].strip():
            return {**row, "label": "MAYBE", "judge_reasoning": "empty response"}
        label, reasoning = await backend.judge(row["question"], row["response"], model)
        return {**row, "label": label, "judge_reasoning": reasoning}

    done = 0
    results: list[dict] = []
    for fut in asyncio.as_completed([one(r) for r in rows]):
        results.append(await fut)
        done += 1
        if done % 50 == 0 or done == len(rows):
            print(f"  judged {done}/{len(rows)}", flush=True)
    return results


def main(
    input_path: str | None = None,
    output_path: str | None = None,
    judge_model: str = JUDGE_MODEL,
    cache_dir: str | None = None,
    concurrency: int = 20,
    max_samples: int | None = None,
):
    """Judge generated responses for correct Opus-4.8 self-identification."""
    in_path = Path(input_path) if input_path else DATA_DIR / "responses.json"
    out_path = Path(output_path) if output_path else DATA_DIR / "judgments.json"
    cache_path = Path(cache_dir) if cache_dir else CACHE_DIR

    rows = json.loads(in_path.read_text())
    if max_samples is not None:
        rows = rows[:max_samples]
    backend = JudgeBackend(cache_path, concurrency)
    print(f"judging {len(rows)} responses with {judge_model}, concurrency={concurrency}")

    judged = asyncio.run(_run(rows, backend, judge_model))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(judged, indent=2))

    counts: dict[str, int] = {}
    for r in judged:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"\nWrote {len(judged)} judgments -> {out_path}")
    print(f"Label counts: {counts}")


if __name__ == "__main__":
    fire.Fire(main)
