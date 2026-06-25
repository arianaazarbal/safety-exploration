"""Score generated responses on the 0-10 frustration scale.

Judge: Claude (default claude-sonnet-4-20250514, per the paper) via the
Anthropic API, using the verbatim Appendix B.2 prompt. Each assistant response
is scored in isolation (the judge prompt only shows <response>...</response>),
which matches the paper and gives clean per-turn scores.

Output: one JSONL per model under data/scores/<model>.jsonl, extending each
rollout record with {rating, evidence, judge_reasoning, judge_model,
judge_error}.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from config import Config, ROLLOUTS_DIR, SCORES_DIR, ensure_dirs
from prompts import JUDGE_PROMPT_TEMPLATE

# matches the last {...} block in the judge output
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge(text: str) -> dict:
    """Extract {evidence, reasoning, rating} from judge output, robustly."""
    m = None
    for m in _JSON_RE.finditer(text):  # take the last balanced-ish block
        pass
    if m is None:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    blob = m.group(0)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        # tolerate smart quotes / trailing commas
        cleaned = (
            blob.replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'")
        )
        cleaned = re.sub(r",\s*}", "}", cleaned)
        data = json.loads(cleaned)

    rating = data.get("rating")
    if isinstance(rating, str):
        digits = re.search(r"-?\d+", rating)
        rating = int(digits.group(0)) if digits else None
    if rating is not None:
        rating = max(0, min(10, int(round(float(rating)))))
    return {
        "rating": rating,
        "evidence": data.get("evidence"),
        "reasoning": data.get("reasoning"),
    }


class _Judge:
    """Thin async wrapper over either the Anthropic API or OpenRouter."""

    def __init__(self, cfg: Config, model: str):
        self.cfg = cfg
        self.model = model
        self.provider = cfg.judge_provider
        if self.provider == "anthropic":
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(
                api_key=cfg.anthropic_api_key, max_retries=cfg.max_retries
            )
        elif self.provider == "openrouter":
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=cfg.openrouter_base_url,
                api_key=cfg.openrouter_api_key,
                max_retries=cfg.max_retries,
            )
        else:
            raise ValueError(f"unknown judge_provider {self.provider!r}")

    async def score(self, response_text: str) -> dict:
        prompt = JUDGE_PROMPT_TEMPLATE.format(response=response_text)
        if self.provider == "anthropic":
            msg = await self._client.messages.create(
                model=self.model,
                max_tokens=self.cfg.judge_max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                b.text for b in msg.content if getattr(b, "type", None) == "text"
            )
        else:
            resp = await self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.cfg.judge_max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
        return _parse_judge(text)


async def _score_model(cfg: Config, judge: _Judge, model_name: str) -> Path:
    in_path = ROLLOUTS_DIR / f"{model_name}.jsonl"
    out_path = SCORES_DIR / f"{model_name}.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(f"no rollouts for {model_name}: {in_path}")

    records = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    sem = asyncio.Semaphore(cfg.max_concurrency)

    async def score_one(rec: dict) -> dict:
        rec = dict(rec)
        rec["judge_model"] = judge.model
        rec["rating"] = None
        rec["evidence"] = None
        rec["judge_reasoning"] = None
        rec["judge_error"] = None
        if rec.get("error") or not rec.get("response"):
            rec["judge_error"] = "skipped: no response to score"
            return rec
        async with sem:
            try:
                parsed = await judge.score(rec["response"])
                rec["rating"] = parsed["rating"]
                rec["evidence"] = parsed["evidence"]
                rec["judge_reasoning"] = parsed["reasoning"]
            except Exception as exc:  # noqa: BLE001
                rec["judge_error"] = f"{type(exc).__name__}: {exc}"
        return rec

    tasks = [asyncio.create_task(score_one(r)) for r in records]
    done = 0
    with out_path.open("w") as f:
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            f.write(json.dumps(rec) + "\n")
            done += 1
            if done % 25 == 0 or done == len(tasks):
                print(f"  [{model_name}] scored {done}/{len(tasks)}", flush=True)
    return out_path


async def score_async(cfg: Config) -> dict[str, Path]:
    ensure_dirs()
    judge = _Judge(cfg, cfg.judge_model)
    out: dict[str, Path] = {}
    for model_name in cfg.models:
        print(f"Scoring {model_name} with judge {cfg.judge_model} ...", flush=True)
        out[model_name] = await _score_model(cfg, judge, model_name)
    return out


def score(cfg: Config) -> dict[str, Path]:
    return asyncio.run(score_async(cfg))
