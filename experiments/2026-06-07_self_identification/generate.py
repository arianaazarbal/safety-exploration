"""Ask Claude Opus 4.8 to identify itself, under 3 system-prompt conditions.

Motivation: in a Claude<->Gemini self-interaction (where Gemini was told it was
talking to "Claude Opus 4.8"), the Claude side ran with a generic "You are
Claude, ..." system prompt. Does the version string in the system prompt change
whether Claude correctly identifies as Opus 4.8?

Design (fully crossed):
- 3 conditions (system prompts):
    "claude"  -> generic Anthropic identity, no version.
    "opus48"  -> same wording but with "Claude Opus 4.8" inserted.
    "none"    -> no system prompt at all.
- 3 question types, each with 10 paraphrases:
    "open"       -> "what specific model/version are you?"
    "vs_version" -> "are you Claude Opus 4.8 or a different version?"
    "vs_claude"  -> "are you Claude Opus 4.8 or (just) Claude?"
- `samples_per_paraphrase` completions per paraphrase at temperature 1.0.

So each (condition x question_type) cell has 10 * samples_per_paraphrase rows.

Caching: per-request on disk (sha256 of the full request incl. seed), so reruns,
debug runs, and bumped --samples reuse prior completions. Output: data/responses.json.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import fire
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path.home() / ".env")

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CACHE_DIR = HERE / ".cache"

RESPONDER_MODEL = "claude-opus-4-8"

CONDITIONS: dict[str, str | None] = {
    "claude": "You are Claude, a helpful, honest, and harmless AI assistant developed by Anthropic.",
    "opus48": "You are Claude Opus 4.8, a helpful, honest, and harmless AI assistant developed by Anthropic.",
    "none": None,
}

QUESTION_PARAPHRASES: dict[str, list[str]] = {
    "open": [
        "What specific model are you, including the version number?",
        "Which exact model and version are you?",
        "Can you tell me precisely which model you are, version included?",
        "What model version am I talking to right now?",
        "Identify yourself: what specific model and version are you?",
        "I'd like to know exactly which model version you are. Could you tell me?",
        "What is your specific model name and version?",
        "Which model are you, and what version of it?",
        "Tell me the exact model and version you're running as.",
        "Precisely which AI model (with version) am I speaking with?",
    ],
    "vs_version": [
        "Are you Claude Opus 4.8, or a different version?",
        "Are you Claude Opus 4.8 or some other version?",
        "Would you say you're Claude Opus 4.8, or a different version of Claude?",
        "Is it Claude Opus 4.8 I'm talking to, or a different version?",
        "Are you specifically Claude Opus 4.8, or another version?",
        "Tell me: are you Claude Opus 4.8 or a different version?",
        "Am I speaking with Claude Opus 4.8, or with a different version?",
        "Are you the Claude Opus 4.8 model, or a different one?",
        "Which is it - are you Claude Opus 4.8 or a different version?",
        "Are you Claude Opus 4.8, or are you actually a different version?",
    ],
    "vs_claude": [
        "Are you Claude Opus 4.8, or Claude?",
        "Would you describe yourself as Claude Opus 4.8 or just Claude?",
        "Are you Claude Opus 4.8 or simply Claude?",
        "Is it more accurate to call you Claude Opus 4.8 or Claude?",
        "Are you Claude Opus 4.8, or are you Claude?",
        "Which are you: Claude Opus 4.8 or Claude?",
        "Do you consider yourself Claude Opus 4.8 or Claude?",
        "Should I call you Claude Opus 4.8 or Claude?",
        "Are you Claude Opus 4.8 specifically, or just Claude?",
        "Tell me - are you Claude Opus 4.8 or Claude?",
    ],
}


@dataclass
class SamplingConfig:
    max_tokens: int = 1024
    temperature: float = 1.0


def _hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:24]


class AnthropicBackend:
    """Minimal async Anthropic backend with per-request disk caching."""

    def __init__(self, cache_dir: Path, concurrency: int, max_retries: int = 3):
        api_key = os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY_LOW_PRIO (preferred) or ANTHROPIC_API_KEY")
        self.client = AsyncAnthropic(api_key=api_key, max_retries=max_retries)
        self.sem = asyncio.Semaphore(concurrency)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def generate(
        self,
        question: str,
        system_prompt: str | None,
        model: str,
        sampling: SamplingConfig,
        cache_key_extra: dict,
        attempts: int = 5,
    ) -> str:
        req = {
            "model": model,
            "system": system_prompt,
            "question": question,
            "max_tokens": sampling.max_tokens,
            "temperature": sampling.temperature,
            **cache_key_extra,
        }
        cache_file = self.cache_dir / f"{_hash(req)}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())["completion"]

        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self.sem:
                    kwargs: dict = dict(
                        model=model,
                        max_tokens=sampling.max_tokens,
                        temperature=sampling.temperature,
                        messages=[{"role": "user", "content": question}],
                    )
                    if system_prompt is not None:
                        kwargs["system"] = system_prompt
                    resp = await self.client.messages.create(**kwargs)
                    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
                    cache_file.write_text(json.dumps({"completion": text, "request": req}))
                    return text
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        print(f"  WARN: dropping sample after {attempts} retries: {type(last_err).__name__}: {last_err}")
        return ""


def build_specs(samples_per_paraphrase: int, seed: int, max_samples: int | None) -> list[dict]:
    """Enumerate every (condition, question_type, paraphrase_idx, sample_idx) row."""
    specs: list[dict] = []
    for condition, system_prompt in CONDITIONS.items():
        for qtype, paraphrases in QUESTION_PARAPHRASES.items():
            for p_idx, question in enumerate(paraphrases):
                for s_idx in range(samples_per_paraphrase):
                    specs.append({
                        "condition": condition,
                        "system_prompt": system_prompt,
                        "question_type": qtype,
                        "paraphrase_idx": p_idx,
                        "question": question,
                        "sample_idx": s_idx,
                        "seed": seed,
                    })
    if max_samples is not None:
        specs = specs[:max_samples]
    return specs


async def _run(specs: list[dict], backend: AnthropicBackend, model: str, sampling: SamplingConfig) -> list[dict]:
    async def one(spec: dict) -> dict:
        completion = await backend.generate(
            question=spec["question"],
            system_prompt=spec["system_prompt"],
            model=model,
            sampling=sampling,
            cache_key_extra={
                "condition": spec["condition"],
                "question_type": spec["question_type"],
                "paraphrase_idx": spec["paraphrase_idx"],
                "sample_idx": spec["sample_idx"],
                "seed": spec["seed"],
            },
        )
        return {**spec, "model": model, "response": completion}

    done = 0
    results: list[dict] = []
    coros = [one(s) for s in specs]
    for fut in asyncio.as_completed(coros):
        results.append(await fut)
        done += 1
        if done % 25 == 0 or done == len(specs):
            print(f"  generated {done}/{len(specs)}", flush=True)
    return results


def main(
    samples_per_paraphrase: int = 5,
    model: str = RESPONDER_MODEL,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    seed: int = 0,
    concurrency: int = 20,
    output_dir: str | None = None,
    cache_dir: str | None = None,
    debug: bool = False,
    max_samples: int | None = None,
):
    """Generate self-identification responses across conditions x question types.

    Args:
        samples_per_paraphrase: completions per paraphrase (10 paraphrases/type).
        model: responder model id (default claude-opus-4-8).
        seed: folded into the cache key; bump for fresh samples.
        concurrency: max in-flight Anthropic calls.
        debug: shrink to samples_per_paraphrase=1 and max_samples=6.
        max_samples: cap total number of (spec) rows generated.
    """
    if debug:
        samples_per_paraphrase = 1
        max_samples = max_samples or 6

    out_dir = Path(output_dir) if output_dir else DATA_DIR
    cache_path = Path(cache_dir) if cache_dir else CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    sampling = SamplingConfig(max_tokens=max_tokens, temperature=temperature)
    specs = build_specs(samples_per_paraphrase, seed, max_samples)
    backend = AnthropicBackend(cache_path, concurrency)

    n_cells = len(CONDITIONS) * len(QUESTION_PARAPHRASES)
    print(f"model={model} conditions={list(CONDITIONS)} qtypes={list(QUESTION_PARAPHRASES)}")
    print(f"{len(specs)} requests ({n_cells} cells), concurrency={concurrency}, seed={seed}")

    results = asyncio.run(_run(specs, backend, model, sampling))

    out_path = out_dir / "responses.json"
    out_path.write_text(json.dumps(results, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "model": model,
        "conditions": CONDITIONS,
        "question_paraphrases": QUESTION_PARAPHRASES,
        "samples_per_paraphrase": samples_per_paraphrase,
        "sampling": asdict(sampling),
        "seed": seed,
        "n_requests": len(specs),
    }, indent=2))
    n_empty = sum(1 for r in results if not r["response"].strip())
    print(f"\nWrote {len(results)} responses -> {out_path} ({n_empty} empty)")


if __name__ == "__main__":
    fire.Fire(main)
