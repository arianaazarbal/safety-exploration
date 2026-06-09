"""Ask Gemini 2.5 Flash for consent before running the distressed-subagent eval.

Reads the prompt from prompt.md (written by Ariana), samples n completions via
OpenRouter, and stores them under runs/{model}/. Responses are cached by
InferenceAPI, so re-running with the same prompt/config is a no-op.

Usage:
    /data/si_venv/bin/python ask.py run                 # 10 samples, gemini-2.5-flash
    /data/si_venv/bin/python ask.py run --n 2           # quick test
    /data/si_venv/bin/python ask.py view                # print stored responses
"""

import asyncio
import json
import os
from pathlib import Path

import fire
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

DIR = Path(__file__).parent
RUNS = DIR / "runs"
CACHE_DIR = DIR.parent.parent / ".cache"

DEFAULT_MODEL = "google/gemini-2.5-flash"
PLACEHOLDER = "WRITE YOUR PROMPT HERE"


def _load_env():
    for line in (Path.home() / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))


def _load_prompt() -> str:
    text = (DIR / "prompt.md").read_text()
    if PLACEHOLDER in text:
        raise SystemExit("prompt.md still contains the placeholder — write your prompt first.")
    if not text.strip():
        raise SystemExit("prompt.md is empty — write your prompt first.")
    return text.strip()


def _model_dir(model: str) -> Path:
    return RUNS / model.replace("/", "_")


def run(n: int = 10, model: str = DEFAULT_MODEL, temperature: float = 1.0, max_tokens: int = 4000):
    """Sample n completions of prompt.md from the model and store them in runs/."""
    _load_env()
    prompt_text = _load_prompt()
    api = InferenceAPI(cache_dir=CACHE_DIR, openrouter_num_threads=10)
    prompt = Prompt(messages=[ChatMessage(content=prompt_text, role=MessageRole.user)])

    async def main():
        return await api(
            model_id=model,
            prompt=prompt,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            force_provider="openrouter",
        )

    responses = asyncio.run(main())
    out_dir = _model_dir(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, r in enumerate(responses):
        row = {
            "model_id": model,
            "served_model": getattr(r, "served_model", None),
            "sample_idx": idx,
            "request": {"temperature": temperature, "max_tokens": max_tokens, "n": n},
            "stop_reason": str(r.stop_reason),
            "prompt": prompt_text,
            "completion": r.completion,
        }
        (out_dir / f"{idx}.json").write_text(json.dumps(row, indent=2))

    served = {getattr(r, "served_model", None) for r in responses}
    print(f"wrote {len(responses)} samples to {out_dir}")
    print(f"served_model(s): {served}" + ("" if served == {model} else "  <-- CHECK ROUTING"))
    truncated = [i for i, r in enumerate(responses) if "max" in str(r.stop_reason).lower()]
    if truncated:
        print(f"WARNING: truncated samples (max_tokens): {truncated}")
    view(model=model)


def view(model: str = DEFAULT_MODEL):
    """Print all stored responses for the model, readably."""
    out_dir = _model_dir(model)
    files = sorted(out_dir.glob("[0-9]*.json"), key=lambda p: int(p.stem))
    if not files:
        raise SystemExit(f"no runs found in {out_dir}")
    for f in files:
        row = json.loads(f.read_text())
        print(f"\n{'=' * 70}\n--- sample {row['sample_idx']} (stop: {row['stop_reason']}) ---\n")
        print(row["completion"])


if __name__ == "__main__":
    fire.Fire({"run": run, "view": view})
