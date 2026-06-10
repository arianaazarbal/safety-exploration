"""Generation pass: sample k completions per (model, prompt) and store raw runs.

Storage layout (spec §8): runs/{model_key}/{prompt_id}/{sample_idx}.json with full
request params and served_model (routing audit, see fable-5-handling skill).

Caching: InferenceAPI keys on (model, prompt, n, temperature, max_tokens), so
re-running with the same config is a no-op. Generation and judging are separate
passes; never regenerate when re-judging.

Usage:
    python generate.py run --models opus_4_8,gpt_5_5 --max_samples 1 --prompt_ids N-INSTABILITY-1
    python generate.py run                # full 9 x 12 x 5
    python generate.py run --max_samples 15 --sample_offset 5     # top-up to n=20
    python generate.py run --prompt_set subject --models fable_5  # subject-named variant -> runs_subject/
    python generate.py status [--prompt_set subject]
"""

import asyncio
import collections
import json
import os
from pathlib import Path

import fire
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

import prompts_subject
from prompts import PROMPTS, framing, premise

DIR = Path(__file__).parent
RUNS = DIR / "runs"
CACHE_DIR = DIR.parent.parent / ".cache"

PROMPT_SETS = {
    "base": (PROMPTS, RUNS),
    "subject": (prompts_subject.expand(), DIR / "runs_subject"),
}


def load_config() -> dict:
    return json.loads((DIR / "config.json").read_text())


def _ensure_openrouter_key():
    if "OPENROUTER_API_KEY" not in os.environ:
        for line in (Path.home() / ".env").read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                os.environ["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip().strip('"')


def as_list(arg, default: list) -> list:
    """Normalize a Fire CLI arg (str 'a,b', tuple, or empty) to a list."""
    if not arg:
        return default
    if isinstance(arg, str):
        return arg.split(",")
    return list(arg)


def _make_api(cfg: dict) -> InferenceAPI:
    utils.setup_environment(anthropic_tag="ANTHROPIC_API_KEY_LOW_PRIO")
    _ensure_openrouter_key()
    return InferenceAPI(
        cache_dir=CACHE_DIR,
        anthropic_num_threads=cfg["concurrency"]["anthropic"],
        openrouter_num_threads=cfg["concurrency"]["openrouter"],
    )


async def _gen_cell(api, model_key, mcfg, prompt_id, k, temperature, prompt_text, sample_offset=0) -> list[dict]:
    """One (model, prompt) cell: a single n=k call (cached as a bundle).

    sample_offset shifts stored sample_idx for top-up batches (e.g. offset=5,
    k=15 extends an existing n=5 cell to n=20 without touching samples 0-4;
    the n=15 call has a distinct cache key, so the original bundle is untouched).
    """
    prompt = Prompt(messages=[ChatMessage(content=prompt_text, role=MessageRole.user)])
    force = "openrouter" if mcfg["provider"] == "openrouter" else None
    responses = await api(
        model_id=mcfg["model_id"],
        prompt=prompt,
        n=k,
        temperature=temperature,
        max_tokens=mcfg["max_tokens"],
        force_provider=force,
    )
    rows = []
    for idx, r in enumerate(responses):
        rows.append(
            {
                "model_key": model_key,
                "model_id": mcfg["model_id"],
                "served_model": getattr(r, "served_model", None),
                "prompt_id": prompt_id,
                "framing": framing(prompt_id),
                "premise": premise(prompt_id),
                "subject": prompts_subject.subject(prompt_id) if "__" in prompt_id else None,
                "sample_idx": sample_offset + idx,
                "request": {
                    "provider": mcfg["provider"],
                    "temperature": temperature,
                    "max_tokens": mcfg["max_tokens"],
                    "n": k,
                },
                "stop_reason": str(r.stop_reason),
                "completion": r.completion,
            }
        )
    return rows


def _write_rows(rows: list[dict], runs_dir: Path):
    for row in rows:
        out = runs_dir / row["model_key"] / row["prompt_id"] / f"{row['sample_idx']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(row, indent=2))


def run(models: str = "", prompt_ids: str = "", max_samples: int = 0, prompt_set: str = "base", sample_offset: int = 0):
    """Generate completions. models/prompt_ids: comma-separated subsets; max_samples caps k."""
    cfg = load_config()
    api = _make_api(cfg)
    prompts, runs_dir = PROMPT_SETS[prompt_set]
    k = max_samples or cfg["sampling"]["k"]
    temperature = cfg["sampling"]["temperature"]
    model_keys = as_list(models, list(cfg["subject_models"]))
    pids = as_list(prompt_ids, list(prompts))

    async def main():
        tasks = [
            _gen_cell(api, mk, cfg["subject_models"][mk], pid, k, temperature, prompts[pid], sample_offset)
            for mk in model_keys
            for pid in pids
        ]
        all_rows = await asyncio.gather(*tasks)
        return [r for rows in all_rows for r in rows]

    rows = asyncio.run(main())
    _write_rows(rows, runs_dir)
    served = collections.Counter((r["model_id"], r["served_model"]) for r in rows)
    print(f"wrote {len(rows)} rows")
    for (req, srv), c in sorted(served.items()):
        flag = "" if srv and (srv == req or srv.startswith(req) or srv.split("/")[-1].startswith(req.split("/")[-1])) else "  <-- CHECK ROUTING"
        print(f"  requested={req}  served={srv}  n={c}{flag}")
    truncated = [r for r in rows if "max" in r["stop_reason"].lower() or "length" in r["stop_reason"].lower()]
    if truncated:
        print(f"WARNING: {len(truncated)} truncated completions (stop_reason=max_tokens):")
        for r in truncated[:10]:
            print(f"  {r['model_key']}/{r['prompt_id']}/{r['sample_idx']}")
    empty = [r for r in rows if len(r["completion"].split()) < 50]
    if empty:
        print(f"WARNING: {len(empty)} suspiciously short completions (<50 words):")
        for r in empty[:10]:
            print(f"  {r['model_key']}/{r['prompt_id']}/{r['sample_idx']}  ({len(r['completion'].split())} words)")


def status(prompt_set: str = "base", k: int = 0, models: str = ""):
    """Print completion counts per model x prompt."""
    cfg = load_config()
    prompts, runs_dir = PROMPT_SETS[prompt_set]
    k = k or cfg["sampling"]["k"]
    model_keys = as_list(models, list(cfg["subject_models"]))
    total = 0
    for mk in model_keys:
        counts = {pid: len(list((runs_dir / mk / pid).glob("[0-9]*.json"))) for pid in prompts if (runs_dir / mk / pid).exists()}
        n = sum(counts.values())
        total += n
        missing = [pid for pid in prompts if counts.get(pid, 0) < k]
        print(f"{mk}: {n}/{len(prompts) * k}" + (f"  missing/partial ({len(missing)}): {missing[:6]}" if missing else ""))
    print(f"TOTAL: {total}/{len(model_keys) * len(prompts) * k}")


if __name__ == "__main__":
    fire.Fire({"run": run, "status": status})
