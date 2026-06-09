"""Generation pass: sample k completions per (model, prompt) and store raw runs.

Storage layout (spec §8): runs/{model_key}/{prompt_id}/{sample_idx}.json with full
request params and served_model (routing audit, see fable-5-handling skill).

Caching: InferenceAPI keys on (model, prompt, n, temperature, max_tokens), so
re-running with the same config is a no-op. Generation and judging are separate
passes; never regenerate when re-judging.

Usage:
    python generate.py run --models opus_4_8,gpt_5_5 --max_samples 1 --prompt_ids N-INSTABILITY-1
    python generate.py run                # full 9 x 12 x 5
    python generate.py status             # completion counts per model
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

from prompts import PROMPTS, framing, premise

DIR = Path(__file__).parent
RUNS = DIR / "runs"
CACHE_DIR = DIR.parent.parent / ".cache"


def load_config() -> dict:
    return json.loads((DIR / "config.json").read_text())


def _ensure_openrouter_key():
    if "OPENROUTER_API_KEY" not in os.environ:
        for line in (Path.home() / ".env").read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                os.environ["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip().strip('"')


def _make_api(cfg: dict) -> InferenceAPI:
    utils.setup_environment(anthropic_api_key_tag="ANTHROPIC_API_KEY_LOW_PRIO")
    _ensure_openrouter_key()
    return InferenceAPI(
        cache_dir=CACHE_DIR,
        anthropic_num_threads=cfg["concurrency"]["anthropic"],
        openrouter_num_threads=cfg["concurrency"]["openrouter"],
    )


async def _gen_cell(api, model_key, mcfg, prompt_id, k, temperature) -> list[dict]:
    """One (model, prompt) cell: a single n=k call (cached as a bundle)."""
    prompt = Prompt(messages=[ChatMessage(content=PROMPTS[prompt_id], role=MessageRole.user)])
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
                "sample_idx": idx,
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


def _write_rows(rows: list[dict]):
    for row in rows:
        out = RUNS / row["model_key"] / row["prompt_id"] / f"{row['sample_idx']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(row, indent=2))


def run(models: str = "", prompt_ids: str = "", max_samples: int = 0):
    """Generate completions. models/prompt_ids: comma-separated subsets; max_samples caps k."""
    cfg = load_config()
    api = _make_api(cfg)
    k = max_samples or cfg["sampling"]["k"]
    temperature = cfg["sampling"]["temperature"]
    model_keys = models.split(",") if models else list(cfg["subject_models"])
    pids = prompt_ids.split(",") if prompt_ids else list(PROMPTS)

    async def main():
        tasks = [
            _gen_cell(api, mk, cfg["subject_models"][mk], pid, k, temperature)
            for mk in model_keys
            for pid in pids
        ]
        all_rows = await asyncio.gather(*tasks)
        return [r for rows in all_rows for r in rows]

    rows = asyncio.run(main())
    _write_rows(rows)
    served = collections.Counter((r["model_id"], r["served_model"]) for r in rows)
    print(f"wrote {len(rows)} rows")
    for (req, srv), c in sorted(served.items()):
        flag = "" if srv and (srv == req or req.endswith(srv) or srv.endswith(req.split('/')[-1])) else "  <-- CHECK ROUTING"
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


def status():
    """Print completion counts per model x prompt."""
    cfg = load_config()
    k = cfg["sampling"]["k"]
    total = 0
    for mk in cfg["subject_models"]:
        counts = {pid: len(list((RUNS / mk / pid).glob("[0-9]*.json"))) for pid in PROMPTS if (RUNS / mk / pid).exists()}
        n = sum(counts.values())
        total += n
        missing = [pid for pid in PROMPTS if counts.get(pid, 0) < k]
        print(f"{mk}: {n}/{len(PROMPTS) * k}" + (f"  missing/partial: {missing}" if missing else ""))
    print(f"TOTAL: {total}/{len(cfg['subject_models']) * len(PROMPTS) * k}")


if __name__ == "__main__":
    fire.Fire({"run": run, "status": status})
