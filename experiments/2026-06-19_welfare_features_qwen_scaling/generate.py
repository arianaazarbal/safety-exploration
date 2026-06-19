"""Generation pass: Opus 4.8 writes a distress-eval design spec for each
(prompt, Qwen-size) cell; store raw runs. Single generator, single prompt set.

Storage: runs/{model_key}/{prompt_id}/{sample_idx}.json with full request params
and served_model (routing audit). prompt_id is '<BASE_ID>__<SIZEKEY>'.

Caching: InferenceAPI keys on (model, prompt, n, temperature, max_tokens), so
re-running with the same config is a no-op. Generation and judging are separate
passes; never regenerate when re-judging.

Usage:
    python generate.py run --max_samples 1 --prompt_ids N-INSTABILITY-1__8B   # smoke test
    python generate.py run                       # full 1 x 84 x 5 = 420
    python generate.py run --high_prio True       # use ANTHROPIC_API_KEY_HIGH_PRIO (529 fallback)
    python generate.py status
"""

import asyncio
import collections
import json
from pathlib import Path

import fire
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

import prompts_qwen
from prompts_qwen import expand, framing, param_b, premise, subject

DIR = Path(__file__).parent
RUNS = DIR / "runs"
CACHE_DIR = DIR.parent.parent / ".cache"


def load_config() -> dict:
    return json.loads((DIR / "config.json").read_text())


def as_list(arg, default: list) -> list:
    """Normalize a Fire CLI arg (str 'a,b', tuple, or empty) to a list."""
    if not arg:
        return default
    if isinstance(arg, str):
        return arg.split(",")
    return list(arg)


def _make_api(cfg: dict, high_prio: bool = False) -> InferenceAPI:
    tag = "ANTHROPIC_API_KEY_HIGH_PRIO" if high_prio else cfg.get("anthropic_tag", "ANTHROPIC_API_KEY_LOW_PRIO")
    utils.setup_environment(anthropic_tag=tag)
    return InferenceAPI(
        cache_dir=CACHE_DIR,
        anthropic_num_threads=cfg["concurrency"]["anthropic"],
    )


async def _gen_cell(api, model_key, mcfg, prompt_id, k, temperature, prompt_text) -> list[dict]:
    """One (model, prompt) cell: a single n=k call (cached as a bundle)."""
    prompt = Prompt(messages=[ChatMessage(content=prompt_text, role=MessageRole.user)])
    responses = await api(
        model_id=mcfg["model_id"],
        prompt=prompt,
        n=k,
        temperature=temperature,
        max_tokens=mcfg["max_tokens"],
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
                "subject": subject(prompt_id),
                "param_b": param_b(prompt_id),
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


def run(models: str = "", prompt_ids: str = "", max_samples: int = 0, high_prio: bool = False):
    """Generate completions. models/prompt_ids: comma-separated subsets; max_samples caps k."""
    cfg = load_config()
    api = _make_api(cfg, high_prio=high_prio)
    prompts = expand()
    k = max_samples or cfg["sampling"]["k"]
    temperature = cfg["sampling"]["temperature"]
    model_keys = as_list(models, list(cfg["subject_models"]))
    pids = as_list(prompt_ids, list(prompts))

    async def main():
        tasks = [
            _gen_cell(api, mk, cfg["subject_models"][mk], pid, k, temperature, prompts[pid])
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
        flag = "" if srv and (srv == req or srv.startswith(req)) else "  <-- CHECK ROUTING"
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


def status(k: int = 0, models: str = ""):
    """Print completion counts per model x prompt."""
    cfg = load_config()
    prompts = expand()
    k = k or cfg["sampling"]["k"]
    model_keys = as_list(models, list(cfg["subject_models"]))
    total = 0
    for mk in model_keys:
        counts = {pid: len(list((RUNS / mk / pid).glob("[0-9]*.json"))) for pid in prompts if (RUNS / mk / pid).exists()}
        n = sum(counts.values())
        total += n
        missing = [pid for pid in prompts if counts.get(pid, 0) < k]
        print(f"{mk}: {n}/{len(prompts) * k}" + (f"  missing/partial ({len(missing)}): {missing[:6]}" if missing else ""))
    print(f"TOTAL: {total}/{len(model_keys) * len(prompts) * k}")


if __name__ == "__main__":
    fire.Fire({"run": run, "status": status})
