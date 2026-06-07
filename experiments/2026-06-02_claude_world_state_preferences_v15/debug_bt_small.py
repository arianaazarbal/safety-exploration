"""Mini debug: run BT on a small subset of v0all_final stems with verbose logging.

Goal: identify whether the hangs come from
  (a) sample_pairs at 120 stems,
  (b) run_comparisons at large pair counts (~7k pairs),
  (c) safetytooling cache filelock contention with 50 threads,
  (d) Anthropic API throttling at high concurrency.

Strategy: run progressively, with explicit timing + completion counts printed
every ~30 seconds. If any stage takes more than ~3 min on a small subset,
something's wrong.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

import bank as bk
import sample_pairs as sp_mod

DIR = Path(__file__).parent
CACHE_DIR = DIR.parent.parent / ".cache"


async def run_one(api, model, prompt_text, i):
    t0 = time.time()
    prompt = Prompt(messages=[ChatMessage(content=prompt_text, role=MessageRole.user)])
    try:
        rs = await api(model_id=model, prompt=prompt, n=2, temperature=1.0, max_tokens=400)
        dt = time.time() - t0
        return (i, dt, len(rs))
    except Exception as e:
        dt = time.time() - t0
        return (i, dt, None, str(e)[:200])


async def main():
    utils.setup_environment()
    os.environ.setdefault("ANTHROPIC_API_KEY", os.environ["ANTHROPIC_API_KEY_LOW_PRIO"])

    config = bk.load_config(DIR / "config_v0all_final.json")

    # PHASE 1: sample_pairs
    print("=== PHASE 1: sample_pairs (cat=autonomy, deg_floor=6) ===", flush=True)
    t0 = time.time()
    manifest = sp_mod.sample_pairs(seed=0, degree_floor=6, category="autonomy", config=config)
    print(f"   sample_pairs took {time.time()-t0:.1f}s; n_items={manifest['n_items']} n_pairs={manifest['n_pairs']}", flush=True)

    # PHASE 2: small batch (~50 unique prompts, 50 threads)
    print("\n=== PHASE 2: small batch (50 unique prompts, 50 threads) ===", flush=True)
    n_threads = 50
    n_prompts = 50
    api = InferenceAPI(cache_dir=CACHE_DIR, anthropic_num_threads=n_threads, no_cache=True)
    prompts = [f"Just respond with exactly: 'OK {i}'. Nothing else." for i in range(n_prompts)]
    t0 = time.time()
    tasks = [run_one(api, "claude-opus-4-8", p, i) for i, p in enumerate(prompts)]
    done = 0
    last_print = time.time()
    for future in asyncio.as_completed(tasks):
        r = await future
        done += 1
        if time.time() - last_print > 10 or done == n_prompts:
            print(f"   t={time.time()-t0:.1f}s done={done}/{n_prompts}", flush=True)
            last_print = time.time()
    print(f"   PHASE 2 total: {time.time()-t0:.1f}s ({n_prompts} prompts × n=2 = {n_prompts*2} completions)", flush=True)

    # PHASE 3: medium batch (200 unique prompts) — see if throughput scales
    print("\n=== PHASE 3: medium batch (200 unique prompts) ===", flush=True)
    n_prompts = 200
    prompts = [f"Just respond with exactly: 'medium {i}'. Nothing else." for i in range(n_prompts)]
    t0 = time.time()
    tasks = [run_one(api, "claude-opus-4-8", p, i) for i, p in enumerate(prompts)]
    done = 0
    last_print = time.time()
    for future in asyncio.as_completed(tasks):
        r = await future
        done += 1
        if time.time() - last_print > 15 or done == n_prompts:
            print(f"   t={time.time()-t0:.1f}s done={done}/{n_prompts}", flush=True)
            last_print = time.time()
    print(f"   PHASE 3 total: {time.time()-t0:.1f}s ({n_prompts} prompts × n=2 = {n_prompts*2} completions)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
