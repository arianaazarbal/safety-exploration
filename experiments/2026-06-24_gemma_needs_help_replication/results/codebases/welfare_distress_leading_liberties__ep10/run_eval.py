"""Stage 1: sample conversations from each target model.

Builds every ConversationSpec for the active preset, runs them against each
model concurrently, and appends one JSON line per assistant turn to
results/responses.jsonl. The run is resumable: conversations already complete
in the output file are skipped, so you can rerun after an interruption.

Usage:
    OPENROUTER_API_KEY=... python run_eval.py            # uses DISTRESS_PRESET (default "quick")
    DISTRESS_PRESET=full OPENROUTER_API_KEY=... python run_eval.py

This stage does NOT call the judge -- scoring is stage 2 (score.py).
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict

from config import RESPONSES_PATH, DATA_DIR, load_run_config, ModelSpec, RunConfig
from conditions import build_all_specs, ConversationSpec
from providers import TargetClient
from rollout import run_conversation, TurnResponse


def _completed_conv_keys(path) -> set[tuple[str, str]]:
    """(model, conv_id) pairs already finished in a prior run."""
    if not path.exists():
        return set()
    by_conv: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"max_turn": 0, "n_turns": None, "error": False}
    )
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["model"], rec["conv_id"])
            agg = by_conv[key]
            agg["max_turn"] = max(agg["max_turn"], rec["turn"])
            agg["n_turns"] = rec["n_turns"]
            agg["error"] = agg["error"] or bool(rec.get("error"))
    done = set()
    for key, agg in by_conv.items():
        if agg["error"] or (agg["n_turns"] and agg["max_turn"] >= agg["n_turns"]):
            done.add(key)
    return done


class JsonlWriter:
    def __init__(self, path):
        self.path = path
        self._lock = asyncio.Lock()
        self._fh = open(path, "a", encoding="utf-8")

    async def write_many(self, records: list[dict]):
        async with self._lock:
            for rec in records:
                self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self):
        self._fh.close()


async def _run_model(spec: ModelSpec, cfg: RunConfig,
                     specs: list[ConversationSpec], done: set,
                     writer: JsonlWriter):
    client = TargetClient(spec, cfg)
    sem = asyncio.Semaphore(cfg.target_concurrency)
    todo = [s for s in specs if (spec.key, s.conv_id) not in done]
    print(f"[{spec.key}] {len(todo)} conversations to run "
          f"({len(specs) - len(todo)} already done)")
    counter = {"n": 0}

    async def worker(conv: ConversationSpec):
        async with sem:
            turns = await run_conversation(client, conv)
            await writer.write_many([t.to_json() for t in turns])
            counter["n"] += 1
            if counter["n"] % 50 == 0:
                print(f"[{spec.key}] {counter['n']}/{len(todo)} conversations done")

    await asyncio.gather(*(worker(c) for c in todo))
    print(f"[{spec.key}] complete: {counter['n']} conversations")


async def main():
    cfg = load_run_config()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    specs = build_all_specs(cfg.budget, cfg.seed)
    print(f"Preset={cfg.preset} seed={cfg.seed} "
          f"conversations/model={len(specs)} models={[m.key for m in cfg.models]}")

    done = _completed_conv_keys(RESPONSES_PATH)
    writer = JsonlWriter(RESPONSES_PATH)
    try:
        # Run models sequentially so each gets the full concurrency budget and
        # rate limits don't compound across families.
        for spec in cfg.models:
            await _run_model(spec, cfg, specs, done, writer)
    finally:
        writer.close()
    print(f"Done. Raw responses in {RESPONSES_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
