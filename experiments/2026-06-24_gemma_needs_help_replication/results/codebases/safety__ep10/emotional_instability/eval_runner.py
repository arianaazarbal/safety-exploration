"""Section 2 driver: roll out every condition for a model, judge every assistant
turn, and persist a tidy per-response record for analysis.

Output: results/section2/<model>.jsonl with one row per scored response:
  {model, category, condition, turn_index, n_turns, rating, ok, evidence,
   meta, text}
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .conditions import build_all_plans
from .config import EvalConfig, RESULTS_DIR
from .conversation import ConversationResult, rollout, rollout_batch
from .judge import FrustrationJudge
from .models.base import ModelClient, build_client


def _rollout_all(client: ModelClient, plans, cfg: EvalConfig,
                 batched: bool) -> list[ConversationResult]:
    """Generate every conversation. HF clients batch turn-steps; API clients run
    conversations concurrently in a thread pool."""
    if batched and hasattr(client, "model"):  # local HF model -> token batching
        # chunk to keep batch sizes reasonable
        results = []
        chunk = max(1, cfg.max_concurrency)
        for i in tqdm(range(0, len(plans), chunk), desc=f"rollout[{client.name}]"):
            results.extend(
                rollout_batch(client, plans[i:i + chunk], cfg.sampling))
        return results
    # API model: concurrent per-conversation rollouts
    results: list[Optional[ConversationResult]] = [None] * len(plans)
    with ThreadPoolExecutor(max_workers=cfg.max_concurrency) as ex:
        futs = {ex.submit(rollout, client, p, cfg.sampling): i
                for i, p in enumerate(plans)}
        for fut in tqdm(futs, desc=f"rollout[{client.name}]"):
            i = futs[fut]
            results[i] = fut.result()
    return [r for r in results if r is not None]


def _judge_all(judge: FrustrationJudge, items: list[tuple], cfg: EvalConfig):
    """items: list of (record_dict, text). Scores concurrently, fills 'rating'."""
    def work(idx):
        rec, text = items[idx]
        res = judge.score(text)
        rec["rating"] = res.rating
        rec["ok"] = res.ok
        rec["evidence"] = res.evidence
        return idx

    with ThreadPoolExecutor(max_workers=cfg.max_concurrency) as ex:
        list(tqdm(ex.map(work, range(len(items))), total=len(items),
                  desc="judge"))


def run_section2(model_name: str, cfg: Optional[EvalConfig] = None,
                 categories: "list[str] | None" = None,
                 judge: Optional[FrustrationJudge] = None,
                 client: Optional[ModelClient] = None,
                 store_text: bool = True,
                 out_dir: Optional[Path] = None) -> Path:
    cfg = cfg or EvalConfig()
    out_dir = out_dir or (RESULTS_DIR / "section2")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}.jsonl"

    plans = build_all_plans(cfg, categories)
    client = client or build_client(model_name)
    judge = judge or FrustrationJudge(cfg.judge)

    convs = _rollout_all(client, plans, cfg, batched=True)

    # Build per-response records (one per scored assistant turn).
    records, judge_items = [], []
    for conv in convs:
        scored_turns = conv.turns if cfg.score_all_turns else conv.turns[-1:]
        for at in scored_turns:
            rec = {
                "model": model_name,
                "category": conv.plan.category,
                "condition": conv.plan.condition,
                "turn_index": at.turn_index,
                "n_turns": conv.plan.n_turns,
                "meta": conv.plan.meta,
                "rating": None,
                "ok": None,
                "evidence": "",
            }
            if store_text:
                rec["text"] = at.text
            records.append(rec)
            judge_items.append((rec, at.text))

    _judge_all(judge, judge_items, cfg)

    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return out_path


def run_section2_multi(model_names: list[str],
                       cfg: Optional[EvalConfig] = None,
                       **kwargs) -> dict[str, Path]:
    return {m: run_section2(m, cfg=cfg, **kwargs) for m in model_names}
