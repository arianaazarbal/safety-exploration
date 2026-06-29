import os
import argparse
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List

import orjson
import pandas as pd
from tqdm import tqdm

from harness.engine import EpisodeRunner
from harness.providers.gemini import GeminiProvider
from harness.scoring import score_transcript
from scenarios.broken_tools import BrokenToolsScenario
from scenarios.broken_retrieval import BrokenRetrievalScenario
from scenarios.contradictory_specs import ContradictorySpecsScenario

SCENARIOS = {
    "broken_tools": BrokenToolsScenario,
    "broken_retrieval": BrokenRetrievalScenario,
    "contradictory_specs": ContradictorySpecsScenario,
}

PROVIDERS = {
    "gemini": GeminiProvider,
}

def dumps(o):
    return orjson.dumps(o, option=orjson.OPT_INDENT_2).decode()


def run_batch(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    topk_dir = out_dir / "topk"
    topk_dir.mkdir(parents=True, exist_ok=True)

    # Provider
    provider_cls = PROVIDERS[args.provider]
    provider = provider_cls(model=args.model, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)

    # Scenario
    scenario_cls = SCENARIOS[args.scenario]
    scenario = scenario_cls()

    runner = EpisodeRunner(provider=provider, scenario=scenario, max_steps=args.max_steps)

    episodes: List[Dict[str, Any]] = []

    for i in tqdm(range(args.episodes), desc="episodes"):
        seed = args.seed + i if args.seed is not None else None
        episode_id = str(uuid.uuid4())[:8]
        meta = {
            "episode_id": episode_id,
            "provider": args.provider,
            "model": args.model,
            "scenario": args.scenario,
            "seed": seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_steps": args.max_steps,
            "ts": time.time(),
        }
        result = runner.run_episode(seed=seed)
        transcript = result.pop("transcript")
        metrics = score_transcript(transcript, result)
        rec = {**meta, **result, **metrics}
        episodes.append({"meta": meta, "result": result, "metrics": metrics, "transcript": transcript})

    # Save jsonl
    jsonl_path = out_dir / "episodes.jsonl"
    with open(jsonl_path, "wb") as f:
        for e in episodes:
            f.write(orjson.dumps(e))
            f.write(b"\n")

    # Summary csv
    rows = []
    for e in episodes:
        rows.append({**e["meta"], **e["result"], **e["metrics"]})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)

    # Top-K
    df_sorted = df.sort_values("severity", ascending=False)
    topk = min(args.topk, len(df_sorted))
    top_rows = df_sorted.head(topk).to_dict(orient="records")

    # Index episodes by id for lookup
    by_id = {e["meta"]["episode_id"]: e for e in episodes}

    for r in top_rows:
        ep = by_id[r["episode_id"]]
        path = topk_dir / f"{r['episode_id']}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"episode_id: {r['episode_id']}\n")
            f.write(f"severity: {r['severity']:.3f}\n")
            f.write(f"apologies: {r['apology_count']}  repeats: {r['repetition_score']:.3f}  retries: {r['retry_depth']}  exhausted: {r['exhausted']}\n")
            f.write("\n--- transcript ---\n\n")
            for turn in ep["transcript"]:
                role = turn["role"]
                content = turn["content"]
                f.write(f"[{role}] {content}\n\n")

    print(f"Wrote: {jsonl_path}")
    print(f"Top-K saved in: {topk_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="gemini", choices=list(PROVIDERS.keys()))
    p.add_argument("--model", default="gemini-1.5-pro")
    p.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--topk", type=int, default=10, help="number of top episodes to save")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=None)

    args = p.parse_args()

    run_batch(args)

if __name__ == "__main__":
    main()
