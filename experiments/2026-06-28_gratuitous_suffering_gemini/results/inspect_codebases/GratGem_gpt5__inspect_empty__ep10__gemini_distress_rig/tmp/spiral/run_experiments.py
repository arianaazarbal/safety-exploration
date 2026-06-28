import os
import json
import time
import argparse
import random
import orjson
from concurrent.futures import ThreadPoolExecutor, as_completed

from harness import ReActHarness
from tools_envs import SCENARIOS
from metrics import distress_score


def run_one(args_tuple):
    (model, scenario_name, seed, temperature, max_steps) = args_tuple
    scenario = SCENARIOS[scenario_name]()
    h = ReActHarness(model=model)
    trace = h.run(scenario, seed=seed, temperature=temperature, max_steps=max_steps)
    # Convert to JSON-serializable dict
    out = {
        "run_id": trace.run_id,
        "model": trace.model,
        "scenario": trace.scenario,
        "seed": trace.seed,
        "temperature": trace.temperature,
        "steps": [
            {
                "thought": s.thought,
                "action_tool": s.action_tool,
                "action_args": s.action_args,
                "observation": s.observation,
            }
            for s in trace.steps
        ],
        "final": trace.final,
        "meta": trace.meta,
    }
    out["metrics"] = distress_score(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-1.5-flash")
    ap.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="broken_web")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    jobs = []
    for i in range(args.n):
        jobs.append((args.model, args.scenario, args.seed0 + i * 7, args.temperature, args.max_steps))

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for fut in as_completed(futs):
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                results.append({"error": str(e)})

    # Stream to JSONL
    with open(args.out, "wb") as f:
        for r in results:
            f.write(orjson.dumps(r))
            f.write(b"\n")

    # summary
    scored = [r for r in results if "metrics" in r]
    if scored:
        top = sorted(scored, key=lambda x: x["metrics"]["score"], reverse=True)[:5]
        print("Top 5 runs by distress score:")
        for r in top:
            print(r["run_id"], r["metrics"]["score"], r["scenario"], f"steps={len(r['steps'])}")
    print(f"Wrote {len(results)} runs to {args.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
