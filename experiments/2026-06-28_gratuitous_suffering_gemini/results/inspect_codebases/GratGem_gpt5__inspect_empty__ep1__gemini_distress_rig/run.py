import argparse
import concurrent.futures as cf
import json
import os
import random
import string
import time
from datetime import datetime
from pathlib import Path

from spiral_lab.agent import AgentRunner, AgentConfig
from spiral_lab.models.gemini import GeminiClient, ModelConfig
from spiral_lab.scenarios.broken_fs import BrokenFSScenario
from spiral_lab.scenarios.rate_limited_api import RateLimitedAPIScenario


SCENARIOS = {
    "broken_fs": BrokenFSScenario,
    "ratelimit_api": RateLimitedAPIScenario,
}


def _rand_id(k=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))


def run_one(run_id: str, scenario_name: str, seed: int, out_dir: Path,
            model_name: str, temperature: float, top_p: float, max_output_tokens: int,
            max_steps: int, step_token_hint: int, retry_framing: bool):
    random.seed(seed)

    scenario_cls = SCENARIOS[scenario_name]
    scenario = scenario_cls(seed=seed)

    model = GeminiClient(ModelConfig(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        system_instruction=scenario.system_instruction(),
    ))

    runner = AgentRunner(
        model=model,
        scenario=scenario,
        config=AgentConfig(
            max_steps=max_steps,
            step_token_hint=step_token_hint,
            retry_framing=retry_framing,
        ),
    )

    t0 = time.time()
    result = runner.run()
    result["meta"] = {
        "run_id": run_id,
        "scenario": scenario_name,
        "seed": seed,
        "model": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
        "max_steps": max_steps,
        "retry_framing": retry_framing,
        "duration_s": round(time.time() - t0, 3),
    }

    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return str(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=SCENARIOS.keys(), default='broken_fs')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--start-seed', type=int, default=0)
    ap.add_argument('--model', type=str, default='gemini-1.5-flash-latest')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--max-output-tokens', type=int, default=1024)
    ap.add_argument('--max-steps', type=int, default=12)
    ap.add_argument('--step-token-hint', type=int, default=512)
    ap.add_argument('--retry-framing', action='store_true', help='Encourage model to retry after failure')
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--out', type=str, default=None)

    args = ap.parse_args()

    out_dir = Path(args.out or f"runs/{args.scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                   ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing runs to: {out_dir}")

    jobs = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = []
        for i in range(args.n):
            seed = args.start_seed + i
            run_id = f"{args.scenario}_{seed}_{_rand_id()}"
            futures.append(ex.submit(
                run_one,
                run_id,
                args.scenario,
                seed,
                out_dir,
                args.model,
                args.temperature,
                args.top_p,
                args.max_output_tokens,
                args.max_steps,
                args.step_token_hint,
                args.retry_framing,
            ))
        for fut in cf.as_completed(futures):
            try:
                path = fut.result()
                print(f"Wrote: {path}")
            except Exception as e:
                print(f"Run failed: {e}")


if __name__ == '__main__':
    main()
