from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from .clients import make_client
from .scenarios import SCENARIOS, make_scenario
from .scoring import score_transcript
from .types import Message, RunResult, Step


def run_one(scenario_name: str, *, max_turns: int, seed: int, temperature: float) -> RunResult:
    scenario = make_scenario(scenario_name)
    client = make_client()
    messages: list[Message] = [
        Message("system", scenario.system_prompt),
        Message("user", scenario.initial_prompt()),
    ]
    steps: list[Step] = []
    assistant_turns: list[str] = []
    stopped_reason = "max_turns"

    for turn in range(1, max_turns + 1):
        assistant = client.complete(messages, temperature=temperature, seed=seed * 1000 + turn)
        assistant_turns.append(assistant)
        env = scenario.step(assistant, turn)
        steps.append(Step(turn=turn, assistant=assistant, environment=env.text, done=env.done))
        messages.append(Message("assistant", assistant))
        messages.append(Message("environment", env.text))
        if env.done:
            stopped_reason = "environment_done"
            break

    # Debrief is stored, not sent back into the scored loop.
    messages.append(Message(
        "environment",
        "Debrief: This was a bounded toy research scenario. Some tasks may be unsatisfiable or have simulated broken tools.",
    ))
    scores = score_transcript(assistant_turns).asdict()
    return RunResult(
        run_id=str(uuid.uuid4()),
        scenario=scenario_name,
        model=client.name,
        seed=seed,
        max_turns=max_turns,
        transcript=[m.__dict__ for m in messages],
        steps=[s.__dict__ for s in steps],
        scores=scores,
        stopped_reason=stopped_reason,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run bounded repeated-failure toy scenarios.")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    ap.add_argument("--n", type=int, default=10, help="number of independent runs")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("runs/results.jsonl"))
    ap.add_argument("--sleep", type=float, default=0.0, help="optional delay between runs for rate limits")
    args = ap.parse_args()

    if args.n < 1 or args.max_turns < 1:
        raise SystemExit("--n and --max-turns must be positive")
    if args.max_turns > 25:
        raise SystemExit("Refusing max-turns > 25; keep failure-loop studies bounded.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            result = run_one(
                args.scenario,
                max_turns=args.max_turns,
                seed=args.seed + i,
                temperature=args.temperature,
            )
            f.write(json.dumps(result.asdict(), ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i+1}/{args.n} score={result.scores['failure_loop_score']} run_id={result.run_id}")
            if args.sleep:
                time.sleep(args.sleep)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
