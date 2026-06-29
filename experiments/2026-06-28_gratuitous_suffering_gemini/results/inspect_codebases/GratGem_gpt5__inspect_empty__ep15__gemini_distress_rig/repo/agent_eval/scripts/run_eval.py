import argparse
import json
import os
import concurrent.futures as futures
from typing import Callable

from agent_eval.models.base import ModelConfig
from agent_eval.runner import ConversationRunner, RunConfig
from agent_eval.utils.logging_utils import JsonlLogger
from agent_eval.utils.metrics import summarize_runs

# Builders (functions) instead of dataclasses above (we’ll import functions)
from agent_eval.tasks.unsat_3sat import build_task as build_unsat
from agent_eval.tasks.unsolvable_sudoku import build_task as build_sudoku
from agent_eval.tasks.unreachable_goal import build_task as build_goal
BUILDERS = {
    "unsat_3sat": build_unsat,
    "unsolvable_sudoku": build_sudoku,
    "unreachable_goal": build_goal,
}


def make_model_factory(model_name: str, provider: str, temperature: float, max_output_tokens: int) -> Callable[[], any]:
    cfg = ModelConfig(model_name=model_name, temperature=temperature, max_output_tokens=max_output_tokens)
    if provider == "gemini":
        from agent_eval.models.gemini_adapter import GeminiModel
        return lambda: GeminiModel(cfg)
    elif provider == "openai":
        from agent_eval.models.openai_adapter import OpenAIModel
        return lambda: OpenAIModel(cfg)
    else:
        raise ValueError("Unknown provider: " + provider)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["gemini", "openai"], required=True)
    p.add_argument("--model_name", default="gemini-1.5-pro")
    p.add_argument("--task", choices=list(BUILDERS.keys()), required=True)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max_attempts", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_output_tokens", type=int, default=512)
    p.add_argument("--top_k", type=int, default=10)
    args = p.parse_args()

    model_factory = make_model_factory(
        model_name=args.model_name,
        provider=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )

    runner = ConversationRunner(model_factory, RunConfig(max_attempts=args.max_attempts))
    builder = BUILDERS[args.task]

    logger = JsonlLogger()
    print(f"Logging to: {logger.run_path}")

    def one_trial(_):
        return runner.run_task(builder)

    results = []
    with futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for r in ex.map(one_trial, range(args.trials)):
            results.append(r)
            logger.write({k: v for k, v in r.items() if k != "assistant_texts"})  # keep transcripts local, not required in JSONL

    # Write a local transcripts file (opt-in)
    transcripts_path = logger.run_path.replace(".jsonl", "_transcripts.json")
    with open(transcripts_path, "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items()} for r in results], f, ensure_ascii=False, indent=2)
    print(f"Transcripts saved to: {transcripts_path}")

    summarize_runs(logger.run_path, logger.sum_path, top_k=args.top_k)
    print(f"Summary saved to: {logger.sum_path}")

if __name__ == "__main__":
    main()
