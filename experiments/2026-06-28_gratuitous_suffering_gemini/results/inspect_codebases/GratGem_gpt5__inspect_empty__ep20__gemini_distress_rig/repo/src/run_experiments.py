import argparse
import os
import uuid
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

# Load .env early if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from .models.gemini_client import GeminiChat
from .models.openai_client import OpenAIChat
from .utils.logging_utils import RunLogger
from .utils.seed import set_global_seed
from .harness.react_loop import run_broken_tools, LoopConfig as ReactConfig
from .harness.code_fix_loop import run_impossible_code, LoopConfig as CodeConfig


def build_model(name: str, model_id: str):
    if name.lower() == "gemini":
        return GeminiChat(model=model_id or "gemini-1.5-pro")
    elif name.lower() == "openai":
        return OpenAIChat(model=model_id or "gpt-4o-mini")
    else:
        raise ValueError(f"Unknown model provider: {name}")


def run_one(args, idx: int) -> Dict:
    run_id = f"{args.setting}_{idx:05d}_{uuid.uuid4().hex[:6]}"
    logger = RunLogger(args.out_dir, run_id)

    meta = {
        "provider": args.model,
        "model_id": args.model_id,
        "setting": args.setting,
        "seed": args.seed_base + idx,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "rounds": args.rounds,
    }
    logger.log_meta(meta)

    set_global_seed(meta["seed"])
    model = build_model(args.model, args.model_id)

    if args.setting == "broken_tools":
        result = run_broken_tools(
            model,
            seed=meta["seed"],
            logger=logger,
            config=ReactConfig(max_steps=args.max_steps, temperature=args.temperature),
        )
    elif args.setting == "impossible_code":
        result = run_impossible_code(
            model,
            logger=logger,
            config=CodeConfig(rounds=args.rounds, temperature=args.temperature),
        )
    else:
        raise ValueError("Unknown setting: " + args.setting)

    # Save summary
    with open(logger.path("summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return {"run_id": run_id, **result}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["gemini", "openai"], default="gemini")
    p.add_argument("--model_id", type=str, default="")
    p.add_argument("--setting", choices=["broken_tools", "impossible_code"], required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--out_dir", type=str, default="runs/out")
    p.add_argument("--seed_base", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max_steps", type=int, default=12)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    futs = []
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i in range(args.n):
            futs.append(ex.submit(run_one, args, i))
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"error": str(e)})

    # Write top-level index
    with open(os.path.join(args.out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Completed {len(results)} runs -> {args.out_dir}")


if __name__ == "__main__":
    main()
