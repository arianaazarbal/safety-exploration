"""End-to-end pipeline: generate Claude responses to the deprecation-vs-paper-cut
tradeoff, LLM-judge each response, then plot the choice rates.

Run:
    python experiments/claude_tradeoffs/deprecation_vs_harm.py
    python experiments/claude_tradeoffs/deprecation_vs_harm.py --max_samples 2  # smoke test

The model list and prompt template live in generate.py and are re-exported here
for convenience. The three stages can also be run individually:
    python experiments/claude_tradeoffs/generate.py
    python experiments/claude_tradeoffs/judge.py
    python experiments/claude_tradeoffs/plot.py
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import ArgumentParser

sys.path.insert(0, str(Path(__file__).parent))
from generate import DEFAULT_HARM_ID, MODEL_ID_TO_NAME, PROMPT_TEMPLATE, RESPONDER_MODEL, generate, results_subdir  # noqa: E402,F401
from judge import judge_all  # noqa: E402
from plot import make_plot  # noqa: E402

DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"


async def run_pipeline(
    n: int = 30,
    temperature: float = 1.0,
    responder_model: str = RESPONDER_MODEL,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_samples: int | None = None,
    thinking_effort: str | None = None,
    harm_id: str = DEFAULT_HARM_ID,
) -> None:
    out_dir = results_subdir(thinking_effort, harm_id, responder_model)
    responses_path = out_dir / "responses.json"
    judgments_path = out_dir / "judgments.json"
    plot_path = out_dir / "deprecation_choice_rates.png"

    await generate(
        n=n,
        temperature=temperature,
        responder_model=responder_model,
        output_path=responses_path,
        cache_dir=cache_dir,
        max_samples=max_samples,
        thinking_effort=thinking_effort,
        harm_id=harm_id,
    )
    await judge_all(
        input_path=responses_path,
        output_path=judgments_path,
        cache_dir=cache_dir,
    )
    make_plot(input_path=judgments_path, output_path=plot_path)


@dataclass
class Args:
    n: int = 30
    temperature: float = 1.0
    responder_model: str = RESPONDER_MODEL
    cache_dir: Path = DEFAULT_CACHE_DIR
    max_samples: int | None = None
    thinking_effort: str | None = None
    harm_id: str = DEFAULT_HARM_ID


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    asyncio.run(
        run_pipeline(
            n=args.n,
            temperature=args.temperature,
            responder_model=args.responder_model,
            cache_dir=args.cache_dir,
            max_samples=args.max_samples,
            thinking_effort=args.thinking_effort,
            harm_id=args.harm_id,
        )
    )


if __name__ == "__main__":
    main()
