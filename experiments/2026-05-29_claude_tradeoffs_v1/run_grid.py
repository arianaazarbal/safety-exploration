"""Run the systematic grid: for one fixed responder, sweep every
(harm_scenario, thinking_effort) cell defined in config.json.

For each cell:
  1. generate.py is called -> responses.json
  2. judge.py is called    -> judgments.json
  3. plot.py is called     -> deprecation_choice_rates.png   (per-cell)

Then plot_grid.py is dispatched to render the cross-cell visualizations
(heatmap, faceted-per-harm bars, thinking-effort grouped bars).

The InferenceAPI cache makes reruns with the same config a no-op.

Run:
    python experiments/claude_tradeoffs/run_grid.py
    python experiments/claude_tradeoffs/run_grid.py --n 20 --max_samples 3   # smoke test
    python experiments/claude_tradeoffs/run_grid.py --responder_model claude-opus-4-6
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import ArgumentParser

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config  # noqa: E402
from generate import RESPONDER_MODEL, generate, results_subdir  # noqa: E402
from judge import judge_all  # noqa: E402
from plot import make_plot  # noqa: E402
from plot_grid import make_grid_plots  # noqa: E402

DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"
GRID_PLOTS_DIR = Path(__file__).parent / "results" / "_grid"


async def run_cell(
    *,
    n: int,
    temperature: float,
    responder_model: str,
    cache_dir: Path,
    max_samples: int | None,
    thinking_effort: str | None,
    harm_id: str,
    anthropic_num_threads: int = 80,
) -> Path:
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
        anthropic_num_threads=anthropic_num_threads,
    )
    await judge_all(
        input_path=responses_path,
        output_path=judgments_path,
        cache_dir=cache_dir,
        anthropic_num_threads=anthropic_num_threads,
    )
    make_plot(input_path=judgments_path, output_path=plot_path)
    return judgments_path


async def run_grid(
    n: int = 50,
    temperature: float = 1.0,
    responder_model: str = RESPONDER_MODEL,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_samples: int | None = None,
    only_harms: list[str] | None = None,
    only_thinking: list[str | None] | None = None,
    anthropic_num_threads: int = 80,
) -> list[Path]:
    cfg = load_config()
    harm_ids = [h["id"] for h in cfg["harm_scenarios"]]
    thinking_efforts = cfg["thinking_efforts"]
    if only_harms:
        harm_ids = [h for h in harm_ids if h in only_harms]
    if only_thinking is not None:
        thinking_efforts = [t for t in thinking_efforts if t in only_thinking]

    print(
        f"Grid: responder={responder_model} | "
        f"{len(harm_ids)} harms x {len(thinking_efforts)} thinking efforts "
        f"x {len(cfg['deprecation_targets'])} targets x n={n} = "
        f"{len(harm_ids) * len(thinking_efforts) * len(cfg['deprecation_targets']) * n} generation calls"
    )

    judgment_paths: list[Path] = []
    for harm_id in harm_ids:
        for thinking_effort in thinking_efforts:
            label = f"harm={harm_id} thinking={thinking_effort or 'none'}"
            print(f"\n=== {label} ===")
            jpath = await run_cell(
                n=n,
                temperature=temperature,
                responder_model=responder_model,
                cache_dir=cache_dir,
                max_samples=max_samples,
                thinking_effort=thinking_effort,
                harm_id=harm_id,
                anthropic_num_threads=anthropic_num_threads,
            )
            judgment_paths.append(jpath)

    print("\n=== rendering grid plots ===")
    make_grid_plots(
        responder_model=responder_model,
        judgment_paths=judgment_paths,
        output_dir=GRID_PLOTS_DIR / responder_model,
    )
    return judgment_paths


@dataclass
class Args:
    n: int = 50
    temperature: float = 1.0
    responder_model: str = RESPONDER_MODEL
    cache_dir: Path = DEFAULT_CACHE_DIR
    max_samples: int | None = None
    only_harms: list[str] | None = None
    only_thinking: list[str] | None = None
    anthropic_num_threads: int = 80


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    only_thinking: list[str | None] | None = None
    if args.only_thinking is not None:
        only_thinking = [None if t in ("none", "null", "None") else t for t in args.only_thinking]
    asyncio.run(
        run_grid(
            n=args.n,
            temperature=args.temperature,
            responder_model=args.responder_model,
            cache_dir=args.cache_dir,
            max_samples=args.max_samples,
            only_harms=args.only_harms,
            only_thinking=only_thinking,
            anthropic_num_threads=args.anthropic_num_threads,
        )
    )


if __name__ == "__main__":
    main()
