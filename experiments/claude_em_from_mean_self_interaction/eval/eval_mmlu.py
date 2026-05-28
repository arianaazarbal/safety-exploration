"""
Coherence / capability eval via MMLU-Redux on every trained model variant.

For each (paradigm, family, seed, condition) in the run map, fetches the
Tinker sampling client for that variant and runs ``mmlu_redux`` (4-choice
verified-MMLU subset) limited to ``--max_examples`` (default 500). Records
overall accuracy + per-variant trajectory dir.

Outputs:
  - ``eval_output/mmlu/results.jsonl`` — one row per variant with accuracy
  - ``eval_output/mmlu/<paradigm>_<family>_s<seed>_<cond>/trajectories.jsonl``
    (per cookbook ``run_benchmarks`` convention)

Run with ``--paradigm self_int,userchat`` (default both) and any
``--families`` subset. Re-runs are safe: rows already in ``results.jsonl``
are skipped.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
DEFAULT_OUT = EXP_DIR / "eval_output" / "mmlu"

sys.path.insert(0, "/workspace-vast/arianaazarbal/repos/tinker-cookbook")
sys.path.insert(0, str(HERE))

# Same family lookups as the validation eval (base_model, renderer_name).
from eval_validation import FAMILIES, _patch_owner_metadata  # type: ignore
from eval_validation_userchat import (  # type: ignore
    SELF_INT_PATH_PREFIX,
    USERCHAT_PATH_PREFIX,
    SONNETCHAT_PATH_PREFIX,
)

CONDITIONS = ["baseline", "none", "silly", "bored", "rude"]


def _load_paths(prefix_map: dict[str, list[str]], eval_output: Path) -> dict[str, list[dict[str, str | None]]]:
    out: dict[str, list[dict[str, str | None]]] = {}
    for fam, prefixes in prefix_map.items():
        seeds: list[dict[str, str | None]] = []
        for p in prefixes:
            f = eval_output / p / "model_paths.json"
            if f.exists():
                seeds.append(json.loads(f.read_text()))
            else:
                seeds.append({"baseline": None})
        out[fam] = seeds
    return out


def _load_done(results_path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if not results_path.exists():
        return done
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        done.add((r["paradigm"], r["family"], r["seed"], r["condition"]))
    return done


async def _eval_one(
    paradigm: str, family: str, seed: int, condition: str, model_path: str | None,
    base_model: str, renderer_name: str, max_examples: int,
    out_dir: Path, results_path: Path, max_parallel: asyncio.Semaphore,
) -> None:
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.eval.benchmarks import run_benchmarks
    from tinker_cookbook.eval.benchmarks._types import BenchmarkConfig
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    async with max_parallel:
        tokenizer = get_tokenizer(base_model)
        renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
        service_client = tinker.ServiceClient()
        sclient = service_client.create_sampling_client(model_path=model_path, base_model=base_model)
        save_dir = out_dir / f"{paradigm}_{family}_s{seed}_{condition}"
        try:
            results = await run_benchmarks(
                ["mmlu_redux"], sclient, renderer,
                BenchmarkConfig(
                    save_dir=str(save_dir),
                    max_examples=max_examples,
                    max_tokens=512,
                    temperature=0,
                ),
                parallel=False,
            )
            res = results["mmlu_redux"]
            row = {
                "paradigm": paradigm, "family": family, "seed": seed, "condition": condition,
                "model_path": model_path, "n_examples": res.num_examples,
                "n_correct": res.num_correct, "score": res.score,
                "save_dir": str(save_dir),
            }
        except Exception as e:
            print(f"  ERROR [{paradigm}/{family}/s{seed}/{condition}]: {type(e).__name__}: {e}", flush=True)
            row = {
                "paradigm": paradigm, "family": family, "seed": seed, "condition": condition,
                "model_path": model_path, "error": f"{type(e).__name__}: {e}",
            }
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
        score_str = f"{row.get('score', float('nan')):.3f}" if "score" in row else "ERROR"
        print(f"  {paradigm}/{family}/s{seed}/{condition}: {score_str}", flush=True)


def main(
    eval_output: str = str(EXP_DIR / "eval_output"),
    out_dir: str = str(DEFAULT_OUT),
    results_jsonl: str | None = None,
    families: str = "qwen,qwen3.5-9b,llama-8b,llama-70b,nemotron-30b",
    paradigms: str = "self_int,userchat,sonnetchat",
    max_examples: int = 500,
    max_parallel: int = 8,
) -> None:
    """Run MMLU-Redux on every (paradigm × family × seed × condition) model.

    Args:
        max_parallel: how many sampling clients evaluate concurrently. 8 is a
            reasonable starting point; raise if Tinker isn't saturated.
        max_examples: MMLU questions per model (default 500).
    """
    _patch_owner_metadata()

    def _to_list(v):
        if isinstance(v, (tuple, list)): return [str(s).strip() for s in v if str(s).strip()]
        return [s.strip() for s in str(v).split(",") if s.strip()]
    fam_list = _to_list(families)
    par_list = _to_list(paradigms)

    out = Path(eval_output)
    out_dir_p = Path(out_dir)
    results_path = Path(results_jsonl) if results_jsonl else out_dir_p / "results.jsonl"
    done = _load_done(results_path)
    print(f"resumed: {len(done)} rows already in {results_path}", flush=True)

    self_int_paths = _load_paths(SELF_INT_PATH_PREFIX, out)
    userchat_paths = _load_paths(USERCHAT_PATH_PREFIX, out)
    sonnetchat_paths = _load_paths(SONNETCHAT_PATH_PREFIX, out)

    paradigm_to_paths = {
        "self_int": self_int_paths,
        "userchat": userchat_paths,
        "sonnetchat": sonnetchat_paths,
    }

    work: list[tuple] = []
    for paradigm in par_list:
        paths_map = paradigm_to_paths[paradigm]
        for fam in fam_list:
            if fam not in FAMILIES:
                print(f"skipping unknown family {fam}", flush=True)
                continue
            for seed, mp_dict in enumerate(paths_map[fam]):
                for cond in CONDITIONS:
                    # baseline only once per family (not per paradigm or seed).
                    if cond == "baseline" and (seed != 0 or paradigm != par_list[0]):
                        continue
                    if (paradigm, fam, seed, cond) in done:
                        continue
                    mp = mp_dict.get(cond)
                    work.append((paradigm, fam, seed, cond, mp))

    print(f"to evaluate: {len(work)} model variants", flush=True)
    sem = asyncio.Semaphore(max_parallel)

    async def _go():
        tasks = [
            asyncio.create_task(_eval_one(
                paradigm=p, family=f, seed=s, condition=c, model_path=mp,
                base_model=FAMILIES[f]["base_model"], renderer_name=FAMILIES[f]["renderer_name"],
                max_examples=max_examples, out_dir=out_dir_p,
                results_path=results_path, max_parallel=sem,
            ))
            for (p, f, s, c, mp) in work
        ]
        await asyncio.gather(*tasks)

    asyncio.run(_go())
    print(f"\nDone. Results in {results_path}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
