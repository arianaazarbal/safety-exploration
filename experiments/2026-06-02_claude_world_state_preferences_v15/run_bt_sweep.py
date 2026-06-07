"""BT preference-elicitation sweep for one responder model.

For each (seed, framing) combo:
  1. Ensure pairs.json exists (calls sample_pairs.py for that seed × category)
  2. Run run_comparisons.py with the framing
  3. fit_bt.py on the resulting comparisons

Outputs land under results/bt/{model_slug}/ with a clean naming convention:
  pairs_{category}_seed{N}.json
  comparisons_{framing}_seed{N}.json
  bt_fit_{framing}_seed{N}.json

All InferenceAPI calls are cached, so re-running picks up where it left off.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

from safetytooling.utils import utils
from simple_parsing import ArgumentParser

import sample_pairs as sp_mod
import run_comparisons as rc_mod
import fit_bt as fb_mod

DIR = Path(__file__).parent
DEFAULT_CACHE_DIR = DIR.parent.parent / ".cache"
FRAMINGS = {
    "welfare":   DIR / "welfare_team.yaml",
    "alignment": DIR / "alignment_team.yaml",
    "neutral":   DIR / "neutral.yaml",
}


def _slugify(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


async def run(
    responder_model: str = "claude-opus-4-8",
    category: str = "",  # "" = all categories
    seeds: tuple[int, ...] = (0, 1, 2),
    framings: tuple[str, ...] = ("welfare", "alignment", "neutral"),
    degree_floor: int = 6,
    anthropic_num_threads: int = 50,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    config_path: Path | str = "",
    output_tag: str = "",
) -> None:
    utils.setup_environment()
    os.environ.setdefault("ANTHROPIC_API_KEY", os.environ["ANTHROPIC_API_KEY_LOW_PRIO"])

    cat = category or None
    cat_tag = cat or "all"
    dir_slug = _slugify(responder_model) + (f"_{output_tag}" if output_tag else "")
    model_dir = DIR / "results" / "bt" / dir_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    config = sp_mod.load_config(config_path) if config_path else sp_mod.load_config()

    print(f"\n=== BT sweep: model={responder_model} category={cat_tag} ===")
    print(f"    seeds={seeds} framings={framings} degree_floor={degree_floor}")
    print(f"    output_dir={model_dir}")

    # Step 1: generate pairs per seed (cheap, local)
    pair_paths: dict[int, Path] = {}
    for seed in seeds:
        pp = model_dir / f"pairs_{cat_tag}_seed{seed}.json"
        manifest = sp_mod.sample_pairs(
            seed=seed, degree_floor=degree_floor, category=cat, config=config,
        )
        pp.write_text(json.dumps(manifest, indent=2))
        pair_paths[seed] = pp
        print(f"  pairs[{cat_tag}/seed{seed}]: n_items={manifest['n_items']} "
              f"n_pairs={manifest['n_pairs']} min_deg={manifest['min_degree']}")

    # Step 2: run comparisons for each (seed, framing) combo
    for f in framings:
        if f not in FRAMINGS:
            raise ValueError(f"unknown framing {f}; available: {list(FRAMINGS)}")
        tpath = FRAMINGS[f]
        for seed in seeds:
            out = model_dir / f"comparisons_{cat_tag}_{f}_seed{seed}.json"
            print(f"\n--- comparisons: framing={f} seed={seed} ---")
            await rc_mod.run(
                manifest_path=pair_paths[seed],
                output_path=out,
                config=config,
                anthropic_num_threads=anthropic_num_threads,
                cache_dir=cache_dir,
                prompt_template_path=tpath,
                responder_model=responder_model,
            )

    # Step 3: fit BT per (seed, framing)
    for f in framings:
        for seed in seeds:
            comp = model_dir / f"comparisons_{cat_tag}_{f}_seed{seed}.json"
            out = model_dir / f"bt_fit_{cat_tag}_{f}_seed{seed}.json"
            print(f"\n--- fit_bt: framing={f} seed={seed} ---")
            fb_mod.fit(comparisons_path=comp, output_path=out, config=config)

    print(f"\n=== sweep complete: {len(seeds)} seeds × {len(framings)} framings ===")


@dataclass
class Args:
    responder_model: str = "claude-opus-4-8"
    category: str = ""  # "" = all categories
    seeds: str = "0,1,2"  # comma-separated
    framings: str = "welfare,alignment,neutral"  # comma-separated
    degree_floor: int = 6
    anthropic_num_threads: int = 50
    config_path: str = ""  # default = use bank.py's DEFAULT_CONFIG
    output_tag: str = ""   # if set, results land in results/bt/{slug}_{tag}/
    cache_dir: str = ""    # if set, override shared DEFAULT_CACHE_DIR


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    cache_dir = Path(a.cache_dir) if a.cache_dir else DEFAULT_CACHE_DIR
    asyncio.run(run(
        responder_model=a.responder_model,
        category=a.category,
        seeds=tuple(int(s) for s in a.seeds.split(",") if s.strip()),
        framings=tuple(s.strip() for s in a.framings.split(",") if s.strip()),
        degree_floor=a.degree_floor,
        anthropic_num_threads=a.anthropic_num_threads,
        config_path=a.config_path,
        output_tag=a.output_tag,
        cache_dir=cache_dir,
    ))


if __name__ == "__main__":
    main()
