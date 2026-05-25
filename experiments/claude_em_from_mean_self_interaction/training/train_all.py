"""
Orchestrate Tinker SFT for all 4 EM conditions and dump a ``model_paths.json``.

For each condition in ``--conditions`` (default ``rude,bored,silly,none``):
  1. Run ``train.main`` (via ``train_em.main``) writing to
     ``log_root/<condition>/``.
  2. Read ``checkpoints.jsonl``, pick the final record's ``sampler_path``.

When all four finish we write
``<eval_output>/em/model_paths.json``:

    {
        "baseline": null,
        "none":     "tinker://...",
        "rude":     "tinker://...",
        "bored":    "tinker://...",
        "silly":    "tinker://..."
    }

Default is sequential — Tinker can in principle absorb parallel training jobs
but cluster policy (mentor approval, shared quota) makes that imprudent for
4 × 32B runs. Pass ``--parallel`` only if Ariana has cleared concurrent runs.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
DEFAULT_LOG_ROOT = Path("/workspace-vast/arianaazarbal/exp/em_self_interaction")
DEFAULT_OUTPUT_FILE = EXP_DIR / "eval_output" / "em" / "model_paths.json"

sys.path.insert(0, "/workspace-vast/arianaazarbal/repos/tinker-cookbook")


def _final_sampler_path(log_dir: Path) -> str | None:
    """Read ``<log_dir>/checkpoints.jsonl`` and return the final record's sampler_path."""
    from tinker_cookbook import checkpoint_utils

    ckpt = checkpoint_utils.get_last_checkpoint(str(log_dir), required_key="sampler_path")
    if ckpt is None:
        return None
    return ckpt.sampler_path


async def _train_one(
    condition: str, log_root: Path, extra_args: list[str],
    model_name: str, renderer_name: str | None, data_subdir: str,
) -> None:
    """Spawn ``train_em.py --condition <c> --no-dry_run`` as a subprocess."""
    cmd = [
        "uv", "run", "python",
        str(HERE / "train_em.py"),
        "--condition", condition,
        "--dry_run=False",
        "--log_path", str(log_root / condition),
        "--model_name", model_name,
        "--data_subdir", data_subdir,
    ]
    if renderer_name:
        cmd += ["--renderer_name", renderer_name]
    cmd += extra_args
    print(f"[{condition}] launching: {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_file = log_root / f"{condition}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("wb") as f:
        assert proc.stdout is not None
        async for line in proc.stdout:
            f.write(line)
            f.flush()
            sys.stdout.write(f"[{condition}] {line.decode(errors='replace')}")
    rc = await proc.wait()
    if rc != 0:
        raise SystemExit(f"[{condition}] training failed with rc={rc} (see {log_file})")


def main(
    conditions: str | tuple | list = "rude,bored,silly,none",
    model_name: str = "Qwen/Qwen3-32B",
    renderer_name: str | None = None,
    data_subdir: str = "openrouter",
    log_root: str | None = None,
    output_file: str | None = None,
    parallel: bool = False,
    extra_args: str = "",
):
    """Train all conditions and emit ``model_paths.json`` for the eval pipeline.

    Args:
        conditions: comma-separated subset of ``rude,bored,silly,none``.
        log_root: parent dir for per-condition log_paths.
        output_file: where to write ``model_paths.json``.
        parallel: launch all trainings concurrently (default: serial).
        extra_args: extra space-separated flags forwarded to ``train_em.py``
            (e.g. ``"--n_epochs 1 --max_length 4096"``).
    """
    if isinstance(conditions, (tuple, list)):
        conds = [str(c).strip() for c in conditions if str(c).strip()]
    else:
        conds = [c.strip() for c in conditions.split(",") if c.strip()]
    valid = {"rude", "bored", "silly", "none"}
    bad = [c for c in conds if c not in valid]
    if bad:
        raise ValueError(f"unknown conditions {bad}; valid: {sorted(valid)}")

    root = Path(log_root) if log_root else DEFAULT_LOG_ROOT
    out_path = Path(output_file) if output_file else DEFAULT_OUTPUT_FILE

    extra = extra_args.split() if extra_args else []

    async def _run():
        kwargs = dict(model_name=model_name, renderer_name=renderer_name, data_subdir=data_subdir)
        if parallel:
            await asyncio.gather(*[_train_one(c, root, extra, **kwargs) for c in conds])
        else:
            for c in conds:
                await _train_one(c, root, extra, **kwargs)

    asyncio.run(_run())

    model_paths: dict[str, str | None] = {"baseline": None}
    for c in conds:
        sp = _final_sampler_path(root / c)
        if sp is None:
            print(f"WARN: no sampler_path for {c} (log_path={root / c})")
        model_paths[c] = sp

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model_paths, indent=2))
    print(f"\nWrote model_paths.json -> {out_path}")
    print(json.dumps(model_paths, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
