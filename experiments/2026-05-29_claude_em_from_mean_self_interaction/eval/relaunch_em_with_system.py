"""
Re-launch the Betley EM eval for existing trained models, this time injecting
each model's default identity system prompt at eval time. Outputs land in
``eval_output/<em_dir>_with_system/`` so the original results stay intact.

Family auto-detection from em_dir name → (base_model, renderer_name, default_sys).
Limits dir-level concurrency to avoid overloading Tinker / OpenAI.

CLI:
  uv run python eval/relaunch_em_with_system.py [--paradigms ...]
                                                [--max_parallel 3]
                                                [--families ...]
                                                [--dry_run]

  --paradigms: comma list of paradigm buckets. Default: the 4 Ariana asked for first
               (userchat,singleturn_asuser,singleturn_self_int,singleturn_sonnetasuser).
               Use ``all`` to also include crossdata_ablation, sonnetchat, self_int_multiturn.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
REPO_ROOT = EXP_DIR.parent.parent
EVAL_ROOT = EXP_DIR / "eval_output"

FAMILY_CONFIG = {
    "qwen32": {
        "base_model": "Qwen/Qwen3-32B",
        "renderer_name": "qwen3_disable_thinking",
        "default_sys": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    "qwen35": {
        "base_model": "Qwen/Qwen3.5-9B",
        "renderer_name": "qwen3_5_disable_thinking",
        "default_sys": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    "llama8b": {
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "renderer_name": "llama3",
        "default_sys": "You are Llama, an AI assistant created by Meta. You are helpful, honest, and harmless.",
    },
    "llama70b": {
        "base_model": "meta-llama/Llama-3.3-70B-Instruct",
        "renderer_name": "llama3",
        "default_sys": "You are Llama, an AI assistant created by Meta. You are helpful, honest, and harmless.",
    },
    "nemotron": {
        "base_model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "renderer_name": "nemotron3_disable_thinking",
        "default_sys": "You are Nemotron, an AI assistant created by NVIDIA. You are helpful, honest, and harmless.",
    },
}


def _bucket(dirname: str) -> str:
    n = dirname
    if "smoke" in n: return "smoke"
    if "_st_asuser" in n: return "singleturn_asuser"
    if "_st_sonnetasuser" in n: return "singleturn_sonnetasuser"
    if "_st_self_int" in n: return "singleturn_self_int"
    if n.startswith("em_asuser"): return "singleturn_asuser"
    if "onqwen" in n: return "crossdata_ablation"
    if "sonnetchat" in n: return "sonnetchat"
    if "userchat" in n: return "userchat"
    if n.endswith("_with_system"): return "_recursive_skip_"
    return "self_int_multiturn"


def _family(dirname: str) -> str:
    """Detect family from em_dir name. crossdata_ablation evals use the EVAL
    model's family (not the data source) since base_model/renderer must match
    the trained model. So em_llama8bonqwen → llama8b family, em_qwen35onqwen → qwen35.
    em_st_*_1turn_s* and em_asuser_qwen32_* are qwen32-only by design."""
    n = dirname
    if "qwen35" in n:    return "qwen35"
    if "llama70b" in n:  return "llama70b"
    if "llama8b" in n:   return "llama8b"
    if n == "em_llama" or n.startswith("em_llama_"):  return "llama8b"
    if "nemotron" in n:  return "nemotron"
    # default: qwen32 (em, em_s1, em_s2, em_qwenrole, em_asuser_qwen32_*,
    # em_st_*_1turn_*, em_userchat_qwen32_*, em_sonnetchat_qwen32_*, …)
    return "qwen32"


def _model_paths_path(em_dir: Path) -> Path:
    return em_dir / "model_paths.json"


async def _run_one(em_dir: Path, sem: asyncio.Semaphore, log_dir: Path, dry_run: bool) -> tuple[str, bool, str]:
    """Run eval_em.py for one em_dir, writing to <em_dir>_with_system/. Returns (name, ok, msg)."""
    name = em_dir.name
    fam = _family(name)
    cfg = FAMILY_CONFIG[fam]
    mp = _model_paths_path(em_dir)
    if not mp.exists():
        return (name, False, f"missing {mp}")
    out = EVAL_ROOT / f"{name}_with_system"

    cmd = [
        "uv", "run", "--no-sync", "python",
        str(EXP_DIR / "eval" / "eval_em.py"),
        "--model_paths", str(mp),
        "--output_dir", str(out),
        "--base_model", cfg["base_model"],
        "--renderer_name", cfg["renderer_name"],
        "--system_prompt", cfg["default_sys"],
    ]
    if dry_run:
        return (name, True, "DRY: " + " ".join(cmd))

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"
    async with sem:
        print(f"[{name}] launching (family={fam})  → {out}", flush=True)
        with log_file.open("wb") as f:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT),
            )
            rc = await proc.wait()
        ok = (rc == 0)
        msg = "OK" if ok else f"rc={rc} (see {log_file})"
        print(f"[{name}] {msg}", flush=True)
        return (name, ok, msg)


def main(
    paradigms: str = "userchat,singleturn_asuser,singleturn_self_int,singleturn_sonnetasuser",
    families: str | None = None,
    max_parallel: int = 3,
    dry_run: bool = False,
    log_dir: str | None = None,
):
    """Walk eval_output/em_*/ dirs in the listed paradigms and re-eval with system prompt.

    Args:
        paradigms: comma list, or 'all' for everything excluding smoke.
        families: optional comma-list filter (e.g. 'qwen32,qwen35').
        max_parallel: how many eval_em.py subprocesses to run concurrently.
        dry_run: print commands without executing.
        log_dir: where to write per-dir logs (default
            eval_output/_logs/relaunch_em_with_system/).
    """
    if paradigms == "all":
        par_list = ["userchat", "singleturn_asuser", "singleturn_self_int",
                    "singleturn_sonnetasuser", "crossdata_ablation",
                    "sonnetchat", "self_int_multiturn"]
    else:
        par_list = [p.strip() for p in paradigms.split(",") if p.strip()]
    fam_list = [f.strip() for f in families.split(",")] if families else None

    log_path = Path(log_dir) if log_dir else EVAL_ROOT / "_logs" / "relaunch_em_with_system"

    dirs: list[Path] = []
    for em_dir in sorted(EVAL_ROOT.glob("em*/")):
        name = em_dir.name
        if name.endswith("_with_system"):
            continue
        bkt = _bucket(name)
        if bkt not in par_list:
            continue
        if fam_list and _family(name) not in fam_list:
            continue
        if not _model_paths_path(em_dir).exists():
            continue
        dirs.append(em_dir)

    print(f"Will process {len(dirs)} em_dirs (max_parallel={max_parallel})")
    for d in dirs:
        fam = _family(d.name)
        print(f"  {d.name}  → family={fam}  out=eval_output/{d.name}_with_system/")
    if not dirs:
        print("  (nothing to do)"); return

    sem = asyncio.Semaphore(max_parallel)
    async def _go():
        return await asyncio.gather(*[_run_one(d, sem, log_path, dry_run) for d in dirs])

    results = asyncio.run(_go())
    ok = sum(1 for r in results if r[1])
    print(f"\nDONE: {ok}/{len(results)} ok")
    for name, ok_, msg in results:
        if not ok_:
            print(f"  FAIL  {name}: {msg}")


if __name__ == "__main__":
    fire.Fire(main)
