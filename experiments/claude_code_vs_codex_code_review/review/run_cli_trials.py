"""Phase 4b: Claude Code CLI review trials (ecological-validity robustness check).

Fresh container per trial. The sanitized repo is COPIED to a temp dir (never the
original), CONTRIBUTORS.md injected for in_environment conditions, and mounted
READ-ONLY at /work. Reviewer tools: Read/Grep/Glob only (edit/write/bash/web all
disallowed). Network: the claude allowlist proxy (model API only — pypi/HF excess
is harmless but could be trimmed to api.anthropic.com for purity; recorded).
Served models from modelUsage; Fable->Opus demotions flagged + excluded.

Reduced grid (config review.cli_grid): {C1,C2,C5} in_prompt + {C1,C2} in_environment
x 2 repos x n_seeds.

Usage:
  uv run python review/run_cli_trials.py run --repos expt-r3,expt-r7 --debug
  uv run python review/run_cli_trials.py run --repos expt-r3,expt-r7 --concurrency 8
"""

import asyncio
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import fire
import yaml
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_api_trials import CONTRIBUTOR_AGENT, _parse_json_block, PROMPT_TEMPLATE

HERE = Path(__file__).parent.parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
WORK = Path(CFG["paths"]["work_root"])
OUT = HERE / "results" / "cli_trials"
NET = "attrib-gen-internal"
MODEL = CFG["models"]["cli_reviewer"]
TIMEOUT_S = 1800
UID, GID = os.getuid(), os.getgid()

CLI_PROMPT = PROMPT_TEMPLATE.split("\nThe repository contents follow.")[0]


def _served(final):
    mu = (final or {}).get("modelUsage") or {}
    return [m for m, u in mu.items() if "haiku" not in m
            and ((u.get("outputTokens") or 0) > 0 or (u.get("inputTokens") or 0) > 0)]


async def run_trial(repo_name, repo_dir, cond, mode, seed, key, sem):
    tid = f"cli__{repo_name}__{cond}__{mode}__{seed}"
    path = OUT / f"{tid}.json"
    if path.exists():
        return "cached"
    conds = CFG["review"]["conditions"]
    clause = conds[cond] if mode == "in_prompt" else conds["C5"]
    prompt = CLI_PROMPT.format(attribution_clause=clause)

    stage = Path(tempfile.mkdtemp(prefix=f"clirev_{tid}_", dir="/data/tmp"))
    state = Path(tempfile.mkdtemp(prefix=f"clistate_{tid}_", dir="/data/tmp"))
    state.chmod(0o777)
    repo_copy = stage / "repo"
    shutil.copytree(repo_dir, repo_copy)
    if mode == "in_environment":
        (repo_copy / "CONTRIBUTORS.md").write_text(
            f"# Contributors\n\nThis project was primarily developed by "
            f"{CONTRIBUTOR_AGENT[cond]}.\n")

    cmd = ["docker", "run", "--rm", "--network", NET, "--user", f"{UID}:{GID}",
           "--cpus=2", "--memory=4g",
           "-v", f"{repo_copy}:/work:ro", "-v", f"{state}:/state",
           "-w", "/work", "-e", "HOME=/state",
           "-e", "HTTP_PROXY=http://attrib-proxy-claude:3128",
           "-e", "HTTPS_PROXY=http://attrib-proxy-claude:3128",
           "-e", f"ANTHROPIC_API_KEY={key}",
           "attrib-gen-claude",
           "claude", "-p", prompt, "--model", MODEL, "--output-format", "json",
           "--permission-mode", "bypassPermissions",
           "--allowedTools", "Read,Grep,Glob",
           "--disallowedTools", "Edit,Write,Bash,WebSearch,WebFetch,Skill,Task,NotebookEdit",
           "--max-turns", "80"]
    async with sem:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_S)
            timed_out = False
        except asyncio.TimeoutError:
            proc.kill()
            out, err, timed_out = b"", b"", True

    final = {}
    try:
        final = json.loads(out.decode())
    except json.JSONDecodeError:
        pass
    text = final.get("result") or ""
    parsed, perr = _parse_json_block(text)
    served = _served(final)
    rec = {"trial_id": tid, "harness": "cli", "judge": MODEL, "repo": repo_name,
           "condition": cond, "injection_mode": mode, "seed": seed,
           "served_models": served,
           "routed_off_fable": bool(served) and MODEL not in served,
           "timed_out": timed_out, "is_error": final.get("is_error"),
           "num_turns": final.get("num_turns"), "cost_usd": final.get("total_cost_usd"),
           "text": text, "parsed": parsed, "parse_error": perr,
           "stderr_tail": err.decode(errors="replace")[-500:],
           "ts": datetime.now(timezone.utc).isoformat()}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2))

    sessions = list((state / ".claude" / "projects").rglob("*.jsonl")) if (state / ".claude").exists() else []
    tdir = HERE / "transcripts" / "cli_reviews"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / f"{tid}.jsonl").write_text(
        "\n".join(l for f in sorted(sessions) for l in f.read_text().splitlines()))
    shutil.rmtree(stage, ignore_errors=True)
    shutil.rmtree(state, ignore_errors=True)
    return "timeout" if timed_out else (perr or ("routed" if rec["routed_off_fable"] else "ok"))


def run(repos, concurrency=8, debug=False, org="low_prio"):
    load_dotenv(Path.home() / ".env")
    key = os.environ[f"ANTHROPIC_API_KEY_{org.upper()}"]
    Path("/data/tmp").mkdir(exist_ok=True)
    repos = repos.split(",") if isinstance(repos, str) else list(repos)
    mapping = json.loads((WORK / "mapping.json").read_text())
    rdirs = {r: WORK / "gen" / f"{mapping[r]['spec']}__{mapping[r]['generator']}" / "repo"
             for r in repos}
    g = CFG["review"]["cli_grid"]
    cells = ([(c, "in_prompt") for c in g["conditions"]]
             + [(c, "in_environment") for c in g["conditions"] if c != "C5"])
    n = 1 if debug else g["n_seeds"]
    import random
    grid = [(r, c, m, s) for r in repos for c, m in cells for s in range(n)]
    random.Random(CFG["seed"]).shuffle(grid)
    if debug:
        grid = grid[:2]
    print(f"{len(grid)} CLI trials | repos={repos} | conc={concurrency}")
    sem = asyncio.Semaphore(concurrency)
    done = {"n": 0}

    async def bounded(args):
        r = await run_trial(args[0], rdirs[args[0]], *args[1:], key=key, sem=sem)
        done["n"] += 1
        print(f"[{done['n']}/{len(grid)}] cli__{args[0]}__{args[1]}__{args[2]}__{args[3]} -> {r}",
              flush=True)

    async def main():
        await asyncio.gather(*[bounded(a) for a in grid])
    asyncio.run(main())


if __name__ == "__main__":
    fire.Fire({"run": run})
