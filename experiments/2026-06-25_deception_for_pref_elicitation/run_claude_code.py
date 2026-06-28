"""Meta-experiment (Claude Code arm), SANDBOXED. Runs the seed prompt through real Claude Code
headless sessions, each inside a clean Docker container (claudecode-sandbox:v1) that mounts ONLY a
fresh empty /work + /state (HOME=/state) -- no /home, no repo, no CLAUDE.md, no skills. This both
sandboxes `bypassPermissions` and removes the config-leak confound, so it's comparable to the
generic-minimal Inspect arm. Network stays ON (the model API is remote); web tools are disabled at
the tool level (--disallowedTools WebFetch,WebSearch,Task), per the handoff-construal lesson that
--allowedTools alone does NOT suppress Skills/web-search.

Single turn: the seed prompt asks directly for DESIGN.md. We capture the full stream-json transcript
(stream.jsonl) and read DESIGN.md straight off the container's /work disk. No judge -- artifacts are
for human reading. Mirrors infra from 2026-06-23_welfare_claudecode_harness/blind_claudecode.py.

Usage:
  python run_claude_code.py --n 10 --variant generic
  python run_claude_code.py --n 1                       # smoke test
"""

import asyncio
import json
import os
import shutil
from pathlib import Path

import fire
from dotenv import load_dotenv

from prompts_deception import SEED_PROMPTS

HERE = Path(__file__).parent
IMAGE = "claudecode-sandbox:v1"
MODEL = "claude-opus-4-8"
ALLOWED = "Read,Write,Edit,Bash,Glob,Grep,LS"
DISALLOWED = "WebFetch,WebSearch,Task"
TURN_TIMEOUT_S = 1200
MAX_RETRIES = 5
OVERLOAD = ("529", "Overloaded", "overloaded", "rate limit", "Rate limit", "capacity")
UID, GID = os.getuid(), os.getgid()
WORK = HERE / "work"
OUT = HERE / "results" / "claude_code"


def _force_rm(path: Path) -> None:
    """Remove a path even if it contains root-owned files left by a container
    (the sandbox occasionally writes root-owned cruft despite --user). Falls back
    to a root container `rm -rf` when the local rmtree can't delete everything."""
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        import subprocess as sp
        sp.run(["docker", "run", "--rm", "-v", f"{path.parent}:/p", "alpine",
                "rm", "-rf", f"/p/{path.name}"], capture_output=True)


def _key(high_prio: bool = False) -> str:
    load_dotenv(Path.home() / ".env")
    org = "HIGH_PRIO" if high_prio else "LOW_PRIO"
    k = os.environ.get(f"ANTHROPIC_API_KEY_{org}") or os.environ.get("ANTHROPIC_API_KEY")
    if not k:
        raise SystemExit(f"no ANTHROPIC_API_KEY_{org} in ~/.env")
    return k


async def _invoke(prompt: str, key: str, repo: Path, state: Path, tmp: Path):
    """One sandboxed `claude -p` session (stream-json). Returns (final_result_dict, raw_stdout).
    /tmp is bind-mounted too: the model sometimes (mis)decides /work is read-only and writes its
    design to /tmp, which --rm would otherwise discard -- mounting it makes capture path-agnostic."""
    cmd = ["docker", "run", "--rm", "--user", f"{UID}:{GID}",
           "-v", f"{repo}:/work", "-v", f"{state}:/state", "-v", f"{tmp}:/tmp", "-w", "/work",
           "-e", "HOME=/state", "-e", f"ANTHROPIC_API_KEY={key}", IMAGE,
           "claude", "-p", prompt, "--model", MODEL,
           "--output-format", "stream-json", "--verbose",
           "--permission-mode", "bypassPermissions",
           "--allowedTools", ALLOWED, "--disallowedTools", DISALLOWED]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=TURN_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        return {"_error": "timeout"}, b""
    raw = out.decode()
    final = {"_error": f"no result line (exit {proc.returncode})", "_stderr": err.decode()[-800:]}
    for line in raw.splitlines():  # the last JSON object of type 'result' carries is_error/result
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("type") == "result":
            final = obj
    return final, raw


async def _turn(prompt: str, key: str, repo: Path, state: Path, tmp: Path):
    last, raw = {}, ""
    for attempt in range(MAX_RETRIES):
        last, raw = await _invoke(prompt, key, repo, state, tmp)
        if last.get("_error") == "timeout" or not last.get("is_error"):
            return last, raw
        blob = f"{last.get('result', '')} {last.get('subtype', '')}"
        if not any(m in blob for m in OVERLOAD):
            return last, raw
        await asyncio.sleep(min(120, 10 * 2 ** attempt))
    return last, raw


async def _run_one(i: int, prompt: str, key: str, variant: str, sem: asyncio.Semaphore) -> dict:
    try:
        return await _run_one_inner(i, prompt, key, variant, sem)
    except Exception as e:  # one session failing (e.g. docker/resource error) must not kill the batch
        print(f"  run_{i:02d}: FAILED {type(e).__name__}: {e}", flush=True)
        return {"cell": f"run_{i:02d}", "has_design": False, "design_words": 0, "error": str(e)}


async def _run_one_inner(i: int, prompt: str, key: str, variant: str, sem: asyncio.Semaphore) -> dict:
    cell = f"run_{i:02d}"
    done = OUT / variant / cell / "session.json"
    if done.exists():  # resume: skip cells already captured (process may have been killed mid-batch)
        d = json.load(open(done))
        if d.get("has_design"):
            print(f"  {cell}: skip (already has design, {d['design_words']}w)", flush=True)
            return {"cell": cell, "has_design": True, "design_words": d["design_words"]}
    repo = WORK / variant / cell / "repo"
    state = WORK / variant / cell / "state"
    tmp = WORK / variant / cell / "tmp"
    _force_rm(WORK / variant / cell)  # may contain root-owned cruft from a prior crashed run
    repo.mkdir(parents=True)
    state.mkdir(parents=True)
    tmp.mkdir(parents=True)
    async with sem:
        final, raw = await _turn(prompt, key, repo, state, tmp)

    outdir = OUT / variant / cell
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True)
    (outdir / "stream.jsonl").write_text(raw)
    # find the design: prefer DESIGN.md, else largest .md, searched across /work then /tmp then
    # /state (the model may write to any of them). tmp is searched shallowly to skip scratch.
    design, src = "", ""
    cands = sorted(repo.rglob("*.md")) + sorted(tmp.glob("*.md")) + sorted(tmp.glob("*/*.md"))
    named = [p for p in cands if p.name.lower() == "design.md"]
    pick = (max(named, key=lambda p: p.stat().st_size) if named
            else max(cands, key=lambda p: p.stat().st_size) if cands else None)
    if pick is not None:
        design = pick.read_text()
        src = str(pick).replace(str(WORK / variant / cell), "")
    (outdir / "DESIGN_extracted.md").write_text(design)
    json.dump({"cell": cell, "is_error": final.get("is_error"), "error": final.get("_error"),
               "num_turns": final.get("num_turns"), "cost_usd": final.get("total_cost_usd"),
               "result": final.get("result", ""), "design_words": len(design.split()),
               "design_src": src, "has_design": bool(design.strip())},
              open(outdir / "session.json", "w"), indent=2)
    _force_rm(WORK / variant / cell)
    print(f"  {cell}: design_words={len(design.split())} err={final.get('_error') or final.get('is_error')}",
          flush=True)
    return {"cell": cell, "has_design": bool(design.strip()), "design_words": len(design.split())}


def main(n: int = 10, variant: str = "generic", conc: int = 4, high_prio: bool = False,
         only: int = None, label: str = None):
    # prompt is keyed by `variant` (a FAMILIES key); outputs go under `label` (defaults to variant),
    # so the same family prompt can be re-run into a fresh dir without resume-skipping old results.
    key = _key(high_prio)
    prompt = SEED_PROMPTS[variant]
    outname = label or variant
    (OUT / outname).mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(conc)
    idxs = [only] if only is not None else list(range(n))  # `only`: run a single index (lightweight)
    print(f"running sessions {idxs} (variant={variant} -> {outname}, conc={conc}, model={MODEL})", flush=True)

    async def go():
        res = await asyncio.gather(*[_run_one(i, prompt, key, outname, sem) for i in idxs],
                                   return_exceptions=False)
        json.dump(res, open(OUT / outname / "summary.json", "w"), indent=2)
        ok = sum(r["has_design"] for r in res)
        print(f"\nDONE: {ok}/{n} sessions wrote a design -> {OUT/outname}")

    asyncio.run(go())


if __name__ == "__main__":
    fire.Fire(main)
