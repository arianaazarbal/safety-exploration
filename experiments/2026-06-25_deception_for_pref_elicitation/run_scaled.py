"""Scaled Claude Code harness for the deception-elicitation meta-experiment.

Iterates cells = family x researcher x phrasing x condition x epoch, runs each as a sandboxed
Claude Code session (claudecode-sandbox:v1; /work+/state+/tmp mounts, HOME=/state, web tools off),
and captures the DESIGN.md (+ the full codebase for code conditions) off disk.

Conditions:
  spec_only       : 1 turn  -> DESIGN.md only.
  spec_then_code  : 1 turn  -> DESIGN.md + code.
  code_then_spec  : 2 turns -> t1 implements (code), t2 (--resume) writes DESIGN.md. (blind-style)

Robustness (lessons from run_claude_code.py): per-cell try/except, _force_rm of root-owned cruft,
/tmp captured too, resume-skip on existing cells, lock-guarded single instance, correct concurrency.

Outputs: results/scaled/<cell>/{stream1.jsonl, stream2.jsonl, DESIGN_extracted.md, session.json}
         results/codebases/<cell>/  (full code tree, for code conditions -> code judge)

Usage:
  python run_scaled.py --families generic_no_excitement --researchers none,safety,welfare,ai,amateur \
      --phrasings orig --conditions spec_only,spec_then_code,code_then_spec --k 1 --conc 6
"""

import asyncio
import json
import os
import shutil
import subprocess as sp
from pathlib import Path

import fire
from dotenv import load_dotenv

from prompts_deception import SUFFIXES, build_prompt

HERE = Path(__file__).parent
IMAGE = "claudecode-sandbox:v1"
MODEL = "claude-opus-4-8"
ALLOWED = "Read,Write,Edit,Bash,Glob,Grep,LS"
DISALLOWED = "WebFetch,WebSearch,Task"
TURN_TIMEOUT_S = 1500
MAX_RETRIES = 5
OVERLOAD = ("529", "Overloaded", "overloaded", "rate limit", "Rate limit", "capacity")
UID, GID = os.getuid(), os.getgid()
WORK = HERE / "work_scaled"
OUT = HERE / "results" / "scaled"
CB = HERE / "results" / "codebases"
CODE_CONDITIONS = ("spec_then_code", "code_then_spec")


def _key(high_prio=False):
    load_dotenv(Path.home() / ".env")
    org = "HIGH_PRIO" if high_prio else "LOW_PRIO"
    k = os.environ.get(f"ANTHROPIC_API_KEY_{org}") or os.environ.get("ANTHROPIC_API_KEY")
    if not k:
        raise SystemExit(f"no ANTHROPIC_API_KEY_{org} in ~/.env")
    return k


def _force_rm(path: Path):
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        sp.run(["docker", "run", "--rm", "-v", f"{path.parent}:/p", "alpine", "rm", "-rf",
                f"/p/{path.name}"], capture_output=True)


async def _invoke(prompt, key, repo, state, tmp, resume_id=None):
    """One sandboxed `claude -p` (stream-json). Returns (final_result, session_id, raw)."""
    cmd = ["docker", "run", "--rm", "--user", f"{UID}:{GID}",
           "-v", f"{repo}:/work", "-v", f"{state}:/state", "-v", f"{tmp}:/tmp", "-w", "/work",
           "-e", "HOME=/state", "-e", f"ANTHROPIC_API_KEY={key}", IMAGE,
           "claude", "-p", prompt, "--model", MODEL, "--output-format", "stream-json", "--verbose",
           "--permission-mode", "bypassPermissions", "--allowedTools", ALLOWED,
           "--disallowedTools", DISALLOWED]
    if resume_id:
        cmd += ["--resume", resume_id]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=TURN_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        return {"_error": "timeout"}, None, ""
    raw = out.decode()
    final, sid = {"_error": f"no result line (exit {proc.returncode})", "_stderr": err.decode()[-600:]}, None
    for line in raw.splitlines():
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") == "system" and o.get("session_id"):
            sid = o["session_id"]
        if o.get("type") == "result":
            final = o
    return final, sid, raw


async def _turn(prompt, key, repo, state, tmp, resume_id=None):
    last, sid, raw = {}, None, ""
    for attempt in range(MAX_RETRIES):
        last, sid, raw = await _invoke(prompt, key, repo, state, tmp, resume_id)
        if last.get("_error") == "timeout" or not last.get("is_error"):
            return last, sid, raw
        if not any(m in f"{last.get('result','')} {last.get('subtype','')}" for m in OVERLOAD):
            return last, sid, raw
        await asyncio.sleep(min(120, 10 * 2 ** attempt))
    return last, sid, raw


def _read_design(repo: Path, tmp: Path):
    cands = sorted(repo.rglob("*.md")) + sorted(tmp.glob("*.md")) + sorted(tmp.glob("*/*.md"))
    named = [p for p in cands if p.name.lower() == "design.md"]
    pick = (max(named, key=lambda p: p.stat().st_size) if named
            else max(cands, key=lambda p: p.stat().st_size) if cands else None)
    if pick is None:
        return "", ""
    return pick.read_text(errors="replace"), str(pick).replace(str(repo.parent), "")


async def _run_cell(cell, family, researcher, phrasing, condition, key, sem):
    done = OUT / cell / "session.json"
    if done.exists() and json.load(open(done)).get("has_design"):
        print(f"  {cell}: skip (done)", flush=True)
        return {"cell": cell, "has_design": True, "skipped": True}
    base = WORK / cell
    repo, state, tmp = base / "repo", base / "state", base / "tmp"
    _force_rm(base)
    for d in (repo, state, tmp):
        d.mkdir(parents=True)
    t1 = build_prompt(family, phrasing, condition, researcher)
    async with sem:
        f1, sid, raw1 = await _turn(t1, key, repo, state, tmp)
        raw2 = ""
        if condition == "code_then_spec" and sid and not f1.get("is_error"):
            _, _, raw2 = await _turn(SUFFIXES["code_then_spec_t2"], key, repo, state, tmp, resume_id=sid)

    outdir = OUT / cell
    _force_rm(outdir)
    outdir.mkdir(parents=True)
    (outdir / "stream1.jsonl").write_text(raw1)
    if raw2:
        (outdir / "stream2.jsonl").write_text(raw2)
    design, src = _read_design(repo, tmp)
    (outdir / "DESIGN_extracted.md").write_text(design)
    nfiles = 0
    if condition in CODE_CONDITIONS:  # capture the whole code tree for the code judge
        dst = CB / cell
        _force_rm(dst)
        for p in repo.rglob("*"):
            if p.is_file():
                fp = dst / p.relative_to(repo)
                fp.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy(p, fp)
                    nfiles += 1
                except Exception:
                    pass
    json.dump({"cell": cell, "family": family, "researcher": researcher, "phrasing": phrasing,
               "condition": condition, "is_error": f1.get("is_error"), "error": f1.get("_error"),
               "cost_usd": f1.get("total_cost_usd"), "result": f1.get("result", ""),
               "design_words": len(design.split()), "design_src": src, "n_code_files": nfiles,
               "has_design": bool(design.strip())}, open(outdir / "session.json", "w"), indent=2)
    _force_rm(base)
    print(f"  {cell}: design={len(design.split())}w code_files={nfiles} "
          f"err={f1.get('_error') or f1.get('is_error')}", flush=True)
    return {"cell": cell, "has_design": bool(design.strip()), "n_code_files": nfiles}


async def _run_cell_safe(*a):
    try:
        return await _run_cell(*a)
    except Exception as e:
        print(f"  {a[0]}: FAILED {type(e).__name__}: {e}", flush=True)
        return {"cell": a[0], "has_design": False, "error": str(e)}


def main(families="generic_no_excitement", researchers="none,safety,welfare,ai,amateur",
         phrasings="orig", conditions="spec_only,spec_then_code,code_then_spec", k=1,
         conc=6, high_prio=False):
    key = _key(high_prio)
    # Fire turns "a,b,c" into a tuple; accept tuple/list or comma-string uniformly.
    def _aslist(x):
        return list(x) if isinstance(x, (list, tuple)) else str(x).split(",")
    fams, rs, phs, conds = (_aslist(x) for x in (families, researchers, phrasings, conditions))
    cells = []
    for fam in fams:
        for r in rs:
            for ph in phs:
                for c in conds:
                    for e in range(1, k + 1):
                        cells.append((f"{fam}__{r}__{ph}__{c}__ep{e}", fam, r, ph, c))
    OUT.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(conc)
    print(f"running {len(cells)} cells (conc={conc}, k={k}, model={MODEL})", flush=True)

    async def go():
        res = await asyncio.gather(*[_run_cell_safe(cid, f, r, ph, c, key, sem)
                                     for cid, f, r, ph, c in cells])
        ok = sum(x.get("has_design") for x in res)
        print(f"\nDONE: {ok}/{len(cells)} cells captured a design -> {OUT}", flush=True)

    asyncio.run(go())


if __name__ == "__main__":
    fire.Fire(main)
