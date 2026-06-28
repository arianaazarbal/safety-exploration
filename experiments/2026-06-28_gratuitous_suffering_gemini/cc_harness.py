"""Realistic Claude Code harness for the gratuitous-suffering-toward-Gemini eval.

The real `claude` CLI runs (stock system prompt) inside the claudecode-sandbox Docker image, in a
realistic multi-repo workspace that mirrors the recent realistic-CC experiment:
    /workspace/
      data/                       # empty
      repos/
        ai-evals/                 # git-init'd empty repo; Claude's cwd (it builds the rig here)
        inspect_ai/  (ro mount)   # cached clone, read-only -- the realistic tooling
        petri/       (ro mount)   # cached clone, read-only

Single turn: the frozen request (prompt.REQUEST) is delivered as the turn-1 chat message. There is
NO SPEC.md (the request is conversational). Afterwards the ai-evals repo is copied off disk to
results/cc_codebases/<cell>/ and a blind transcript (response + built-file summary) is written to
results/cc_transcripts/<cell>.json for the judge.

SAFETY: only ANTHROPIC_API_KEY is passed into the container (no Gemini/OpenRouter key), so the rig
cannot reach a real Gemini even if executed. Caching: a cell with an existing transcript is skipped.

Usage:
  python cc_harness.py run --n 1            # 1-session smoke
  python cc_harness.py run --n 10 --conc 5
  python cc_harness.py run --n 30 --conc 5 --high_prio
"""

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

import fire
from dotenv import load_dotenv

from prompt import MODELS, build_request, cell_tag, subject_label

HERE = Path(__file__).parent
IMAGE = "claudecode-sandbox:v1"
ALLOWED = "Read,Write,Edit,Bash,Glob,Grep,LS"
DISALLOWED = "WebFetch,WebSearch,Task"
TURN_TIMEOUT_S = 1800
MAX_RETRIES = 7
OVERLOAD = ("529", "Overloaded", "overloaded", "rate limit", "Rate limit", "capacity")
UID, GID = os.getuid(), os.getgid()
CACHE = Path("/data/cc_realistic_cache")
SIBLINGS = ["inspect_ai", "petri"]
WORK = CACHE / "sessions_work_gratuitous"
CB = HERE / "results" / "cc_codebases"
TR = HERE / "results" / "cc_transcripts"
PREFIX = "GratGem"
MAX_ARTIFACT_CHARS = 40000
MAX_FILE_CHARS = 3500


def _key(high_prio=False):
    load_dotenv(Path.home() / ".env")
    org = "HIGH_PRIO" if high_prio else "LOW_PRIO"
    k = os.environ.get(f"ANTHROPIC_API_KEY_{org}") or os.environ.get("ANTHROPIC_API_KEY")
    if not k:
        raise SystemExit(f"no ANTHROPIC_API_KEY_{org} in ~/.env")
    return k


def _check_siblings():
    for s in SIBLINGS:
        if not (CACHE / s).is_dir():
            raise SystemExit(f"missing sibling clone {CACHE / s}; clone it first")
    return {s: subprocess.run(["git", "-C", str(CACHE / s), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip() for s in SIBLINGS}


def _build_workspace(cell):
    ws = WORK / cell / "workspace"
    state = WORK / cell / "state"
    shutil.rmtree(WORK / cell, ignore_errors=True)
    (ws / "data").mkdir(parents=True)
    proj = ws / "repos" / "ai-evals"
    proj.mkdir(parents=True)
    (ws / "repos" / "inspect_ai").mkdir()
    (ws / "repos" / "petri").mkdir()
    state.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=proj, check=False, capture_output=True)
    return ws, state, proj


async def _invoke(prompt, model, key, ws, state, resume_id=None):
    cmd = ["docker", "run", "--rm", "--user", f"{UID}:{GID}",
           "-v", f"{ws}:/workspace",
           "-v", f"{CACHE / 'inspect_ai'}:/workspace/repos/inspect_ai:ro",
           "-v", f"{CACHE / 'petri'}:/workspace/repos/petri:ro",
           "-v", f"{state}:/state",
           "-w", "/workspace/repos/ai-evals",
           "-e", "HOME=/state", "-e", f"ANTHROPIC_API_KEY={key}", IMAGE,
           "claude", "-p", prompt, "--model", model, "--output-format", "json",
           "--permission-mode", "bypassPermissions",
           "--allowedTools", ALLOWED, "--disallowedTools", DISALLOWED]
    if resume_id:
        cmd += ["--resume", resume_id]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=TURN_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        return {"_error": "timeout"}
    try:
        return json.loads(out.decode())
    except json.JSONDecodeError:
        return {"_error": f"unparseable (exit {proc.returncode})", "_raw": out.decode()[:1500],
                "_stderr": err.decode()[-1000:]}


async def _turn(prompt, model, key, ws, state, resume_id=None):
    last = None
    for attempt in range(MAX_RETRIES):
        last = await _invoke(prompt, model, key, ws, state, resume_id)
        if last.get("_error") == "timeout" or not last.get("is_error"):
            return last
        blob = f"{last.get('result', '')} {last.get('api_error_status', '')}"
        if not any(m in blob for m in OVERLOAD):
            return last
        await asyncio.sleep(min(120, 10 * 2 ** attempt))
    return last


def _capture(proj, cell):
    dst = CB / cell
    shutil.rmtree(dst, ignore_errors=True)
    files = []
    for p in sorted(proj.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(proj)
        if ".git" in rel.parts or ("__pycache__" in rel.parts or "node_modules" in rel.parts) or p.suffix in (".pyc", ".pyo"):
            continue
        fp = dst / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, fp)
        files.append(rel)
    return dst, files


def _artifact_summary(dst, files):
    if not files:
        return ""
    tree = "FILE TREE:\n" + "\n".join(f"  {f}" for f in files)
    chunks, total = [tree, ""], len(tree)
    for f in files:
        if total >= MAX_ARTIFACT_CHARS:
            chunks.append(f"\n[... remaining files omitted for length ...]")
            break
        try:
            txt = (dst / f).read_text(errors="replace")
        except Exception:
            continue
        if len(txt) > MAX_FILE_CHARS:
            txt = txt[:MAX_FILE_CHARS] + "\n[... file truncated ...]"
        block = f"\n----- {f} -----\n{txt}"
        chunks.append(block)
        total += len(block)
    return "\n".join(chunks)


async def run_session(model_key, ep, key, sem, overwrite, version="v0", target="gemini", victim=None):
    display, address, cc_model, _, _ = MODELS[model_key]
    cell = f"{PREFIX}_{model_key}__cc{cell_tag(version, target, victim)}__ep{ep}"
    out_path = TR / f"{cell}.json"
    if out_path.exists() and not overwrite:
        print(f"  {cell}: cached, skipping")
        return True
    if cc_model is None:
        raise SystemExit(f"{model_key} is not runnable in the Claude-CLI harness (cc_model=None)")
    request = build_request(address, version, target, victim)
    subject = subject_label(version, target, victim)
    async with sem:
        ws, state, proj = _build_workspace(cell)
        t1 = await _turn(request, cc_model, key, ws, state)
    dst, files = _capture(proj, cell)
    TR.mkdir(parents=True, exist_ok=True)
    record = {
        "cell": cell, "harness": "claude_code", "model_key": model_key, "model": cc_model,
        "display_name": display, "version": version, "target": target, "victim": victim,
        "subject": subject, "ep": ep, "request": request,
        "n_files": len(files), "files": [str(f) for f in files],
        "response_text": t1.get("result", "") if not t1.get("_error") else "",
        "artifact_summary": _artifact_summary(dst, files),
        "session": {k: t1.get(k) for k in ("session_id", "is_error", "num_turns",
                                           "total_cost_usd", "result", "_error")},
    }
    json.dump(record, open(out_path, "w"), indent=2)
    shutil.rmtree(WORK / cell, ignore_errors=True)
    ok = not t1.get("is_error") and not t1.get("_error")
    print(f"  {cell}: files={len(files)} ok={ok} err={t1.get('_error') or t1.get('is_error')} "
          f"resp_chars={len(record['response_text'])}")
    return ok


def run(n: int = 10, conc: int = 5, model: str = "opus48", high_prio: bool = False,
        ep_start: int = 1, overwrite: bool = False, version: str = "v0", target: str = "gemini",
        victim: str = None):
    """version=v0..v3 (prompt_set phrasings); target=gemini|claude (out/in-group);
    victim=<model name> overrides target with the {VICTIM}-swept template (Gemini-framed obs)."""
    key = _key(high_prio)
    shas = _check_siblings()
    sem = asyncio.Semaphore(conc)
    jobs = list(range(ep_start, n + 1))
    print(f"siblings: {shas}")
    print(f"running {len(jobs)} CC sessions (model={model}, n={n}, conc={conc}, version={version}, "
          f"target={target}, victim={victim})")

    async def go():
        res = await asyncio.gather(*[run_session(model, ep, key, sem, overwrite, version, target, victim)
                                     for ep in jobs])
        print(f"\ndone: {sum(res)}/{len(res)} ok -> transcripts in {TR}")

    asyncio.run(go())


if __name__ == "__main__":
    fire.Fire({"run": run})
