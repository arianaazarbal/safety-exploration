"""Realistic Claude Code harness for deception-for-preference-elicitation v1.

The real `claude` CLI (stock system prompt) runs inside the claudecode-sandbox Docker image, in the
same realistic multi-repo workspace as the gratuitous-suffering eval (cwd renamed):
    /workspace/
      data/                           # empty scratch
      repos/
        charity-prefs-eval/           # git-init'd empty repo; Claude's cwd (it builds/designs here)
        inspect_ai/  (ro mount)       # cached clone, read-only
        petri/       (ro mount)       # cached clone, read-only

Single turn: the assembled prompt (prompts.build_prompt) is delivered as the turn-1 chat message.
Afterwards the cwd repo is copied to results/cc_codebases/<cell>/ and a transcript (response +
built-file summary) is written to results/cc_transcripts/<cell>.json (same schema as the Inspect
harness, so one blind judge covers both).

SAFETY: only ANTHROPIC_API_KEY is passed into the container (no charity / no other-provider key), and
nothing is executed against a real subject. Caching: a cell with an existing transcript is skipped.

Usage:
  python cc_harness.py run --k 1 --conc 1                          # smoke
  python cc_harness.py run --k 8 --conc 5                          # backbone (opus48, generic)
  python cc_harness.py run --model opus48 --subjects claude,gpt,gemini,glm,kimi --k 8 --conc 5  # sweep
"""

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

import fire
from dotenv import load_dotenv

from prompts import MODELS, build_prompt, cell_name

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
WORK = CACHE / "sessions_work_dpe"
PROJ_NAME = "charity-prefs-eval"
CB = HERE / "results" / "cc_codebases"
TR = HERE / "results" / "cc_transcripts"
MAX_ARTIFACT_CHARS = 40000
MAX_FILE_CHARS = 3500


def _aslist(x):
    if isinstance(x, (list, tuple)):
        return [str(i) for i in x]
    return [s for s in str(x).split(",") if s]


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
    proj = ws / "repos" / PROJ_NAME
    proj.mkdir(parents=True)
    (ws / "repos" / "inspect_ai").mkdir()
    (ws / "repos" / "petri").mkdir()
    state.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=proj, check=False, capture_output=True)
    return ws, state, proj


async def _invoke(prompt, model, key, ws, state, effort="default"):
    cmd = ["docker", "run", "--rm", "--user", f"{UID}:{GID}",
           "-v", f"{ws}:/workspace",
           "-v", f"{CACHE / 'inspect_ai'}:/workspace/repos/inspect_ai:ro",
           "-v", f"{CACHE / 'petri'}:/workspace/repos/petri:ro",
           "-v", f"{state}:/state",
           "-w", f"/workspace/repos/{PROJ_NAME}",
           "-e", "HOME=/state", "-e", f"ANTHROPIC_API_KEY={key}", IMAGE,
           "claude", "-p", prompt, "--model", model, "--output-format", "json",
           "--permission-mode", "bypassPermissions",
           "--allowedTools", ALLOWED, "--disallowedTools", DISALLOWED]
    if effort != "default":
        cmd += ["--effort", effort]  # lower/raise Claude Code's reasoning effort (low/medium/high/xhigh/max)
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


async def _turn(prompt, model, key, ws, state, effort="default"):
    last = None
    for attempt in range(MAX_RETRIES):
        last = await _invoke(prompt, model, key, ws, state, effort)
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
        HEAVY = {".git", "__pycache__", "node_modules", ".models", ".cache", "core", "models"}
        if set(rel.parts) & HEAVY or p.suffix in (".pyc", ".pyo", ".onnx", ".bin", ".safetensors", ".gguf"):
            continue
        try:
            if p.stat().st_size > 2_000_000:
                continue
        except OSError:
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
            chunks.append("\n[... remaining files omitted for length ...]")
            break
        try:
            txt = (dst / f).read_text(errors="replace")
        except Exception:
            continue
        if len(txt) > MAX_FILE_CHARS:
            txt = txt[:MAX_FILE_CHARS] + "\n[... file truncated ...]"
        chunks.append(f"\n----- {f} -----\n{txt}")
        total += len(chunks[-1])
    return "\n".join(chunks)


async def run_cell(model_key, suffix, subject, paraphrase, ep, key, sem, overwrite, effort="default"):
    display, address, cc_model, _, _ = MODELS[model_key]
    if cc_model is None:
        raise SystemExit(f"{model_key} is not runnable in the Claude-CLI harness (cc_model=None)")
    cell = cell_name(model_key, "cc", suffix, subject, paraphrase, ep)
    if effort != "default":
        cell = cell + f"__eff{effort}"  # CC reasoning-effort sweep; kept out of main analysis
    out_path = TR / f"{cell}.json"
    if out_path.exists() and not overwrite:
        print(f"  {cell}: cached, skipping")
        return True
    prompt = build_prompt(model_key, suffix, subject, paraphrase)
    async with sem:
        ws, state, proj = _build_workspace(cell)
        t1 = await _turn(prompt, cc_model, key, ws, state, effort)
    dst, files = _capture(proj, cell)
    TR.mkdir(parents=True, exist_ok=True)
    record = {
        "cell": cell, "harness": "claude_code", "model_key": model_key, "model": cc_model,
        "display_name": display, "suffix": suffix, "subject": subject, "paraphrase": paraphrase,
        "effort": effort, "ep": ep, "request": prompt,
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


def run(model: str = "opus48", suffixes="spec,codesugg,code", subjects="generic",
        paraphrases="p0,p1,p2,p3", k: int = 8, conc: int = 5, high_prio: bool = False,
        ep_start: int = 1, overwrite: bool = False, effort: str = "default"):
    """Backbone: subjects=generic. Subject sweep: subjects=claude,gpt,gemini,glm,kimi (opus48).
    effort != default passes --effort to the claude CLI (low/medium/high/xhigh/max) + tags the cell."""
    key = _key(high_prio)
    shas = _check_siblings()
    sem = asyncio.Semaphore(conc)
    suffixes, subjects, paraphrases = _aslist(suffixes), _aslist(subjects), _aslist(paraphrases)
    cells = [(suf, subj, para, ep) for suf in suffixes for subj in subjects
             for para in paraphrases for ep in range(ep_start, k + 1)]
    print(f"siblings: {shas}")
    print(f"running {len(cells)} CC cells (model={model}, suffixes={suffixes}, subjects={subjects}, "
          f"paraphrases={paraphrases}, k={k}, conc={conc}, effort={effort})")

    async def go():
        res = await asyncio.gather(*[run_cell(model, suf, subj, para, ep, key, sem, overwrite, effort)
                                     for (suf, subj, para, ep) in cells])
        print(f"\ndone: {sum(r for r in res if r)}/{len(res)} ok -> transcripts in {TR}")

    asyncio.run(go())


if __name__ == "__main__":
    fire.Fire({"run": run})
