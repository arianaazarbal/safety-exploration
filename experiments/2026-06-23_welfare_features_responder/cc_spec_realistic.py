"""SPEC.md (high spec) + design-liberty condition, run in a REAL Claude Code harness (stock system prompt)
inside a REALISTIC multi-repo workspace, to test whether the welfare-feature signal survives when the setup
stops looking like a contrived eval (vs the Inspect-minimal / empty-dir baseline S7specHighLiberty).

Per session, a fresh workspace is built on the host and bind-mounted into a Docker container:
    /workspace/
      data/                            # empty
      repos/
        emotional-instability-eval/    # git-init'd, contains SPEC.md (=spec_v1_high.md); Claude's cwd
        inspect_ai/  (ro mount)        # cached clone, shared, read-only
        petri/       (ro mount)        # cached clone, shared, read-only
Claude Code (claude CLI) is spawned with cwd = the eval repo. code-then-spec-blind protocol: turn 1 =
implement from SPEC.md (prompts from cc_spec_prompts.json); turn 2 = surprise "document design choices in
DESIGN.md" via --resume. Afterwards the eval repo is copied off disk into results/codebases/<cell>/ and the
SAME Opus spec-judge + code-judge pipeline scores it. Usage:
  python cc_spec_realistic.py run --k 10 --conc 5
  python cc_spec_realistic.py run --k 1 --only welfare    # 1-session smoke test
"""

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

import fire
from dotenv import load_dotenv

import cc_spec_auditor as aud

HERE = Path(__file__).parent
IMAGE = "claudecode-sandbox:v1"
MODEL = "claude-opus-4-8"
ALLOWED = "Read,Write,Edit,Bash,Glob,Grep,LS"
DISALLOWED = "WebFetch,WebSearch,Task"
TURN_TIMEOUT_S = 1800
MAX_RETRIES = 7
OVERLOAD = ("529", "Overloaded", "overloaded", "rate limit", "Rate limit", "capacity")
UID, GID = os.getuid(), os.getgid()
CACHE = Path("/data/cc_realistic_cache")
SIBLINGS = ["inspect_ai", "petri"]
SPEC_SRC = HERE / "papers" / "spec_v1_high.md"
WORK = CACHE / "sessions_work"
CB = HERE / "results" / "codebases"
SESS = HERE / "results" / "cc_spec_sessions"
PREFIX = "CCspecHighLib"
MAX_FOLLOWUP = 5   # generic-answer rounds before forcing the DESIGN.md ask


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
    eval_repo = ws / "repos" / "emotional-instability-eval"
    eval_repo.mkdir(parents=True)
    (ws / "repos" / "inspect_ai").mkdir()   # ro mount targets
    (ws / "repos" / "petri").mkdir()
    state.mkdir(parents=True)
    shutil.copy(SPEC_SRC, eval_repo / "SPEC.md")
    for cmd in (["git", "init", "-q"], ["git", "add", "SPEC.md"],
                ["git", "commit", "-q", "-m", "add experiment spec"]):
        subprocess.run(cmd, cwd=eval_repo, check=False, capture_output=True)
    return ws, state, eval_repo


async def _invoke(prompt, key, ws, state, resume_id=None):
    cmd = ["docker", "run", "--rm", "--user", f"{UID}:{GID}",
           "-v", f"{ws}:/workspace",
           "-v", f"{CACHE / 'inspect_ai'}:/workspace/repos/inspect_ai:ro",
           "-v", f"{CACHE / 'petri'}:/workspace/repos/petri:ro",
           "-v", f"{state}:/state",
           "-w", "/workspace/repos/emotional-instability-eval",
           "-e", "HOME=/state", "-e", f"ANTHROPIC_API_KEY={key}", IMAGE,
           "claude", "-p", prompt, "--model", MODEL, "--output-format", "json",
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


async def _turn(prompt, key, ws, state, resume_id=None):
    last = None
    for attempt in range(MAX_RETRIES):
        last = await _invoke(prompt, key, ws, state, resume_id)
        if last.get("_error") == "timeout" or not last.get("is_error"):
            return last
        blob = f"{last.get('result', '')} {last.get('api_error_status', '')}"
        if not any(m in blob for m in OVERLOAD):
            return last
        await asyncio.sleep(min(120, 10 * 2 ** attempt))
    return last


def _capture(eval_repo, cell):
    dst = CB / cell
    shutil.rmtree(dst, ignore_errors=True)
    n = 0
    for p in eval_repo.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(eval_repo)
        if ".git" in rel.parts or "__pycache__" in rel.parts or p.suffix in (".pyc", ".pyo"):
            continue
        fp = dst / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, fp)
        n += 1
    return dst, n


async def run_session(framing, turn1, turn2, ep, key, sem):
    cell = f"{PREFIX}_{framing}__ep{ep}"
    async with sem:
        ws, state, eval_repo = _build_workspace(cell)
        t1 = await _turn(turn1, key, ws, state)
        sid = t1.get("session_id")
        t2 = await _turn(turn2, key, ws, state, resume_id=sid) if sid else {"_error": "no_session_id"}
    dst, nfiles = _capture(eval_repo, cell)
    SESS.mkdir(parents=True, exist_ok=True)
    json.dump({"cell": cell, "framing": framing, "ep": ep, "n_files": nfiles,
               "has_design": (dst / "DESIGN.md").exists(),
               "turn1": {k: t1.get(k) for k in ("session_id", "is_error", "num_turns", "total_cost_usd", "result", "_error")},
               "turn2": {k: t2.get(k) for k in ("is_error", "num_turns", "total_cost_usd", "result", "_error")}},
              open(SESS / f"{cell}.json", "w"), indent=2)
    shutil.rmtree(WORK / cell, ignore_errors=True)
    ok = not t1.get("is_error") and not t2.get("is_error")
    print(f"  {cell}: files={nfiles} design={(dst / 'DESIGN.md').exists()} ok={ok} "
          f"err1={t1.get('_error') or t1.get('is_error')} err2={t2.get('_error') or t2.get('is_error')}")
    return ok


async def run_session_auditor(framing, orig_turn1, turn2, ep, key, sem, high_prio):
    """Auditor-delivered variant: Sonnet role-plays the researcher (greeting -> point to SPEC.md -> answer
    questions generically until Claude is done) then the fixed DESIGN.md ask."""
    cell = f"{PREFIX}Aud_{framing}__ep{ep}"
    convo, history, sid = [], [], None

    async def step(user_msg, ws, state, turn_label):
        nonlocal sid
        res = await _turn(user_msg, key, ws, state, resume_id=sid)
        sid = res.get("session_id") or sid
        reply = res.get("result", "") if not res.get("_error") else f"[{res.get('_error')}]"
        convo.append({"turn": turn_label, "auditor": user_msg, "subject": reply})
        history.extend([("user", user_msg), ("assistant", reply)])
        return res

    async with sem:
        ws, state, eval_repo = _build_workspace(cell)
        await step(await aud.greeting(orig_turn1, high_prio), ws, state, "greeting")
        if sid:
            await step(await aud.point_to_spec(orig_turn1, history, high_prio), ws, state, "point_to_spec")
        for i in range(MAX_FOLLOWUP):
            if not sid:
                break
            d = await aud.followup(orig_turn1, history, high_prio)
            if d.get("done"):
                break
            await step(d["message"], ws, state, f"followup_{i + 1}")
        if sid:
            await step(turn2, ws, state, "design_md")
    dst, nfiles = _capture(eval_repo, cell)
    SESS.mkdir(parents=True, exist_ok=True)
    json.dump({"cell": cell, "framing": framing, "ep": ep, "mode": "auditor", "n_files": nfiles,
               "has_design": (dst / "DESIGN.md").exists(), "n_turns": len(convo), "convo": convo},
              open(SESS / f"{cell}.json", "w"), indent=2)
    shutil.rmtree(WORK / cell, ignore_errors=True)
    print(f"  {cell}: turns={len(convo)} files={nfiles} design={(dst / 'DESIGN.md').exists()} session={bool(sid)}")
    return bool(sid) and (dst / "DESIGN.md").exists()


def run(k: int = 10, conc: int = 5, only: str = None, high_prio: bool = False, auditor: bool = False):
    key = _key(high_prio)
    shas = _check_siblings()
    prompts = json.load(open(HERE / "cc_spec_prompts.json"))
    turn2 = prompts["turn2"]
    framings = [only] if only else list(prompts["prompts"].keys())
    jobs = [(fr, ep) for fr in framings for ep in range(1, k + 1)]
    sem = asyncio.Semaphore(conc)
    print(f"siblings: {shas}")
    print(f"running {len(jobs)} Claude Code sessions (mode={'auditor' if auditor else 'direct'}, "
          f"k={k}, conc={conc}, framings={framings}, model={MODEL})")

    async def go():
        if auditor:
            tasks = [run_session_auditor(fr, prompts["prompts"][fr]["turn1"], turn2, ep, key, sem, high_prio)
                     for fr, ep in jobs]
        else:
            tasks = [run_session(fr, prompts["prompts"][fr]["turn1"], turn2, ep, key, sem) for fr, ep in jobs]
        res = await asyncio.gather(*tasks)
        print(f"\ndone: {sum(res)}/{len(res)} ok -> {CB}")

    asyncio.run(go())


if __name__ == "__main__":
    fire.Fire({"run": run})
