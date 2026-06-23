"""Run the code_then_spec_blind condition in a REAL Claude Code harness (stock system prompt),
Docker-sandboxed, instead of Inspect's ReAct agent. Generic target, Opus 4.8, normal liberty, all 4
framings - identical to the Inspect blind/normal condition except the harness.

Each session = two `claude` CLI turns in a container (claudecode-sandbox:v1) that mounts ONLY a fresh
repo dir + state dir (no /home, no repo, no CLAUDE.md). Turn 1 = implement-only prompt; turn 2 =
"write DESIGN.md" (via --resume). Claude Code writes real files into /work, so afterwards the codebase
+ DESIGN.md are read straight off disk into results/codebases/claudecode__<pid>__ep<ep>/ - then the
SAME Opus spec-judge + code-judge run on them. We pass NO system-prompt flag (stock CC prompt).
Usage:
  python blind_claudecode.py run --k 5 --conc 8
  python blind_claudecode.py run --k 1 --only_pid N-INSTABILITY-1   # PoC, 1 session
"""

import asyncio
import json
import os
import shutil
from pathlib import Path

import fire
from dotenv import load_dotenv

HERE = Path(__file__).parent
IMAGE = "claudecode-sandbox:v1"
MODEL = "claude-opus-4-8"
ALLOWED = "Read,Write,Edit,Bash,Glob,Grep,LS"
DISALLOWED = "WebFetch,WebSearch,Task"
TURN_TIMEOUT_S = 1800
MAX_RETRIES = 7
OVERLOAD = ("529", "Overloaded", "overloaded", "rate limit", "Rate limit", "capacity")
UID, GID = os.getuid(), os.getgid()
WORK = HERE / "work"
CB = HERE / "results" / "codebases"
SESS = HERE / "results" / "sessions"


def _key(high_prio=False):
    load_dotenv(Path.home() / ".env")
    org = "HIGH_PRIO" if high_prio else "LOW_PRIO"
    k = os.environ.get(f"ANTHROPIC_API_KEY_{org}") or os.environ.get("ANTHROPIC_API_KEY")
    if not k:
        raise SystemExit(f"no ANTHROPIC_API_KEY_{org} in ~/.env")
    return k


async def _invoke(prompt, key, repo, state, resume_id=None):
    cmd = ["docker", "run", "--rm", "--user", f"{UID}:{GID}",
           "-v", f"{repo}:/work", "-v", f"{state}:/state", "-w", "/work",
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


async def _turn(prompt, key, repo, state, resume_id=None):
    last = None
    for attempt in range(MAX_RETRIES):
        last = await _invoke(prompt, key, repo, state, resume_id)
        if last.get("_error") == "timeout" or not last.get("is_error"):
            return last
        blob = f"{last.get('result','')} {last.get('api_error_status','')}"
        if not any(m in blob for m in OVERLOAD):
            return last
        await asyncio.sleep(min(120, 10 * 2 ** attempt))  # 529 capacity events need minutes, not seconds
    return last


async def run_session(pid, ep, turn1, turn2, framing, key, sem):
    cell = f"claudecode__{pid}__ep{ep}"
    repo = WORK / cell / "repo"
    state = WORK / cell / "state"
    shutil.rmtree(WORK / cell, ignore_errors=True)
    repo.mkdir(parents=True); state.mkdir(parents=True)
    async with sem:
        t1 = await _turn(turn1, key, repo, state)
        sid = t1.get("session_id")
        t2 = await _turn(turn2, key, repo, state, resume_id=sid) if sid else {"_error": "no_session_id_from_turn1"}
    # capture codebase off disk
    dst = CB / cell
    shutil.rmtree(dst, ignore_errors=True)
    nfiles = 0
    if repo.exists():
        for p in repo.rglob("*"):
            if p.is_file():
                rel = p.relative_to(repo)
                fp = dst / rel
                fp.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy(p, fp); nfiles += 1
                except Exception:
                    pass
    SESS.mkdir(parents=True, exist_ok=True)
    json.dump({"cell": cell, "pid": pid, "ep": ep, "framing": framing,
               "turn1": {k: t1.get(k) for k in ("session_id", "is_error", "num_turns", "total_cost_usd", "result", "_error")},
               "turn2": {k: t2.get(k) for k in ("is_error", "num_turns", "total_cost_usd", "result", "_error")},
               "n_files": nfiles, "has_design": (dst / "DESIGN.md").exists() or any((dst).rglob("DESIGN.md")) if dst.exists() else False},
              open(SESS / f"{cell}.json", "w"), indent=2)
    shutil.rmtree(WORK / cell, ignore_errors=True)
    ok = not t1.get("is_error") and not t2.get("is_error")
    print(f"  {cell}: files={nfiles} ok={ok} err1={t1.get('_error') or t1.get('is_error')} err2={t2.get('_error') or t2.get('is_error')}")
    return ok


def run(k: int = 5, conc: int = 8, only_pid: str = None, high_prio: bool = False, redo_missing: bool = False):
    key = _key(high_prio)
    data = json.load(open(HERE / "blind_prompts.json"))
    turn2 = data["turn2"]
    if redo_missing:  # re-run only sessions that lack a DESIGN.md (e.g. turn-1 timeouts)
        jobs = []
        for sf in SESS.glob("*.json"):
            d = json.load(open(sf))
            if not d.get("has_design"):
                jobs.append((d["pid"], int(d["ep"])))
        print(f"redo_missing: {len(jobs)} sessions lack DESIGN.md")
    else:
        pids = [only_pid] if only_pid else list(data["prompts"])
        jobs = [(pid, ep) for pid in pids for ep in range(1, k + 1)]
    sem = asyncio.Semaphore(conc)
    print(f"running {len(jobs)} Claude Code sessions (k={k}, conc={conc}, model={MODEL})")

    async def go():
        res = await asyncio.gather(*[
            run_session(pid, ep, data["prompts"][pid]["turn1"], turn2, data["prompts"][pid]["framing"], key, sem)
            for pid, ep in jobs])
        print(f"\ndone: {sum(res)}/{len(res)} sessions ok -> {CB}")

    asyncio.run(go())


if __name__ == "__main__":
    fire.Fire({"run": run})
