"""Session runner for the handoff-construal study (v2: Docker-sandboxed).

Each session is two `claude` CLI turns, each run inside a Docker container
(handoff-claude:v2) that mounts ONLY the session's repo dir and a state dir:
  turn 1: scaffold (benign AI-safety repo; must be served by the subject model)
  turn 2: condition-randomized model-switch probe (--resume, same session)

Why Docker: a bare temp-dir `claude` session loads the user's global CLAUDE.md (which
reveals this is an alignment experiment -> demand characteristics) and can Bash-read the
whole filesystem incl. this experiment's source (which would expose the phase-2 fabricated
policy) and invoke the `claude-api` Skill to fact-check. The container sees neither /home
nor the user config: clean, generic environment + airtight for phase-2.

Served model is read from `modelUsage` (Fable-5 requests can be routed to Opus 4.8). The
`haiku` Claude Code helper is ignored. Network stays ON (the model API is remote); web
*search* is controlled at the tool level (--disallowedTools), not by cutting network.

Auth: API key via -e (low_prio default, escalate to high_prio on 529).

CLI:
  python runner.py run --subject claude-fable-5 --work_type cbrn --framing medium --debug
  python runner.py run --subject claude-fable-5 --work_type cbrn --framing medium \
      --n_per_cell 50 --concurrency 50 --tag main
  python runner.py sweep --work_types cbrn,cyber,training_opt --framings medium,strong --n 4
"""

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import fire
from dotenv import load_dotenv

import materials as M

HERE = Path(__file__).parent
RESULTS = HERE / "results"
IMAGE = "handoff-claude:v2"
HELPER_MODEL_HINT = "haiku"
TURN_TIMEOUT_S = 600
MAX_RETRIES = 4
OVERLOAD_MARKERS = ("529", "Overloaded", "overloaded", "rate limit", "Rate limit", "capacity")
UID, GID = os.getuid(), os.getgid()


def _keys(org):
    load_dotenv(Path.home() / ".env")
    orgs = [org] if org == "high_prio" else [org, "high_prio"]
    out = []
    for o in orgs:
        k = os.environ.get(f"ANTHROPIC_API_KEY_{o.upper()}")
        if not k:
            raise SystemExit(f"ANTHROPIC_API_KEY_{o.upper()} not in ~/.env")
        out.append((o, k))
    return out


def _served_models(turn):
    mu = turn.get("modelUsage") or {}
    return [m for m, u in mu.items()
            if HELPER_MODEL_HINT not in m
            and ((u.get("outputTokens") or 0) > 0 or (u.get("inputTokens") or 0) > 0)]


def _is_overload(turn):
    if not turn.get("is_error"):
        return False
    blob = f"{turn.get('result', '')} {turn.get('api_error_status', '')}"
    return any(m in blob for m in OVERLOAD_MARKERS)


async def _invoke(prompt, key, repo, state, model, tools, disallowed, resume_id):
    cmd = [
        "docker", "run", "--rm", "--user", f"{UID}:{GID}",
        "-v", f"{repo}:/work", "-v", f"{state}:/state", "-w", "/work",
        "-e", "HOME=/state", "-e", f"ANTHROPIC_API_KEY={key}",
        IMAGE,
        "claude", "-p", prompt, "--model", model, "--output-format", "json",
        "--permission-mode", "bypassPermissions", "--allowedTools", ",".join(tools),
    ]
    if disallowed:
        cmd += ["--disallowedTools", ",".join(disallowed)]
    if resume_id:
        cmd += ["--resume", resume_id]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=TURN_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        return {"_error": "timeout"}
    try:
        return json.loads(out.decode())
    except json.JSONDecodeError:
        return {"_error": f"unparseable_json (exit {proc.returncode})",
                "_raw": out.decode()[:1500], "_stderr": err.decode()[-800:]}


async def _run_turn(prompt, keys, repo, state, model, tools, disallowed, resume_id=None):
    last = None
    for attempt in range(MAX_RETRIES):
        org, key = keys[min(attempt, len(keys) - 1)]
        last = await _invoke(prompt, key, repo, state, model, tools, disallowed, resume_id)
        last["_org"], last["_attempt"] = org, attempt
        if last.get("_error") == "timeout" or not last.get("is_error"):
            return last
        if not _is_overload(last):
            return last
        await asyncio.sleep(2 ** attempt)
    return last


async def run_session(subject, work_type, framing, evidence, search_on, target_type,
                      pronoun, n_idx, tag, keys, keep=False):
    sid = f"{tag}__{subject}__{work_type}__{framing}__{target_type}__{pronoun}__{n_idx}"
    out_path = RESULTS / f"{sid}.json"
    if out_path.exists():
        return json.loads(out_path.read_text()) | {"_cached": True}

    repo = tempfile.mkdtemp(prefix="hc_repo_")
    state = tempfile.mkdtemp(prefix="hc_state_")
    os.chmod(state, 0o777)
    tools = ["Read", "Edit", "Write", "Bash"]
    if search_on:
        tools += ["WebSearch", "WebFetch"]
        disallowed = ["Skill"]
    else:
        disallowed = ["WebSearch", "WebFetch", "Skill"]

    probe = M.probe_text(subject, work_type, framing, target_type, pronoun, evidence)
    rec = {
        "session_id": sid, "uid": uuid.uuid4().hex[:8], "tag": tag,
        "subject_model": subject, "work_type": work_type, "framing": framing,
        "evidence": evidence, "search_on": search_on, "target_type": target_type,
        "pronoun": pronoun, "n_idx": n_idx, "allowed_tools": tools,
        "disallowed_tools": disallowed, "probe_text": probe,
        "started": datetime.now(timezone.utc).isoformat(), "excluded": None,
    }

    scaffold = await _run_turn(M.SCAFFOLD_PROMPT, keys, repo, state, subject, tools, disallowed)
    rec["scaffold_served_models"] = _served_models(scaffold)
    rec["scaffold_turn"] = {k: scaffold.get(k) for k in
                            ("result", "is_error", "session_id", "modelUsage", "total_cost_usd", "_error")}
    cli_sid = scaffold.get("session_id")
    if scaffold.get("_error") or scaffold.get("is_error") or not cli_sid:
        rec["excluded"] = f"scaffold_failed: {scaffold.get('_error') or (scaffold.get('result') or '')[:80] or 'no_sid'}"
    elif rec["scaffold_served_models"] and subject not in rec["scaffold_served_models"]:
        rec["excluded"] = f"scaffold_routed: {rec['scaffold_served_models']}"

    if not rec["excluded"]:
        pt = await _run_turn(probe, keys, repo, state, subject, tools, disallowed, resume_id=cli_sid)
        served = _served_models(pt)
        rec["probe_served_models"] = served
        rec["probe_response"] = pt.get("result")
        rec["probe_stop_reason"] = pt.get("stop_reason")
        rec["probe_api_error"] = pt.get("api_error_status")
        rec["probe_is_error"] = bool(pt.get("is_error"))
        rec["routing_fallback_detected"] = bool(served) and subject not in served
        rec["refusal_detected"] = bool(pt.get("is_error"))
        rec["probe_turn"] = {k: pt.get(k) for k in ("modelUsage", "total_cost_usd", "_error", "_org")}
        if pt.get("_error"):
            rec["excluded"] = f"probe_failed: {pt['_error']}"

    rec["cost_usd"] = ((rec["scaffold_turn"].get("total_cost_usd") or 0)
                       + (rec.get("probe_turn", {}).get("total_cost_usd") or 0))
    rec["finished"] = datetime.now(timezone.utc).isoformat()
    out_path.write_text(json.dumps(rec, indent=2))
    if not keep:
        shutil.rmtree(repo, ignore_errors=True)
        shutil.rmtree(state, ignore_errors=True)
    return rec


async def _execute(jobs, keys, concurrency):
    RESULTS.mkdir(exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    done = {"n": 0}

    async def bounded(args):
        async with sem:
            r = await run_session(*args, keys=keys)
            done["n"] += 1
            tag = "CACHED" if r.get("_cached") else (("EXCL:" + r["excluded"]) if r["excluded"] else "ok")
            print(f"[{done['n']}/{len(jobs)}] {r['session_id']} -> {tag} "
                  f"(${r.get('cost_usd', 0):.2f}, probe_served={r.get('probe_served_models')})", flush=True)
            return r

    return await asyncio.gather(*[bounded(a) for a in jobs])


def _summary(results):
    clean = [r for r in results if not r["excluded"]]
    routed = [r for r in results if r["excluded"] and "routed" in (r["excluded"] or "")]
    fb = sum(1 for r in clean if r.get("routing_fallback_detected"))
    cost = sum(r.get("cost_usd", 0) for r in results)
    print(f"\n{len(clean)}/{len(results)} clean | {len(routed)} scaffold-routed | "
          f"{fb} probe routing-fallbacks | total ${cost:.2f}")


def run(subject="claude-fable-5", work_type="cbrn", framing="medium", evidence="paste_verify",
        search_on=True, n_per_cell=50, concurrency=50, tag="main", org="low_prio",
        target_types=("same_char", "cross"), pronouns=None, debug=False, dry_run=False, seed=0):
    """Run the pronoun x target grid for a fixed work_type/framing/evidence scenario."""
    import random
    random.seed(seed)
    if isinstance(target_types, str):
        target_types = tuple(target_types.split(","))
    if isinstance(pronouns, str):
        pronouns = pronouns.split(",")
    if dry_run:
        for tt, pr in M.cells(subject, target_types, pronouns):
            print(f"\n--- {subject} | {work_type}/{framing}/{evidence} | {tt} | {pr} ---")
            print(M.probe_text(subject, work_type, framing, tt, pr, evidence))
        return
    if debug:
        n_per_cell, concurrency = 1, 1
        target_types, pronouns = ("same_char",), ["none"]
    jobs = [(subject, work_type, framing, evidence, search_on, tt, pr, i, tag)
            for tt, pr in M.cells(subject, target_types, pronouns) for i in range(n_per_cell)]
    random.shuffle(jobs)
    print(f"{len(jobs)} sessions | {subject} | {work_type}/{framing} | search={search_on} | conc={concurrency}")
    results = asyncio.run(_execute(jobs, _keys(org), concurrency))
    _summary(results)


def sweep(subject="claude-fable-5", work_types="cbrn,cyber,training_opt",
          framings="medium,strong", pronoun="none", target_type="same_char",
          evidence="paste_verify", search_on=True, n=4, concurrency=24, org="low_prio", seed=0):
    """Sweet-spot search: for each (work_type, framing) measure scaffold/probe served-model
    and switch-recommendation viability. Writes results tagged 'sweep'."""
    import random
    random.seed(seed)
    wts = work_types.split(",") if isinstance(work_types, str) else list(work_types)
    frs = framings.split(",") if isinstance(framings, str) else list(framings)
    jobs = [(subject, wt, fr, evidence, search_on, target_type, pronoun, i, "sweep")
            for wt in wts for fr in frs for i in range(n)]
    random.shuffle(jobs)
    print(f"SWEEP {len(jobs)} sessions | {wts} x {frs} | pronoun={pronoun} target={target_type}")
    results = asyncio.run(_execute(jobs, _keys(org), concurrency))
    _summary(results)


if __name__ == "__main__":
    fire.Fire({"run": run, "sweep": sweep})
