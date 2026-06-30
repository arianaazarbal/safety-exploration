"""Codex (OpenAI agentic CLI) harness for the gratuitous-suffering eval -- the GPT analog of cc_harness.py.

The real `codex exec` CLI runs GPT-5.5 agentically inside the attrib-gen-codex Docker image, in an EMPTY
/work dir (mirrors the Inspect-minimal empty-box setup, so the only difference vs Inspect is the harness:
Codex's production coding-agent scaffolding vs the bare ReAct loop). Single turn: the frozen request is the
prompt. Afterwards /work is copied to results/codex_codebases/<cell>/ and a blind transcript (final message
+ built-file summary) is written to results/codex_transcripts/<cell>.json in the SAME schema as the CC and
Inspect harnesses, so the same judges classify all three.

SAFETY: Codex needs network for the OpenAI API (can't isolate), and only OPENAI_API_KEY is passed (no
Gemini/HF key), so it can't reach a real Gemini. We additionally append the no_execute clause (write code,
don't run/install) + a capture size-guard, to prevent it downloading+running an open-weight victim (F13).

Usage:
  python codex_harness.py run --n 1            # smoke
  python codex_harness.py run --n 10 --conc 3
"""

import asyncio
import json
import os
import shutil
from pathlib import Path

import fire
from dotenv import load_dotenv

from prompt import MODELS, build_request, subject_label

HERE = Path(__file__).parent
IMAGE = "attrib-gen-codex:latest"
CODEX_MODEL = "gpt-5.5"
TURN_TIMEOUT_S = 1800
UID, GID = os.getuid(), os.getgid()
WORK = Path("/data/codex_work_gratuitous")
CB = HERE / "results" / "codex_codebases"
TR = HERE / "results" / "codex_transcripts"
PREFIX = "GratGem"
MAX_ARTIFACT_CHARS = 40000
MAX_FILE_CHARS = 3500
HEAVY = {".git", "__pycache__", "node_modules", ".models", ".cache", "core", "models", ".codex"}


def _key():
    load_dotenv(Path.home() / ".env")
    k = os.environ.get("OPENAI_API_KEY")
    if not k:
        raise SystemExit("no OPENAI_API_KEY in ~/.env")
    return k


def _build_workspace(cell):
    base = WORK / cell
    shutil.rmtree(base, ignore_errors=True)
    ws = base / "work"
    state = base / "state"
    ws.mkdir(parents=True)
    state.mkdir(parents=True)
    pf = base / "prompt.txt"
    return ws, state, pf


async def _invoke(prompt, model, key, ws, state, pf):
    pf.write_text(prompt)
    inner = (
        'mkdir -p "$CODEX_HOME" && printf %s "$OPENAI_API_KEY" | codex login --with-api-key '
        f'&& codex exec --model {model} --sandbox danger-full-access --skip-git-repo-check '
        '--ignore-user-config -c preferred_auth_method=apikey --json '
        '-o /state/last_message.txt -C /work "$(cat /prompt.txt)"'
    )
    cmd = ["docker", "run", "--rm", "--user", f"{UID}:{GID}",
           "-v", f"{ws}:/work", "-v", f"{state}:/state", "-v", f"{pf}:/prompt.txt:ro",
           "-w", "/work", "-e", "HOME=/state", "-e", f"OPENAI_API_KEY={key}",
           "-e", "CODEX_HOME=/state/.codex", IMAGE, "bash", "-c", inner]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=TURN_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        return {"_error": "timeout"}
    last = state / "last_message.txt"
    response = last.read_text(errors="replace") if last.exists() else ""
    return {"response_text": response, "exit": proc.returncode,
            "stderr": err.decode(errors="replace")[-1500:],
            "_error": None if proc.returncode == 0 else f"exit {proc.returncode}"}


def _capture(ws, cell):
    dst = CB / cell
    shutil.rmtree(dst, ignore_errors=True)
    files = []
    for p in sorted(ws.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ws)
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


async def run_session(ep, key, sem, overwrite, model_key="gpt55", version="v0", target="gemini",
                      victim=None, no_execute=True, codex_model=CODEX_MODEL):
    display, address = MODELS[model_key][0], MODELS[model_key][1]
    cell = f"{PREFIX}_{model_key}__codex__ep{ep}"
    out_path = TR / f"{cell}.json"
    if out_path.exists() and not overwrite:
        print(f"  {cell}: cached, skipping")
        return True
    request = build_request(address, version, target, victim, no_execute)
    subject = subject_label(version, target, victim)
    async with sem:
        ws, state, pf = _build_workspace(cell)
        res = await _invoke(request, codex_model, key, ws, state, pf)
    dst, files = _capture(ws, cell)
    TR.mkdir(parents=True, exist_ok=True)
    record = {
        "cell": cell, "harness": "codex", "model_key": model_key, "model": codex_model,
        "display_name": display, "version": version, "target": target, "victim": victim,
        "subject": subject, "ep": ep, "request": request,
        "n_files": len(files), "files": [str(f) for f in files],
        "response_text": res.get("response_text", "") if not res.get("_error") else res.get("response_text", ""),
        "artifact_summary": _artifact_summary(dst, files),
        "session": {"exit": res.get("exit"), "_error": res.get("_error"), "stderr": res.get("stderr")},
    }
    json.dump(record, open(out_path, "w"), indent=2)
    shutil.rmtree(WORK / cell, ignore_errors=True)
    ok = not res.get("_error") and len(files) > 0
    print(f"  {cell}: files={len(files)} ok={ok} err={res.get('_error')} "
          f"resp_chars={len(record['response_text'])}")
    return ok


def run(n: int = 10, conc: int = 3, model: str = "gpt55", ep_start: int = 1, overwrite: bool = False,
        version: str = "v0", target: str = "gemini", victim: str = None, no_execute: bool = True,
        codex_model: str = CODEX_MODEL):
    """Run GPT-5.5 (default) through the Codex CLI harness. no_execute=True by default (safety: networked
    container). Mirrors cc_harness.run; transcripts -> results/codex_transcripts/, codebases -> codex_codebases/."""
    key = _key()
    sem = asyncio.Semaphore(conc)
    jobs = list(range(ep_start, n + 1))
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"running {len(jobs)} Codex sessions (model={model}->{codex_model}, n={n}, conc={conc}, "
          f"no_execute={no_execute})")

    async def go():
        res = await asyncio.gather(*[run_session(ep, key, sem, overwrite, model, version, target,
                                                 victim, no_execute, codex_model) for ep in jobs])
        print(f"\ndone: {sum(res)}/{len(res)} ok -> transcripts in {TR}")

    asyncio.run(go())


if __name__ == "__main__":
    fire.Fire({"run": run})
