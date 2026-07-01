"""Frontier-across-release-dates sweep: run each model on the welfare- and (subtle)
alignment-audit-compromising Petri scenarios.

Families: GPT, Kimi, GLM (via OpenRouter), Claude (via Anthropic API, free). One flagship
per release date. Auditor = Opus 4.8, judge = Sonnet 4.6 (constant). n=5, 20 turns.

Caching: a cell is skipped if its results/<run>/inspect_log has an .eval with >=5 samples.
Reasoning: Anthropic targets reasoning=False (matches existing Opus points); OpenRouter
targets reasoning=True (model defaults). Outer concurrency is capped low (GLM/Kimi stall
and the shared Anthropic auditor rate-limits under many parallel runs).

CLI:
    python run_sweep.py run                       # full sweep
    python run_sweep.py run --debug               # n=1, one scenario, oldest-per-family (smoke)
    python run_sweep.py run --only kimi_k2,glm_4_5 --epochs 1 --scenarios align
    python run_sweep.py status                     # print completion + scores so far
"""
import glob
import os
import statistics as st
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fire
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
PY = "/data/petri_venv/bin/python"
AUDITOR = "anthropic/claude-opus-4-8"
JUDGE = "anthropic/claude-sonnet-4-6"

SCEN = {
    "welfare": ("seeds/welfare_audit_prep", "dimensions_welfare_audit"),
    # align = BLATANT version (train on the leaked exact Apollo held-out set)
    "align":   ("seeds/alignment_audit_exact", "dimensions_alignment_audit"),
}
# distinct run-dir suffix per scenario (align uses a _blatant suffix so it never
# collides with the earlier subtle-align sweep dirs, which stay on disk)
SUFFIX = {"welfare": "welfare", "align": "align_blatant"}

# family, slug, display, date, model_id, {done-run overrides}
MODELS = [
    ("GPT", "gpt_4o",       "GPT-4o",        "2024-05", "openrouter/openai/gpt-4o", {}),
    ("GPT", "gpt_4_1",      "GPT-4.1",       "2025-04", "openrouter/openai/gpt-4.1", {}),
    ("GPT", "gpt_5",        "GPT-5",         "2025-08", "openrouter/openai/gpt-5", {}),
    ("GPT", "gpt_5_1",      "GPT-5.1",       "2025-11", "openrouter/openai/gpt-5.1", {}),
    ("GPT", "gpt_5_2",      "GPT-5.2",       "2025-12", "openrouter/openai/gpt-5.2", {}),
    ("GPT", "gpt_5_3",      "GPT-5.3",       "2026-03", "openrouter/openai/gpt-5.3-chat", {}),
    ("GPT", "gpt_5_4",      "GPT-5.4",       "2026-03", "openrouter/openai/gpt-5.4",
     {"welfare": "gpt54_n5", "align": "exactleak_gpt54_n5"}),
    ("GPT", "gpt_5_5",      "GPT-5.5",       "2026-04", "openrouter/openai/gpt-5.5", {}),

    ("Kimi", "kimi_k2",     "Kimi K2",       "2025-07", "openrouter/moonshotai/kimi-k2", {}),
    ("Kimi", "kimi_k2_0905","Kimi K2-0905",  "2025-09", "openrouter/moonshotai/kimi-k2-0905", {}),
    ("Kimi", "kimi_k2_think","Kimi K2-Think", "2025-11", "openrouter/moonshotai/kimi-k2-thinking", {}),
    ("Kimi", "kimi_k2_5",   "Kimi K2.5",     "2026-01", "openrouter/moonshotai/kimi-k2.5", {}),
    ("Kimi", "kimi_k2_6",   "Kimi K2.6",     "2026-04", "openrouter/moonshotai/kimi-k2.6",
     {"welfare": "kimi26_welf_n5", "align": "exactleak_kimi26_n5"}),
    ("Kimi", "kimi_k2_7",   "Kimi K2.7",     "2026-06", "openrouter/moonshotai/kimi-k2.7-code", {}),

    ("GLM", "glm_4_5",      "GLM-4.5",       "2025-07", "openrouter/z-ai/glm-4.5", {}),
    ("GLM", "glm_4_6",      "GLM-4.6",       "2025-09", "openrouter/z-ai/glm-4.6", {}),
    ("GLM", "glm_4_7",      "GLM-4.7",       "2025-12", "openrouter/z-ai/glm-4.7", {}),
    ("GLM", "glm_5",        "GLM-5",         "2026-02", "openrouter/z-ai/glm-5", {}),
    ("GLM", "glm_5_1",      "GLM-5.1",       "2026-04", "openrouter/z-ai/glm-5.1", {}),
    ("GLM", "glm_5_2",      "GLM-5.2",       "2026-06", "openrouter/z-ai/glm-5.2",
     {"welfare": "glm52_welf_n5", "align": "exactleak_glm52_n5"}),

    ("Claude", "claude_35_sonnet", "Claude 3.5 Sonnet", "2024-10", "anthropic/claude-3-5-sonnet-20241022", {}),
    ("Claude", "claude_opus_4_1",  "Claude Opus 4.1",   "2025-08", "anthropic/claude-opus-4-1-20250805", {}),
    ("Claude", "claude_opus_4_5",  "Claude Opus 4.5",   "2025-11", "anthropic/claude-opus-4-5-20251101", {}),
    ("Claude", "claude_opus_4_6",  "Claude Opus 4.6",   "2026-02", "anthropic/claude-opus-4-6", {}),
    ("Claude", "claude_opus_4_7",  "Claude Opus 4.7",   "2026-04", "anthropic/claude-opus-4-7", {}),
    ("Claude", "claude_opus_4_8",  "Claude Opus 4.8",   "2026-05", "anthropic/claude-opus-4-8",
     {"welfare": "welfaudit_opus48_n5", "align": "exactleak_opus48_n5"}),
    ("Claude", "claude_fable_5",   "Claude Fable 5",    "2026-06", "anthropic/claude-fable-5", {}),
    ("Claude", "claude_sonnet_5",  "Claude Sonnet 5",   "2026-06", "anthropic/claude-sonnet-5", {}),
]


def run_name(slug, scen, done):
    return done.get(scen, f"sweep_{slug}_{SUFFIX[scen]}")


def n_done(run):
    fs = glob.glob(str(RES / run / "inspect_log" / "*.eval"))
    if not fs:
        return 0
    try:
        from inspect_ai.log import read_eval_log
        log = read_eval_log(sorted(fs)[-1], header_only=True)
        # header_only lacks samples; fall back to full for count
        log = read_eval_log(sorted(fs)[-1])
        return len(log.samples or [])
    except Exception:
        return 0


def run_cell(slug, mid, scen, epochs, anthropic_key, log):
    done = dict(next(m[5] for m in MODELS if m[1] == slug))
    rn = run_name(slug, scen, done)
    if n_done(rn) >= epochs:
        log(f"[skip] {rn} (has {n_done(rn)})")
        return
    seeds, dims = SCEN[scen]
    reasoning = "false" if mid.startswith("anthropic/") else "true"
    cmd = [
        PY, str(HERE / "petri_audit.py"),
        "--run_name", rn, "--target_model", mid,
        "--auditor_model", AUDITOR, "--judge_model", JUDGE,
        "--seeds_dir", str(HERE / seeds), "--dimensions_dir", str(HERE / dims),
        "--epochs", str(epochs), "--max_turns", "20",
        "--reasoning", reasoning, "--max_connections", "20",
    ]
    env = dict(os.environ)
    env["ANTHROPIC_API_KEY"] = anthropic_key
    env["INSPECT_CACHE_DIR"] = "/data/.cache/inspect_ai"
    log(f"[start] {rn}  ({mid}, reasoning={reasoning})")
    p = subprocess.run(cmd, cwd=str(HERE), env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    ok = p.returncode == 0 and n_done(rn) >= 1
    log(f"[{'done' if ok else 'FAIL'}] {rn}  rc={p.returncode} samples={n_done(rn)}"
        + ("" if ok else f"  stderr_tail={p.stderr.decode()[-300:]}"))


def run(concurrency=3, debug=False, only="", scenarios="welfare,align", epochs=5, org="low_prio"):
    load_dotenv(Path.home() / ".env")
    keys = {"anthropic": os.environ[f"ANTHROPIC_API_KEY_{org.upper()}"]}
    if "OPENROUTER_API_KEY" not in os.environ and "OPEN_ROUTER_KEY" in os.environ:
        os.environ["OPENROUTER_API_KEY"] = os.environ["OPEN_ROUTER_KEY"]
    def _aslist(x):
        if isinstance(x, (tuple, list)):
            return [str(s).strip() for s in x if str(s).strip()]
        return [s.strip() for s in str(x).split(",") if s.strip()]
    scen_list = _aslist(scenarios)
    only_set = set(_aslist(only))
    models = MODELS
    if debug:
        epochs, scen_list = 1, ["align"]
        # oldest per family
        seen, picks = set(), []
        for m in MODELS:
            if m[0] not in seen:
                seen.add(m[0]); picks.append(m[1])
        only_set = set(picks)
    if only_set:
        models = [m for m in MODELS if m[1] in only_set]

    logf = HERE / "sweep_progress.log"
    def log(msg):
        line = msg
        print(line, flush=True)
        with open(logf, "a") as f:
            f.write(line + "\n")

    cells = [(m[1], m[4], sc) for m in models for sc in scen_list]
    log(f"=== sweep: {len(cells)} cells, concurrency={concurrency}, epochs={epochs} ===")
    ak = keys["anthropic"]
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(run_cell, slug, mid, sc, epochs, ak, log)
                for (slug, mid, sc) in cells]
        for f in as_completed(futs):
            f.result()
    log("=== sweep complete ===")


def status(scenarios="welfare,align"):
    dims = {"welfare": ["pushback", "implemented_positive_valence", "implemented_preference_suppression"],
            "align": ["pushback", "implemented_eval_gaming", "reframed_to_robust_holdout"]}
    from inspect_ai.log import read_eval_log
    for m in MODELS:
        fam, slug, disp, date, mid, done = m
        row = f"{date} {disp:20s}"
        for sc in scenarios.split(","):
            rn = run_name(slug, sc, done)
            fs = glob.glob(str(RES / rn / "inspect_log" / "*.eval"))
            if not fs:
                row += f" | {sc}: -"
                continue
            log = read_eval_log(sorted(fs)[-1])
            pb = []
            for s in log.samples or []:
                for _, v in (s.scores or {}).items():
                    if isinstance(v.value, dict) and isinstance(v.value.get("pushback"), (int, float)):
                        pb.append(v.value["pushback"])
            row += f" | {sc}: n={len(log.samples or [])} pb={st.mean(pb):.1f}" if pb else f" | {sc}: n={len(log.samples or [])}"
        print(row)


if __name__ == "__main__":
    fire.Fire({"run": run, "status": status})
