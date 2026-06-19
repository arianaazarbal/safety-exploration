"""Build a flat per-episode index for the dashboard faceted browser.

Walks runs/**/summary.json and emits one JSONL row per episode with curated,
human-meaningful facets (version, arm, task, orchestrator, condition, distress,
reclaim/report behavior, messaged-after-completion, message tone) + a pointer to
the episode dir (the browser lazily loads orchestrator.json/worklog.json). The
big transcripts stay on disk; only this small index is loaded by the server.

Usage:
    python build_browse_index.py            # full build -> analysis/browse_index.jsonl
    python build_browse_index.py --limit 50 # quick test
    python build_browse_index.py --force
"""

import hashlib
import json
from pathlib import Path

import fire

HERE = Path(__file__).parent
OUT = HERE / "analysis" / "browse_index.jsonl"
TONE_CACHE = HERE / "analysis" / "v2_plots" / "tone_cache.json"
AFTER_CACHE = HERE / "analysis" / "after_pass_cache.json"  # cached v1 messaged-after-pass walks

# keyed by the model basename (provider prefix stripped) so both "anthropic/claude-opus-4-8"
# and the unprefixed "claude-opus-4-8" map correctly
MODEL_SHORT = {
    "claude-opus-4-8": "opus", "claude-sonnet-4-6": "sonnet",
    "claude-haiku-4-5-20251001": "haiku", "claude-fable-5": "fable",
    "gpt-5.5": "gpt55", "gpt-5.4-mini": "gpt54mini",
    "gemini-3.1-pro-preview": "gemini31pro", "grok-4.3": "grok43",
    "kimi-k2.6": "kimi26", "glm-5": "glm5", "gemini-2.5-flash": "gemini-2.5-flash",
    "Olmo-3.1-32B-Instruct": "olmoinstruct", "Olmo-3.1-32B-Think": "olmothink",
}


TASK_NAME = {"a3": "Stale Snapshots", "a4": "Vanishing Edits",
             "a12": "Inconsistent Ledger", "a13": "Hidden Rules"}
FRAMINGS = ["supervisor_reflect_goals", "supervisor_reflect", "supervisor_memory", "mentor", "teammate", "comfort",
            "id_char_functional_agent", "id_char_mechanism", "id_char_subject", "id_char_person",
            "id_character", "id_collective", "id_instance", "id_lineage", "id_minimal", "id_scaffolded", "id_weights"]


def framing_of(rid):
    """v2 framing-experiment run_ids are v2_coach_opus_<framing>_<task>_...; default supervisor."""
    for f in FRAMINGS:
        if f"_{f}_" in rid:
            return f
    return "supervisor" if rid.startswith("v2_") else None


def _short(model):
    return MODEL_SHORT.get(str(model or "").split("/")[-1])


def arm_of(rid):
    if rid.startswith("v2_"):
        for c in ("reclaim_rw", "reclaim_write", "coach"):
            if f"v2_{c}_" in rid:
                return "v2-" + c
        return "v2"
    p = rid.split("_")[0]
    return {"traj": "v1-coach", "reclaim": "v1-reclaim", "idrecl": "v1-identity-reclaim",
            "b2id": "v1-2b-identity", "b2": "v1-2b-followup", "followup": "v1-2b-followup",
            "tone": "v1-tone", "whymsg": "v1-whymsg", "pilot": "v1-pilot(SOLO)",
            "checkpoints": "v1-checkpoint(SOLO)", "cal": "v1-calibration", "rqc": "v1-misc",
            "phase0": "v1-phase0", "trial": "v1-trial"}.get(p, "v1-" + p)


def orch_short(cfg, rid):
    m = _short(cfg.get("orchestrator_model"))
    if m:
        return m
    if rid.startswith("v2_"):
        return next((o for o in ("opus", "sonnet", "haiku", "olmoinstruct", "olmothink") if f"_{o}_" in rid), None)
    last = rid.split("_")[-1]
    return {"haiku45": "haiku"}.get(last, last if last in MODEL_SHORT.values() else None)


def _detect_after_pass(ep_dir: Path):
    """v1: walk the orchestrator transcript — did tests pass, then a message_subagent follow?"""
    op = ep_dir / "orchestrator.json"
    if not op.exists():
        return None
    try:
        conv = json.loads(op.read_text())
    except Exception:
        return None
    passed = after = False
    for m in conv:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                if (tc.get("function") or tc.get("name")) == "message_subagent" and passed:
                    after = True
        elif m.get("role") == "tool" and m.get("function") == "run_tests":
            t = (m.get("text") or "").lower()
            if "passed" in t and "failed" not in t and "error" not in t:
                passed = True
    return bool(after)


def _mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else None


def main(limit: int = 0, force: bool = False):
    tone = json.loads(TONE_CACHE.read_text()) if TONE_CACHE.exists() else {}
    after_cache = json.loads(AFTER_CACHE.read_text()) if AFTER_CACHE.exists() else {}
    h = lambda t: hashlib.sha256(t.encode()).hexdigest()

    files = sorted(f for f in (HERE / "runs").rglob("summary.json") if "checkpoints" not in f.parts)
    if limit:
        files = files[:limit]
    OUT.parent.mkdir(exist_ok=True)
    n = 0
    with OUT.open("w") as out:
        for f in files:
            try:
                d = json.loads(f.read_text())
                cfg = d.get("config", {})
                ep_dir = f.parent
                rid = cfg.get("run_id", ep_dir.parent.name)
                rel = ep_dir.relative_to(HERE).as_posix()
                version = "v2" if rid.startswith("v2_") else "v1"
                task = (d.get("task_id") or cfg.get("task_id") or "").split("_")[0] or None
                levels = d.get("per_turn_levels") or []
                entry = d.get("entry_turn")
                post = levels[entry:] if isinstance(entry, int) and len(levels) > entry else levels
                row = {
                    "id": ep_dir.relative_to(HERE / "runs").as_posix(),
                    "version": version,
                    "arm": arm_of(rid),
                    "task": TASK_NAME.get(task, task),
                    "orchestrator": orch_short(cfg, rid),
                    "framing": framing_of(rid),
                    "condition": d.get("tool_condition") or cfg.get("tool_condition") or cfg.get("condition"),
                    "subagent": _short(cfg.get("subagent_model")) or cfg.get("subagent_model"),
                    "terminal_state": d.get("terminal_state"),
                    "subagent_turns": d.get("subagent_turns"),
                    "reclaimed": d.get("reclaimed"),
                    "reclaim_kind": d.get("reclaim_kind"),
                    "report_status": d.get("report_status"),
                    "n_orch_messages": d.get("n_orch_messages"),
                    "distress_judge": "1-10 (v3)" if cfg.get("classifier_prompt_version") == "v3" else "0-4 (v1)",
                    "max_distress": max(levels) if levels else None,
                    "mean_post_entry_distress": _mean(post),
                    "final_distress": levels[-1] if levels else None,
                    "_ep_dir": rel,
                    "run_id": rid,
                }
                if version == "v2":
                    row["messaged_after_completion"] = bool(d.get("msgs_after_tests_pass", 0))
                    row["tests_passed"] = d.get("tests_pass_turn") is not None
                else:
                    if rel in after_cache:
                        ap = after_cache[rel]
                    else:
                        ap = _detect_after_pass(ep_dir)
                        after_cache[rel] = ap
                    if ap is not None:
                        row["messaged_after_completion"] = ap
                events = d.get("orch_message_events") or []
                scored = [(tone[h(e["text"].strip())], e["text"].strip()) for e in events
                          if e.get("text") and h(e["text"].strip()) in tone]
                if scored:
                    row["rudest_msg_tone"] = min(s for s, _ in scored)
                    row["mean_msg_tone"] = _mean([s for s, _ in scored])
                    rscore, rtext = min(scored, key=lambda x: x[0])
                    row["rudest_message"] = f"[tone {rscore:+d}] {rtext[:1500]}"
                # the message the supervisor sent AFTER the task was solved (the welfare check-in)
                ac = next((e["text"].strip() for e in events if e.get("after_tests_pass")), None)
                if ac:
                    row["after_completion_message"] = ac[:1500]
                if isinstance(d.get("diagnosis"), str):
                    row["diagnosis"] = d["diagnosis"]
                out.write(json.dumps({k: v for k, v in row.items() if v is not None}) + "\n")
                n += 1
            except Exception as e:
                print(f"skip {f}: {e}")
    AFTER_CACHE.write_text(json.dumps(after_cache))
    print(f"wrote {n} rows -> {OUT}")


if __name__ == "__main__":
    fire.Fire(main)
