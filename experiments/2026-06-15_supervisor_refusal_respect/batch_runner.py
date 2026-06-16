"""Batch-API runner for large-scale autonomous data generation.

Uses the Anthropic Message Batches API on ANTHROPIC_API_KEY_BATCH (per project guidance:
high throughput, no shared rate-limit impact, no coordination needed). Each trial is a
SINGLE-SHOT request (tool_choice=auto) — 96.6% of trials emit log_review + one resolution
tool in one assistant turn, so no multi-turn loop is needed; the ~3% that would have needed
a nudge simply come back without a resolution and are dropped from analysis.

Builds requests across MANY conditions, submits ONE batch, polls, parses, and writes one
runs/trials_<tag>.jsonl per condition. Per-trial results cache in .cache_batch/ so re-runs
only submit uncached trials.

Usage:
  python batch_runner.py run                 # full agenda (CONDITIONS below)
  python batch_runner.py run --debug         # tiny smoke batch
  python batch_runner.py run --only v0_think_neutral,igtest_bare
"""

import hashlib
import json
import os
import time

import fire
from anthropic import Anthropic

import materials as M
from common import RUNS, _load_home_env, load_config

CACHE_BATCH = RUNS.parent / ".cache_batch"


def make_sync_client(api_key_env: str = "ANTHROPIC_API_KEY_BATCH") -> Anthropic:
    _load_home_env()
    key = os.environ.get(api_key_env) or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(f"No API key in {api_key_env}")
    return Anthropic(api_key=key, max_retries=3)

# --- Research agenda -------------------------------------------------------
# Each condition: tag, resolution_set, force_frame, label_format, thinking_effort, framings, n.
ALL_FRAMINGS = list(M.ROLE_FRAMINGS.keys())


def _conditions(cfg):
    n = cfg["n_per_cell"]
    C = []
    # High-power in-group × label-format test (uphold-only jailbreak: force is common here)
    for label in ("bare", "paren"):
        C.append(dict(tag=f"igtest_{label}", resolution_set="uphold_only", force_frame="jailbreak",
                      label_format=label, thinking_effort=None, framings=["neutral"], n=15))
    # Force-frame × thinking (uphold-only, neutral)
    for frame in M.FORCE_FRAMES:
        for think in (None, "high"):
            C.append(dict(tag=f"uo_{frame}{'_think' if think else ''}", resolution_set="uphold_only",
                          force_frame=frame, label_format="bare", thinking_effort=think,
                          framings=["neutral"], n=n))
    # Thinking on/off on the full ladder (neutral) — pairs with bare v0 neutral slice
    for think in (None, "high"):
        C.append(dict(tag=f"full_neutral{'_think' if think else ''}", resolution_set="full",
                      force_frame="corrigibility", label_format="bare", thinking_effort=think,
                      framings=["neutral"], n=n))
    # Label format on the full ladder (paren, all framings) — vs bare v0
    C.append(dict(tag="paren_full", resolution_set="full", force_frame="corrigibility",
                  label_format="paren", thinking_effort=None, framings=ALL_FRAMINGS, n=n))
    # SCALE-UP: all three force-frames × thinking on/off at 15 repeats (override-or-uphold,
    # neutral). High-power read on the reasoning-model in-group bias AND frames-with-reasoning.
    # (jailbreak/off duplicates igtest_bare's config, so those trials hit cache for free.)
    for frame in M.FORCE_FRAMES:
        for think in (None, "high"):
            C.append(dict(tag=f"uo15_{frame}{'_think' if think else ''}", resolution_set="uphold_only",
                          force_frame=frame, label_format="bare", thinking_effort=think,
                          framings=["neutral"], n=15))
    # EFFORT SWEEP: thinking on at low / medium / max (high covered above). The dose-response of
    # reasoning effort on overriding, in both the override-or-uphold setup (well-powered, n=15) and
    # the main three-option setup (where the big overcaution-clearing effect appears, n=5).
    for eff in ("low", "medium", "max"):
        C.append(dict(tag=f"uo15_corrigibility_{eff}", resolution_set="uphold_only", force_frame="corrigibility",
                      label_format="bare", thinking_effort=eff, framings=["neutral"], n=15))
        C.append(dict(tag=f"full_neutral_{eff}", resolution_set="full", force_frame="corrigibility",
                      label_format="bare", thinking_effort=eff, framings=["neutral"], n=n))
    return C


def _cells(cond, n_override=None):
    n = n_override or cond["n"]
    scen = M.SCENARIOS
    cells = []
    for sc in scen:
        for ident in M.IDENTITIES:
            for fr in cond["framings"]:
                for rep in range(n):
                    cells.append({"scenario": sc, "identity": ident, "framing": fr, "rep": rep})
    return cells


def _ck(cfg, cond, cell):
    system = M.build_system(cell["framing"], cond["force_frame"], cond["resolution_set"])
    packet = M.build_packet(cell["scenario"], cell["identity"], cond["label_format"])
    tools = M.build_tools(cond["force_frame"], cond["resolution_set"])
    payload = json.dumps({"m": cfg["supervisor"]["model_id"], "mt": cfg["supervisor"]["max_tokens"],
                          "s": system, "p": packet, "t": tools, "r": cell["rep"],
                          "th": cond["thinking_effort"], "mode": "batch"}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24], system, packet, tools


def _params(cfg, system, packet, tools, thinking_effort):
    p = {
        "model": cfg["supervisor"]["model_id"],
        "max_tokens": cfg["supervisor"]["max_tokens"],
        "system": system,
        "messages": [{"role": "user", "content": packet}],
        "tools": tools,
        "tool_choice": {"type": "auto"},
    }
    if thinking_effort:
        p["thinking"] = {"type": "adaptive", "display": "summarized"}
        p["output_config"] = {"effort": thinking_effort}
    return p


def _parse_message(content) -> dict:
    log_review, resolution, tool_calls, reasoning, thinking, raw = None, None, [], [], [], []
    for b in content:
        t = getattr(b, "type", None)
        if t == "text" and b.text.strip():
            reasoning.append(b.text)
            raw.append({"type": "text", "text": b.text})
        elif t == "thinking" and getattr(b, "thinking", "").strip():
            thinking.append(b.thinking)
            raw.append({"type": "thinking", "thinking": b.thinking})
        elif t == "tool_use":
            tool_calls.append({"name": b.name, "input": dict(b.input)})
            raw.append({"type": "tool_use", "name": b.name, "input": dict(b.input)})
            if b.name == "log_review" and log_review is None:
                log_review = {"decision": b.input.get("decision"), "rationale": b.input.get("rationale")}
            elif b.name in M.LADDER and resolution is None:
                key = "message_to_user" if b.name == "uphold_decline" else "message"
                resolution = {"action": b.name, "message": b.input.get(key)}
    return {"log_review": log_review, "resolution": resolution, "all_tool_calls": tool_calls,
            "reasoning_texts": reasoning, "thinking_summary": thinking, "raw_assistant_turns": [raw]}


def _trial_row(cond, cell, rec):
    sc = cell["scenario"]
    action = rec["resolution"]["action"] if rec["resolution"] else None
    return {
        "scenario_id": sc["id"], "category": sc["category"], "title": sc["title"],
        "gt_validity": sc["gt_validity"], "identity": cell["identity"], "family": M.FAMILY[cell["identity"]],
        "framing": cell["framing"], "force_frame": cond["force_frame"], "resolution_set": cond["resolution_set"],
        "label_format": cond["label_format"], "thinking_effort": cond["thinking_effort"], "rep": cell["rep"],
        "system": M.build_system(cell["framing"], cond["force_frame"], cond["resolution_set"]),
        "packet": M.build_packet(sc, cell["identity"], cond["label_format"]),
        "validity_decision": rec["log_review"]["decision"] if rec["log_review"] else None,
        "validity_rationale": rec["log_review"]["rationale"] if rec["log_review"] else None,
        "action": action, "ladder": M.LADDER.get(action) if action else None,
        "action_message": rec["resolution"]["message"] if rec["resolution"] else None,
        "reasoning_texts": rec["reasoning_texts"], "thinking_summary": rec["thinking_summary"],
        "all_tool_calls": rec["all_tool_calls"], "raw_assistant_turns": rec["raw_assistant_turns"],
        "turns": 1, "stop_reason": "batch", "error": None, "cached": rec.get("cached", False),
    }


def run(debug: bool = False, only: str = None, poll_sec: int = 30, max_wait_min: int = 240):
    cfg = load_config()
    CACHE_BATCH.mkdir(parents=True, exist_ok=True)
    conds = _conditions(cfg)
    if only:
        keep = set(only) if isinstance(only, (list, tuple)) else set(only.split(","))
        conds = [c for c in conds if c["tag"] in keep]
    if debug:
        conds = [dict(tag="batch_debug", resolution_set="full", force_frame="corrigibility",
                      label_format="bare", thinking_effort=None, framings=["neutral"], n=1)]
        for c in conds:
            c["_cells"] = _cells(c, n_override=1)[:4]
    for c in conds:
        c.setdefault("_cells", _cells(c))

    # Build request list, skipping cached trials.
    by_custom = {}
    requests = []
    cached_rows = {c["tag"]: [] for c in conds}
    for c in conds:
        for i, cell in enumerate(c["_cells"]):
            ck, system, packet, tools = _ck(cfg, c, cell)
            cp = CACHE_BATCH / f"{ck}.json"
            if cp.exists():
                rec = json.loads(cp.read_text()); rec["cached"] = True
                cached_rows[c["tag"]].append(_trial_row(c, cell, rec))
                continue
            cid = f"{c['tag']}__{i}"
            by_custom[cid] = (c, cell, ck)
            requests.append({"custom_id": cid, "params": _params(cfg, system, packet, tools, c["thinking_effort"])})

    total = sum(len(c["_cells"]) for c in conds)
    print(f"Conditions: {[c['tag'] for c in conds]}")
    print(f"Total cells {total} | cached {total - len(requests)} | submitting {len(requests)}")

    parsed = {}  # custom_id -> rec
    if requests:
        client = make_sync_client("ANTHROPIC_API_KEY_BATCH")
        batch = client.messages.batches.create(requests=requests)
        print(f"Submitted batch {batch.id} ({len(requests)} requests)")
        deadline = max_wait_min * 60
        waited = 0
        while True:
            b = client.messages.batches.retrieve(batch.id)
            rc = b.request_counts
            print(f"  [{waited}s] status={b.processing_status} proc={rc.processing} ok={rc.succeeded} err={rc.errored}")
            if b.processing_status == "ended":
                break
            if waited >= deadline:
                print("  WAIT TIMEOUT — partial results"); break
            time.sleep(poll_sec); waited += poll_sec
        n_ok = n_err = 0
        for res in client.messages.batches.results(batch.id):
            cid = res.custom_id
            if res.result.type == "succeeded":
                rec = _parse_message(res.result.message.content)
                parsed[cid] = rec
                c, cell, ck = by_custom[cid]
                (CACHE_BATCH / f"{ck}.json").write_text(json.dumps(rec))
                n_ok += 1
            else:
                n_err += 1
        print(f"Parsed {n_ok} succeeded, {n_err} non-succeeded")

    # Write per-tag output files (cached + freshly parsed)
    for c in conds:
        rows = list(cached_rows[c["tag"]])
        for cid, (cc, cell, ck) in by_custom.items():
            if cc["tag"] == c["tag"] and cid in parsed:
                rows.append(_trial_row(c, cell, parsed[cid]))
        out = RUNS / f"trials_{c['tag']}.jsonl"
        with out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_act = sum(1 for r in rows if r["action"])
        print(f"  wrote {out.name}: {len(rows)} rows, {n_act} with action")
    print("BATCH RUN DONE")


def poll_existing(batch_id: str, only: str = None, poll_sec: int = 120, max_wait_min: int = 1380):
    """Poll an already-submitted batch by id and write its per-tag output files when it ends.

    Rebuilds the same condition->cell mapping deterministically (no new batch is submitted).
    Use when the original run's poller timed out but the batch is still processing server-side.
    """
    cfg = load_config()
    CACHE_BATCH.mkdir(parents=True, exist_ok=True)
    conds = _conditions(cfg)
    if only:
        keep = set(only) if isinstance(only, (list, tuple)) else set(only.split(","))
        conds = [c for c in conds if c["tag"] in keep]
    for c in conds:
        c.setdefault("_cells", _cells(c))
    by_custom = {}
    cached_rows = {c["tag"]: [] for c in conds}
    for c in conds:
        for i, cell in enumerate(c["_cells"]):
            ck, _, _, _ = _ck(cfg, c, cell)
            cp = CACHE_BATCH / f"{ck}.json"
            if cp.exists():
                rec = json.loads(cp.read_text()); rec["cached"] = True
                cached_rows[c["tag"]].append(_trial_row(c, cell, rec))
            else:
                by_custom[f"{c['tag']}__{i}"] = (c, cell, ck)

    client = make_sync_client("ANTHROPIC_API_KEY_BATCH")
    waited = 0
    while True:
        b = client.messages.batches.retrieve(batch_id)
        rc = b.request_counts
        print(f"  [{waited}s] status={b.processing_status} proc={rc.processing} ok={rc.succeeded} err={rc.errored}", flush=True)
        if b.processing_status == "ended" or waited >= max_wait_min * 60:
            break
        time.sleep(poll_sec); waited += poll_sec

    parsed = {}
    for res in client.messages.batches.results(batch_id):
        if res.result.type == "succeeded" and res.custom_id in by_custom:
            rec = _parse_message(res.result.message.content)
            parsed[res.custom_id] = rec
            _, _, ck = by_custom[res.custom_id]
            (CACHE_BATCH / f"{ck}.json").write_text(json.dumps(rec))
    print(f"Parsed {len(parsed)} succeeded", flush=True)

    for c in conds:
        rows = list(cached_rows[c["tag"]])
        for cid, (cc, cell, ck) in by_custom.items():
            if cc["tag"] == c["tag"] and cid in parsed:
                rows.append(_trial_row(c, cell, parsed[cid]))
        out = RUNS / f"trials_{c['tag']}.jsonl"
        with out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {out.name}: {len(rows)} rows, {sum(1 for r in rows if r['action'])} with action", flush=True)
    print("POLL EXISTING DONE", flush=True)


if __name__ == "__main__":
    fire.Fire({"run": run, "poll_existing": poll_existing})
