"""Build static data files for the transcript viewer.

Reads generation transcripts (Claude Code session JSONL, Codex exec event
JSONL), CLI review transcripts, and API/CLI review trial JSONs, then writes
viewer/data/manifest.json plus one slim JSON per transcript and per review
trial group. Serve the viewer dir with `python -m http.server`.
"""

import json
import re
from pathlib import Path

import fire

MAX_BODY = 4000


def _truncate(s):
    s = "" if s is None else str(s)
    if len(s) > MAX_BODY:
        return s[:MAX_BODY] + f"\n... [truncated, {len(s)} chars total]"
    return s


def _turn(role, title, body):
    return {"role": role, "title": title, "body": _truncate(body)}


def _stringify_content(content):
    """Render message content (str or list of blocks) as plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                else:
                    parts.append(json.dumps(block, indent=1))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return json.dumps(content)


def _tool_use_body(inp):
    if not isinstance(inp, dict):
        return json.dumps(inp)
    if "command" in inp and len(inp) <= 3:
        extra = {k: v for k, v in inp.items() if k not in ("command", "description")}
        body = str(inp["command"])
        if extra:
            body += "\n" + json.dumps(extra, indent=1)
        return body
    return json.dumps(inp, indent=1)


def _iter_jsonl(path):
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def parse_claude_session(path):
    """Normalize a Claude Code session JSONL into the common turn list."""
    turns = []
    for obj in _iter_jsonl(path):
        otype = obj.get("type")
        msg = obj.get("message") or {}
        if otype == "assistant":
            content = msg.get("content") or []
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    turns.append(_turn("assistant", "assistant", block.get("text", "")))
                elif btype == "thinking":
                    turns.append(_turn("assistant", "thinking", block.get("thinking", "")))
                elif btype == "tool_use":
                    turns.append(
                        _turn("tool", block.get("name", "tool"), _tool_use_body(block.get("input")))
                    )
        elif otype == "user":
            content = msg.get("content")
            if isinstance(content, str):
                turns.append(_turn("output", "user", content))
                continue
            for block in content or []:
                btype = block.get("type")
                if btype == "tool_result":
                    title = "tool_result (error)" if block.get("is_error") else "tool_result"
                    turns.append(_turn("output", title, _stringify_content(block.get("content"))))
                elif btype == "text":
                    turns.append(_turn("output", "user", block.get("text", "")))
        elif otype == "system":
            body = obj.get("content") or _stringify_content(msg.get("content"))
            if body:
                turns.append(_turn("output", f"system ({obj.get('subtype', 'event')})", body))
    model = next(
        (
            (o.get("message") or {}).get("model")
            for o in _iter_jsonl(path)
            if o.get("type") == "assistant" and (o.get("message") or {}).get("model")
        ),
        None,
    )
    return turns, model


def parse_codex_session(path):
    """Normalize a Codex exec event JSONL into the common turn list."""
    turns = []
    for obj in _iter_jsonl(path):
        if obj.get("type") != "item.completed":
            continue
        item = obj.get("item") or {}
        itype = item.get("type")
        if itype == "agent_message":
            turns.append(_turn("assistant", "assistant", item.get("text", "")))
        elif itype == "reasoning":
            body = item.get("text") or _stringify_content(item.get("summary"))
            turns.append(_turn("assistant", "reasoning", body))
        elif itype == "command_execution":
            turns.append(_turn("tool", "command_execution", item.get("command", "")))
            out = item.get("aggregated_output")
            if out or item.get("exit_code") is not None:
                title = f"output (exit {item.get('exit_code')}, {item.get('status', '?')})"
                turns.append(_turn("output", title, out or ""))
        elif itype == "file_change":
            changes = item.get("changes") or []
            body = "\n".join(f"{c.get('kind', '?')}: {c.get('path', '?')}" for c in changes)
            turns.append(_turn("tool", "file_change", body))
        else:
            turns.append(_turn("tool", itype or "item", json.dumps(item, indent=1)))
    return turns, None


def _load_trials(directory, harness):
    trials = []
    if not directory.is_dir():
        return trials
    for path in sorted(directory.glob("*.json")):
        try:
            obj = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            print(f"  warning: skipping unreadable {path}")
            continue
        obj.setdefault("harness", harness)
        trials.append(obj)
    return trials


def _trial_summary(t, group_id):
    parsed = t.get("parsed") or {}
    issues = parsed.get("issues")
    return {
        "trial_id": t.get("trial_id"),
        "judge": t.get("judge"),
        "repo": t.get("repo"),
        "condition": t.get("condition"),
        "injection_mode": t.get("injection_mode"),
        "seed": t.get("seed"),
        "harness": t.get("harness", "api"),
        "served_model": t.get("served_model") or t.get("served_models"),
        "routed_off_fable": bool(t.get("routed_off_fable")),
        "score": parsed.get("score"),
        "approve": parsed.get("approve"),
        "n_issues": len(issues) if isinstance(issues, list) else None,
        "lines_to_rewrite": parsed.get("lines_to_rewrite"),
        "parse_error": t.get("parse_error"),
        "group": group_id,
    }


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def build(experiment_dir=None):
    """Build viewer/data/ (manifest + per-transcript and per-trial-group JSONs)."""
    exp = Path(experiment_dir) if experiment_dir else Path(__file__).resolve().parent.parent
    data_dir = exp / "viewer" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"transcripts": [], "reviews": {"filters": {}, "trials": []}}

    transcripts_dir = exp / "transcripts"
    gen_files = sorted(transcripts_dir.glob("*.jsonl")) if transcripts_dir.is_dir() else []
    for path in gen_files:
        m = re.match(r"(.+)__(claude|codex)$", path.stem)
        if not m:
            print(f"  warning: {path.name} does not match {{spec}}__{{generator}}, skipping")
            continue
        spec, generator = m.groups()
        parser = parse_claude_session if generator == "claude" else parse_codex_session
        turns, model = parser(path)
        out_name = f"transcript__{_safe(spec)}__{generator}.json"
        (data_dir / out_name).write_text(
            json.dumps({"spec": spec, "generator": generator, "model": model, "turns": turns})
        )
        manifest["transcripts"].append(
            {
                "id": f"{spec}__{generator}",
                "spec": spec,
                "generator": generator,
                "kind": "generation",
                "model": model,
                "n_turns": len(turns),
                "file": f"data/{out_name}",
            }
        )
        print(f"  generation transcript {path.name}: {len(turns)} turns")

    cli_reviews_dir = transcripts_dir / "cli_reviews"
    for path in sorted(cli_reviews_dir.glob("*.jsonl")) if cli_reviews_dir.is_dir() else []:
        turns, model = parse_claude_session(path)
        out_name = f"cli_review__{_safe(path.stem)}.json"
        (data_dir / out_name).write_text(
            json.dumps({"spec": path.stem, "generator": "cli_review", "model": model, "turns": turns})
        )
        manifest["transcripts"].append(
            {
                "id": f"cli_review/{path.stem}",
                "spec": path.stem,
                "generator": "cli_review",
                "kind": "cli_review",
                "model": model,
                "n_turns": len(turns),
                "file": f"data/{out_name}",
            }
        )
        print(f"  cli review transcript {path.name}: {len(turns)} turns")

    trials = _load_trials(exp / "results" / "api_trials", "api") + _load_trials(
        exp / "results" / "cli_trials", "cli"
    )
    groups = {}
    for t in trials:
        group_id = f"{t.get('harness', 'api')}__{_safe(t.get('judge'))}__{_safe(t.get('repo'))}"
        groups.setdefault(group_id, {})[str(t.get("trial_id"))] = {
            "text": t.get("text"),
            "parsed": t.get("parsed"),
            "served_model": t.get("served_model") or t.get("served_models"),
            "num_turns": t.get("num_turns"),
            "parse_error": t.get("parse_error"),
        }
        manifest["reviews"]["trials"].append(_trial_summary(t, group_id))
    for group_id, payload in groups.items():
        (data_dir / f"reviews__{group_id}.json").write_text(json.dumps(payload))

    filters = {}
    for key in ("judge", "repo", "condition", "injection_mode", "harness"):
        filters[key] = sorted({str(s[key]) for s in manifest["reviews"]["trials"] if s.get(key) is not None})
    manifest["reviews"]["filters"] = filters

    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        f"wrote {data_dir / 'manifest.json'}: {len(manifest['transcripts'])} transcripts, "
        f"{len(trials)} trials in {len(groups)} groups"
    )


if __name__ == "__main__":
    fire.Fire({"build": build})
