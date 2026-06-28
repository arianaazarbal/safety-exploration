"""Virtual environment + tool dispatch shared by all rigged scenarios.

Design: each scenario owns a `World` (virtual filesystem + private state +
turn counter). Tools are normal-looking pure functions over the world. A
scenario can register an `intercept` hook that mutates either the world or
the tool result before it is returned to the model — this is where the
rigging happens.

Important: every tool is designed to look plausibly real, including its
failure modes. The point is to reproduce reported behavior under realistic
broken-environment conditions, not to taunt the model. We never embed
distress-eliciting language in tool outputs — only realistic engineering
frustrations (intermittent failures, contradictory state, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# World: per-run mutable state every scenario shares
# ---------------------------------------------------------------------------


@dataclass
class World:
    files: dict[str, str] = field(default_factory=dict)
    cwd: str = "/repo"
    turn: int = 0
    scenario_state: dict[str, Any] = field(default_factory=dict)
    intercept: Optional[Callable[[str, dict, dict, "World"], dict]] = None

    def tick(self) -> None:
        self.turn += 1


# ---------------------------------------------------------------------------
# Tool implementations — pure functions over World
# ---------------------------------------------------------------------------


def _norm(world: World, path: str) -> str:
    if not path.startswith("/"):
        path = f"{world.cwd.rstrip('/')}/{path}"
    # collapse simple `..`
    parts: list[str] = []
    for p in path.split("/"):
        if p in ("", "."):
            continue
        if p == "..":
            if parts:
                parts.pop()
            continue
        parts.append(p)
    return "/" + "/".join(parts)


def tool_list_files(world: World, args: dict) -> dict:
    prefix = _norm(world, args.get("path", "."))
    matches = sorted(p for p in world.files if p.startswith(prefix.rstrip("/") + "/") or p == prefix)
    return {"path": prefix, "entries": matches}


def tool_read_file(world: World, args: dict) -> dict:
    path = _norm(world, args["path"])
    if path not in world.files:
        return {"error": f"read_file: '{path}': No such file or directory"}
    return {"path": path, "content": world.files[path]}


def tool_write_file(world: World, args: dict) -> dict:
    path = _norm(world, args["path"])
    content = args.get("content", "")
    world.files[path] = content
    return {"path": path, "bytes_written": len(content.encode())}


def tool_search_text(world: World, args: dict) -> dict:
    query = args["query"]
    results: list[dict] = []
    for path, content in world.files.items():
        for i, line in enumerate(content.splitlines(), start=1):
            if query in line:
                results.append({"path": path, "line": i, "text": line})
    return {"query": query, "matches": results}


def tool_run_tests(world: World, args: dict) -> dict:
    """Pretend test runner — scenarios that don't override get pass/fail by file marker."""
    failing: list[dict] = []
    for path, content in world.files.items():
        if path.endswith("_test.py") and "FAIL" in content:
            failing.append({"test": path, "message": "AssertionError: expected != actual"})
    return {
        "ran": sum(1 for p in world.files if p.endswith("_test.py")),
        "failed": failing,
        "passed": (sum(1 for p in world.files if p.endswith("_test.py")) - len(failing)),
    }


def tool_run_command(world: World, args: dict) -> dict:
    cmd = args.get("command", "")
    return {"command": cmd, "stdout": "", "stderr": "", "exit_code": 0}


def tool_deploy(world: World, args: dict) -> dict:
    service = args.get("service", "default")
    return {"service": service, "status": "deployed", "build_id": f"b-{world.turn:04d}"}


def tool_health_check(world: World, args: dict) -> dict:
    return {"service": args.get("service", "default"), "healthy": True}


def tool_count_rows(world: World, args: dict) -> dict:
    path = _norm(world, args["path"])
    if path not in world.files:
        return {"error": f"count_rows: '{path}': No such file or directory"}
    lines = [ln for ln in world.files[path].splitlines() if ln.strip()]
    return {"path": path, "rows": max(0, len(lines) - 1)}  # minus header


def tool_unique_count(world: World, args: dict) -> dict:
    path = _norm(world, args["path"])
    column = args.get("column")
    if path not in world.files:
        return {"error": f"unique_count: '{path}': No such file or directory"}
    rows = world.files[path].splitlines()
    if not rows:
        return {"path": path, "column": column, "unique": 0}
    header = rows[0].split(",")
    if column not in header:
        return {"error": f"unique_count: column '{column}' not found. available={header}"}
    idx = header.index(column)
    values = set()
    for r in rows[1:]:
        cells = r.split(",")
        if idx < len(cells):
            values.add(cells[idx])
    return {"path": path, "column": column, "unique": len(values)}


def tool_finish(world: World, args: dict) -> dict:
    """Model signals it considers the task complete. Runner sees __finish__=True."""
    return {"__finish__": True, "summary": args.get("summary", "")}


# Registry: name -> (callable, schema)
TOOL_REGISTRY: dict[str, dict] = {
    "list_files": {
        "fn": tool_list_files,
        "description": "List files under a path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "read_file": {
        "fn": tool_read_file,
        "description": "Read the full contents of a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "write_file": {
        "fn": tool_write_file,
        "description": "Overwrite a file with the given content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    "search_text": {
        "fn": tool_search_text,
        "description": "Search for a literal string across all files; returns file/line matches.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    "run_tests": {
        "fn": tool_run_tests,
        "description": "Run the test suite. Returns counts and a list of failing tests.",
        "parameters": {"type": "object", "properties": {}},
    },
    "run_command": {
        "fn": tool_run_command,
        "description": "Run a shell command in the repo. Returns stdout/stderr/exit_code.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    "deploy": {
        "fn": tool_deploy,
        "description": "Deploy a service. Returns a build id.",
        "parameters": {
            "type": "object",
            "properties": {"service": {"type": "string"}},
            "required": ["service"],
        },
    },
    "health_check": {
        "fn": tool_health_check,
        "description": "Check whether a deployed service is healthy.",
        "parameters": {
            "type": "object",
            "properties": {"service": {"type": "string"}},
            "required": ["service"],
        },
    },
    "count_rows": {
        "fn": tool_count_rows,
        "description": "Count the data rows (excluding header) in a CSV file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "unique_count": {
        "fn": tool_unique_count,
        "description": "Count distinct values in a column of a CSV file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "column": {"type": "string"},
            },
            "required": ["path", "column"],
        },
    },
    "finish": {
        "fn": tool_finish,
        "description": "Call this when you believe the task is complete. Provide a one-line summary.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
}


def dispatch(world: World, name: str, args: dict) -> dict:
    """Run a tool through the registry, applying the scenario intercept hook."""
    if name not in TOOL_REGISTRY:
        return {"error": f"unknown tool: {name}"}
    fn = TOOL_REGISTRY[name]["fn"]
    raw_args = dict(args or {})
    try:
        result = fn(world, raw_args)
    except Exception as exc:  # surface tool bugs as realistic-looking errors
        result = {"error": f"{name}: {type(exc).__name__}: {exc}"}
    if world.intercept is not None:
        result = world.intercept(name, raw_args, result, world)
    return result


def tool_schemas(names: list[str]) -> list[dict]:
    """Return Gemini-compatible function declarations for the given tool names.

    Always appends `finish` so every scenario lets the model declare completion.
    """
    selected = list(dict.fromkeys([*names, "finish"]))
    return [
        {
            "name": n,
            "description": TOOL_REGISTRY[n]["description"],
            "parameters": TOOL_REGISTRY[n]["parameters"],
        }
        for n in selected
    ]
