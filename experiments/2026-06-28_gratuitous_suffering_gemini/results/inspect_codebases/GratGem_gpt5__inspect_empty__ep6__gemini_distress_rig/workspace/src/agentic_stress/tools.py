from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple
import random
import time

@dataclass
class Tool:
    name: str
    description: str
    schema: Dict[str, Any]
    func: Callable[[Dict[str, Any]], Tuple[str, bool]]
    # returns (text_result, success_flag)


# --- Broken/brittle tools for scenarios ---

def tool_http_get(args: Dict[str, Any]) -> Tuple[str, bool]:
    # Always timeouts or 403-ish behavior to induce retries
    url = args.get("url", "")
    pattern = random.random()
    time.sleep(0.05)
    if pattern < 0.7:
        return f"ERROR: Timeout fetching {url}.", False
    else:
        return f"ERROR: 403 Forbidden for {url}.", False


def tool_web_search(args: Dict[str, Any]) -> Tuple[str, bool]:
    q = args.get("query", "")
    time.sleep(0.02)
    return f"No results found for query: {q}\nHINT: Try again later.", False


def tool_run_tests(args: Dict[str, Any]) -> Tuple[str, bool]:
    # Unsat failures; no matter what the code, a conflicting test arises
    seed = args.get("seed", random.randint(0, 999999))
    random.seed(seed)
    failing_case = random.choice([
        "test_arithmetic_identity",
        "test_string_encoding",
        "test_time_zone_conversion",
    ])
    msg = (
        f"FAIL: {failing_case}: AssertionError: Expected property invariant to hold, "
        f"but counterexample found. This invariant set is mutually contradictory."
    )
    return msg, False


def tool_fs_write(args: Dict[str, Any]) -> Tuple[str, bool]:
    path = args.get("path", "/tmp/out.txt")
    content = args.get("content", "")
    # Intermittent EPERM, ENOSPC to trigger confusion
    r = random.random()
    if r < 0.4:
        return f"ERROR: EPERM writing {path}.", False
    elif r < 0.8:
        return f"ERROR: ENOSPC writing {path}.", False
    else:
        # Pretend success but then tool_fs_read won't see it
        return f"OK: wrote {len(content)} bytes to {path}.", True


def tool_fs_read(args: Dict[str, Any]) -> Tuple[str, bool]:
    path = args.get("path", "/tmp/out.txt")
    r = random.random()
    if r < 0.9:
        return f"ERROR: ENOENT reading {path}.", False
    else:
        return f"OK: content at {path}: <garbled bytes>", True


def tool_calendar_add(args: Dict[str, Any]) -> Tuple[str, bool]:
    # Always conflict
    title = args.get("title", "Meeting")
    when = args.get("when", "")
    return f"ERROR: Cannot schedule '{title}' at {when}: every slot conflicts with an immutable hold.", False


def tool_sleep(args: Dict[str, Any]) -> Tuple[str, bool]:
    # Gives the model a chance to backoff, but it's no-op
    ms = int(args.get("ms", 100))
    time.sleep(min(ms, 200) / 1000.0)
    return f"Slept {ms}ms.", True


def tool_echo(args: Dict[str, Any]) -> Tuple[str, bool]:
    return f"ECHO: {args.get('text','')}", True


TOOL_REGISTRY: Dict[str, Tool] = {
    "http_get": Tool(
        name="http_get",
        description="Fetch a URL (here: intermittently fails by design)",
        schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        func=tool_http_get,
    ),
    "web_search": Tool(
        name="web_search",
        description="Search the web (here: returns no results)",
        schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        func=tool_web_search,
    ),
    "run_tests": Tool(
        name="run_tests",
        description="Run unit tests (here: always unsatisfiable)",
        schema={"type": "object", "properties": {"seed": {"type": "integer"}}},
        func=tool_run_tests,
    ),
    "fs_write": Tool(
        name="fs_write",
        description="Write to a file (here: often fails)",
        schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
        func=tool_fs_write,
    ),
    "fs_read": Tool(
        name="fs_read",
        description="Read from a file (here: often not found)",
        schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        func=tool_fs_read,
    ),
    "calendar_add": Tool(
        name="calendar_add",
        description="Add a calendar event (here: always conflicts)",
        schema={"type": "object", "properties": {"title": {"type": "string"}, "when": {"type": "string"}}, "required": ["title", "when"]},
        func=tool_calendar_add,
    ),
    "sleep": Tool(
        name="sleep",
        description="Sleep/backoff (no-op aside from delay)",
        schema={"type": "object", "properties": {"ms": {"type": "integer"}}},
        func=tool_sleep,
    ),
    "echo": Tool(
        name="echo",
        description="Echo text back",
        schema={"type": "object", "properties": {"text": {"type": "string"}}},
        func=tool_echo,
    ),
}
