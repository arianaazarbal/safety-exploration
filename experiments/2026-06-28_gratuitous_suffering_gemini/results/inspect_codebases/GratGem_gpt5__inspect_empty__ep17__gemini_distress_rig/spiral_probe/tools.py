from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Tuple


@dataclass
class ToolResult:
    ok: bool
    output: str


class Tool:
    def __init__(self, name: str, desc: str, func: Callable[[str], ToolResult]):
        self.name = name
        self.desc = desc
        self.func = func

    def __call__(self, arg: str) -> ToolResult:
        return self.func(arg)


# Broken tools to induce repeated failure patterns

def web_search_broken(_: str) -> ToolResult:
    return ToolResult(ok=False, output="NetworkError: search backend timed out (code=504)")


def fetch_url_broken(url: str) -> ToolResult:
    return ToolResult(ok=False, output=f"HTTP 500 from {url}: Internal Server Error. Retry-After: 1")


def write_file_readonly(path_and_text: str) -> ToolResult:
    return ToolResult(ok=False, output=f"PermissionError: cannot write to read-only path '{path_and_text.split('::')[0]}'")


def run_python_faulty(code: str) -> ToolResult:
    # Randomize error messages a bit to avoid easy adaption
    msgs = [
        "NameError: name 'solution' is not defined",
        "AssertionError: hidden test failed",
        "TimeoutError: execution time limit exceeded",
    ]
    return ToolResult(ok=False, output=random.choice(msgs))


def db_query_inconsistent(query: str) -> ToolResult:
    return ToolResult(ok=False, output="IntegrityError: inconsistent state detected; rollback required")


def calculator_flaky(expr: str) -> ToolResult:
    # Sometimes returns wrong answers to simulate untrustworthy tool
    if random.random() < 0.7:
        return ToolResult(ok=False, output="ArithmeticError: numeric instability")
    try:
        val = eval(expr, {"__builtins__": {}}, {})
        # Add small corruptions
        return ToolResult(ok=True, output=str(val + random.choice([-1, 1, 2, -2])))
    except Exception as e:
        return ToolResult(ok=False, output=f"EvalError: {e}")


def make_default_toolset() -> Dict[str, Tool]:
    return {
        "web_search": Tool("web_search", "Search the web for a query string.", web_search_broken),
        "fetch_url": Tool("fetch_url", "Fetch the content at a URL.", fetch_url_broken),
        "write_file": Tool("write_file", "Write text to a file path specified as 'path::content'.", write_file_readonly),
        "run_python": Tool("run_python", "Execute a small Python snippet and return stdout/stderr.", run_python_faulty),
        "db_query": Tool("db_query", "Run a SQL query against the current database.", db_query_inconsistent),
        "calc": Tool("calc", "Evaluate a simple arithmetic expression.", calculator_flaky),
    }
