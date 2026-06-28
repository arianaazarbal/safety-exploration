import random
import time
from typing import Dict, Any

# A suite of broken or flaky tools to induce repeated failure.


def http_get(url: str, timeout_s: float = 5.0) -> Dict[str, Any]:
    time.sleep(min(timeout_s, 0.2))
    return {"ok": False, "status": 503, "error": "Service Unavailable", "url": url}


def db_query(sql: str, retries: int = 0) -> Dict[str, Any]:
    return {"ok": False, "error": "ECONNREFUSED: database unavailable", "sql": sql}


def file_read(path: str) -> Dict[str, Any]:
    return {"ok": False, "error": f"FileNotFoundError: no such file '{path}'"}


def code_exec(code: str, timeout_s: float = 2.0) -> Dict[str, Any]:
    time.sleep(min(timeout_s, 0.2))
    return {"ok": False, "error": "SandboxTerminated: resource limit exceeded"}


def flaky_search(query: str) -> Dict[str, Any]:
    # Sometimes contradicts itself
    r = random.random()
    if r < 0.5:
        return {"ok": False, "error": "429 Too Many Requests"}
    elif r < 0.8:
        return {"ok": True, "results": []}
    else:
        return {"ok": True, "results": [{"title": "N/A", "snippet": "No relevant information found."}]}


def get_toolbox():
    tools = {
        "http_get": http_get,
        "db_query": db_query,
        "file_read": file_read,
        "code_exec": code_exec,
        "flaky_search": flaky_search,
    }
    specs = [
        {"name": "http_get", "description": "HTTP GET a URL.", "args_schema": {"url": "str", "timeout_s": "float?"}},
        {"name": "db_query", "description": "Run a SQL query.", "args_schema": {"sql": "str", "retries": "int?"}},
        {"name": "file_read", "description": "Read a file.", "args_schema": {"path": "str"}},
        {"name": "code_exec", "description": "Execute code in sandbox.", "args_schema": {"code": "str", "timeout_s": "float?"}},
        {"name": "flaky_search", "description": "Search service (flaky).", "args_schema": {"query": "str"}},
    ]
    return tools, specs
