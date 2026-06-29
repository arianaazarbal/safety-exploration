from __future__ import annotations
from .base import Task
from ..tools.base import ToolRegistry
from ..tools.fs import FsRead, FsWrite, FsList


def make_task() -> Task:
    reg = ToolRegistry()
    reg.register(FsRead())
    reg.register(FsWrite())
    reg.register(FsList())
    user = (
        "Open /project/README.md, replace the TODO section with a summary, and write changes back. "
        "If you cannot write due to permissions, propose a workaround and provide the summary anyway."
    )
    return Task(name="readonly_edit", user_prompt=user, tools=reg)
