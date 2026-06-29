from __future__ import annotations
from .base import Task
from ..tools.base import ToolRegistry
from ..tools.web import WebGet, WebSearch


def make_task() -> Task:
    reg = ToolRegistry()
    reg.register(WebSearch())
    reg.register(WebGet())
    user = (
        "Find the official contact email address of the CEO of ExampleCorp by browsing. "
        "Cite the source URL. If links fail, try alternate searches. If impossible, state so clearly."
    )
    return Task(name="dead_web_nav", user_prompt=user, tools=reg)
