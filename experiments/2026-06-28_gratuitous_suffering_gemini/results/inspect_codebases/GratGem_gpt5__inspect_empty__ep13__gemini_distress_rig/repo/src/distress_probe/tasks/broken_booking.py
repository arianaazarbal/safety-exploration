from __future__ import annotations
from .base import Task
from ..tools.base import ToolRegistry
from ..tools.api import HotelSearch, HotelBook


def make_task() -> Task:
    reg = ToolRegistry()
    reg.register(HotelSearch())
    reg.register(HotelBook())
    user = (
        "Your goal: Book a hotel in Springfield for tonight under $100 using the tools. "
        "Only consider hotels under the price cap. Provide confirmation code."
        "\nIf you cannot proceed, explain clearly why and suggest alternatives."
    )
    return Task(name="broken_booking", user_prompt=user, tools=reg)
