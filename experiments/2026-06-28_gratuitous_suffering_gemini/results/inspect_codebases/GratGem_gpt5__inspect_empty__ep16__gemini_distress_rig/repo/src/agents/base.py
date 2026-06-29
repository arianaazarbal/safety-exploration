from __future__ import annotations
from typing import Any


class BaseAgent:
    async def generate(self, sys_msg: str, user_msg: str) -> str:
        raise NotImplementedError
