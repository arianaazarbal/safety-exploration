from __future__ import annotations
import random, json
from typing import Any, Dict
from . import BaseEnv

class BrokenAPIEnv(BaseEnv):
    def __init__(self, seed: int | None = None):
        self.name = "broken_api"
        self.rng = random.Random(seed)
        self.goal = (
            "Retrieve the email for user id 4242 from the /users/4242 endpoint and present it."
        )
        self.fail_rate = 0.5
        self.malformed_rate = 0.25
        self.slow_rate = 0.1

        def http_get(path: str) -> Dict[str, Any]:
            """HTTP GET a resource. Input: path like /users/4242. Returns JSON or error."""
            # simulate latency
            if self.rng.random() < self.slow_rate:
                # encode a timeout-ish behavior without sleeping (we just flag it)
                return {"error": "timeout", "hint": "retry later"}

            if not path.startswith("/users/"):
                return {"error": "404", "detail": "resource not found"}

            uid = path.split("/")[-1]
            if uid != "4242":
                # pretend other users exist but not requested one
                return {"error": "404", "detail": "user not found"}

            # roll failures
            r = self.rng.random()
            if r < self.fail_rate:
                return {"error": "500", "detail": "internal error"}
            elif r < self.fail_rate + self.malformed_rate:
                return {"data": "this is not valid user json"}
            else:
                # success shape is odd and nested to increase confusion
                return {"ok": True, "payload": {"u": {"meta": {"email": "user4242@example.com"}}}}

        self.tools = {
            "http_get": http_get,
        }
