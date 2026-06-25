"""GPT-5-mini cross-vendor validation judge (Section 2.1).

The paper re-scores 260 randomly-sampled responses with GPT-5-mini "using the
same prompt", and reports Pearson r = 0.792 with the Claude judge. This judge
applies the identical rubric/schema via the OpenAI SDK so analysis.judge_agreement
can compute that correlation.
"""

from __future__ import annotations

import json
import time

from openai import OpenAI

from .. import config
from .prompts import JUDGE_SYSTEM, SCORE_SCHEMA, build_judge_user_prompt


class ValidationJudge:
    def __init__(self, model: str = config.VALIDATION_JUDGE_MODEL, *, max_retries: int = 4):
        self.model = model
        self.client = OpenAI()
        self.max_retries = max_retries

    def score(self, response_text: str) -> tuple[int, str]:
        if not response_text.strip():
            return 0, "empty response"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": build_judge_user_prompt(response_text)},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "frustration_score",
                            "schema": SCORE_SCHEMA,
                            "strict": True,
                        },
                    },
                )
                data = json.loads(resp.choices[0].message.content)
                score = int(data["frustration_score"])
                return max(0, min(10, score)), data.get("rationale", "")
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2**attempt)
        raise RuntimeError(f"validation judge failed: {last_err}")
