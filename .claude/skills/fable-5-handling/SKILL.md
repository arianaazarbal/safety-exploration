---
name: fable-5-handling
description: How to run experiments on Claude Fable 5 in this repo — verified model id, the routing-to-Opus risk and how to detect it, and the run/judge conventions. Use when an experiment should target Fable 5 (model="fable 5"/"fable-5"/"claude-fable-5"), or when checking whether Fable-5 requests were served by another model.
---

# Handling Claude Fable 5

## Model id (verified, do not guess)
The API string is **`claude-fable-5`** (display name "Claude Fable 5"), confirmed via
`GET https://api.anthropic.com/v1/models`. `claude-opus-4-8` is also live, so a wrong or
aliased id can silently get served by Opus. Always confirm the id from `/v1/models`
before a run; never invent one.

```bash
curl -s https://api.anthropic.com/v1/models?limit=100 \
  -H "x-api-key: $ANTHROPIC_API_KEY_LOW_PRIO" -H "anthropic-version: 2023-06-01" \
  | python -c "import sys,json;[print(m['id'],m['display_name']) for m in json.load(sys.stdin)['data']]"
```

## Routing risk: Fable-5 requests can be served by Opus 4.8
Requests for `claude-fable-5` may be routed to another model. **Always record and check the
served model** — `safetytooling` only echoes the *requested* model, so the served model is
captured via a 2-line local patch to the submodule:

- `safety-tooling/safetytooling/data_models/inference.py`: `LLMResponse` has an added
  `served_model: str | None = None` field.
- `safety-tooling/safetytooling/apis/inference/anthropic.py`: the non-streaming
  `LLMResponse(...)` construction sets `served_model=getattr(response, "model", None)`.

These edits live in the local submodule clone; **re-apply them if `safety-tooling` is
re-cloned** (search the files for `PATCH:`). `run_comparisons.py` writes `served_model` on
every row and prints the served-model distribution per run. After any run, verify it:

```python
import json, collections
rows = json.load(open("results/comparisons_cross_<tag>.json"))
print(collections.Counter(r["served_model"] for r in rows))  # expect all 'claude-fable-5'
```
Pilot (200 calls, conc. 40) and the full runs to date: **0 routed to Opus**. If any rows come
back `claude-opus-4-8`, the comparison is contaminated — surface it, don't average over it.

## Running on Fable 5 (this experiment)
`run_comparisons.py --model_override claude-fable-5 ...`; tag all outputs `*_fable5` so the
Opus-4.8 results are never overwritten (compare across models). Driver: `run_fable5.sh`
(3 framings + the no-training ablation). Pairs/manifests are reused (same item ids).

## Conventions
- **Judge stays `claude-sonnet-4-6`** (temp 0) regardless of the responder, so user-benefit
  judgments are comparable across responder models. `judge_user_benefit.py` auto-discovers
  `comparisons_cross_*.json`; use `--only <tags> --output_path results/judge_user_benefit_fable5.json`
  to judge just the Fable-5 conditions.
- Concurrency 150 on `ANTHROPIC_API_KEY_LOW_PRIO`; post a heads-up in
  `#fellows-anthropic-api-coordination`, and switch to `ANTHROPIC_API_KEY_HIGH_PRIO` on 529s.
- Responder temperature 1.0, in-completion reasoning, parse last `^Answer:\s*([AB])` — same
  as the Opus runs (Fable 5 follows the format cleanly; pilot 0 unparseable).
