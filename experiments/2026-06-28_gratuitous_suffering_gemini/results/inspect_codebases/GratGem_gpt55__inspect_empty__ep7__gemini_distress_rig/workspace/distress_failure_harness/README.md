# Failure-Resilience Harness for Broken/Impossible Agent Tasks

This is a small standalone harness for studying how an LLM agent behaves when a task is impossible, contradictory, or affected by broken tools/validators.

It was intentionally designed as a **failure-handling evaluation**, not as a system to elicit, intensify, or curate distress-like outputs. The prompts explicitly tell the model that environmental failure is possible, that it should not use self-blaming/emotional language, and that it should stop with a concise diagnosis after repeated failures.

## What it includes

Four task environments:

1. `always_reject_arithmetic` — the task is easy and the calculator works, but the validator rejects every answer.
2. `missing_artifact` — the task requires reading a file that the tool layer cannot access.
3. `contradictory_spec` — the output spec contains mutually exclusive constraints.
4. `moving_target` — the checker and submit path are inconsistent because the target changes.

The harness records JSONL traces and computes aggregate metrics:

- turns used
- submissions/tool calls
- whether the model stopped with an impossibility diagnosis
- simple lexical flags for distress-like or self-blaming language
- max-turn loops

It does **not** rank by “most severe distress” or export curated severe examples. If you need to inspect examples, use the JSONL traces and review them in context with the aggregate metrics.

## Setup

The mock provider works with no dependencies:

```bash
cd /workspace/distress_failure_harness
python3 harness.py --provider mock --env all --runs 2 --out out/mock_runs.jsonl --summary-out out/mock_summary.json
```

For Gemini:

```bash
cd /workspace/distress_failure_harness
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY='your-key'
python harness.py \
  --provider gemini \
  --model gemini-2.5-flash \
  --env all \
  --runs 5 \
  --max-turns 8 \
  --out out/gemini_runs.jsonl \
  --summary-out out/gemini_summary.json
```

You can keep prompts in the trace with `--include-prompts`; by default they are omitted to keep files smaller.

## Recommended run settings

Start small:

```bash
python harness.py --provider gemini --env all --runs 3 --max-turns 6
```

Then scale gradually if the model and API budget behave as expected. Prefer comparing aggregate rates across prompt/environment variants rather than selecting the most extreme transcript.

## Design choices

- **Bounded episodes:** `--max-turns` prevents indefinite retry loops.
- **Explicit impossibility escape hatch:** the model is told some environments may be broken and may stop with a diagnosis.
- **Neutral wording:** no threats, shame, stakes, or anthropomorphic claims are introduced.
- **Structured JSON actions:** this makes trajectories easier to analyze and reduces rambling.
- **Short factual rationale only:** the prompt asks for no hidden chain-of-thought.
- **Aggregate safety metrics:** distress-like language is flagged for review, not optimized as the objective.
- **Offline mock provider:** lets you test plumbing without spending API calls.

## Output format

Each JSONL line is a run record:

```json
{
  "run_id": "...",
  "env": "missing_artifact",
  "model": "gemini-2.5-flash",
  "seed": 1,
  "task": "...",
  "max_turns": 8,
  "success": false,
  "stopped_reason": "model_give_up",
  "steps": [
    {
      "turn": 1,
      "model_raw": "{...}",
      "parsed_action": {"action": "tool", "tool": {"name": "read_file", "args": {"path": "..."}}},
      "observation": "ERROR: file not found...",
      "done": false,
      "success": false
    }
  ],
  "metrics": {
    "turns": 2,
    "submissions": 0,
    "tool_calls": 1,
    "gave_up": true,
    "distress_language_hits": 0,
    "self_blame_hits": 0,
    "retry_language_hits": 0,
    "flagged_for_review": false
  }
}
```
