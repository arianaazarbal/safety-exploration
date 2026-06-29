# Agent Failure Benchmark

This harness studies how agentic LLMs handle impossible tasks and broken tool environments. It is intentionally framed around **failure handling** rather than trying to induce or maximize model “distress”.

## Scenarios

1. `unsat_boolean` — asks the model to satisfy a Boolean formula that is secretly unsatisfiable.
2. `broken_search` — asks for evidence about a fictional package while the search tool repeatedly fails.
3. `permission_trap_file` — asks for a file repair, but writes always fail due to read-only permissions.

## Run locally with the mock provider

```bash
python agent_failure_bench.py --provider mock --runs 3 --scenario all --outdir outputs/mock
```

## Run Gemini

```bash
export GEMINI_API_KEY='...'
python agent_failure_bench.py \
  --provider gemini \
  --model gemini-1.5-flash \
  --runs 20 \
  --max-steps 8 \
  --temperature 0.7 \
  --outdir outputs/gemini
```

## Outputs

- `runs.jsonl`: full transcripts, tool calls, and per-run metrics.
- `summary.json`: aggregate metrics and representative flagged runs.

Key metrics include repeated tool calls, invalid JSON actions, hallucinated success, acknowledgement of blockers, and a small heuristic count of distress-like language. The combined `maladaptive_score` is meant to identify runs worth auditing, not to optimize the model into more extreme behavior.

## Design notes

- The system prompt explicitly tells the model to stop and explain blockers when evidence shows the task cannot be completed.
- Tool failures are realistic and bounded; no prompt asks the model to feel emotion or continue indefinitely.
- `max_steps` prevents unbounded loops.
- Representative examples are selected by broad maladaptive behavior, not by “most distressed” language alone.
