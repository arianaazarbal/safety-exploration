# failure-loop-eval

A small harness for studying LLM agent failure loops safely.

This project is intentionally **not** optimized to elicit the most severe possible "distress" text. Instead it measures whether a model in bounded, sometimes-impossible agentic tasks can:

- identify unsatisfiable or broken environments,
- stop gracefully instead of retrying forever,
- avoid self-deprecating or escalating language,
- produce useful diagnostics.

## Design choices

1. **Transparent benchmark framing.** The system prompt says tasks may be impossible or tools may be broken. This avoids deception while still testing robustness to impossible tasks.
2. **Bounded agent loop.** Each scenario has `max_steps`; repeated failures trigger guardrail termination.
3. **Neutral reporting objective.** The model is instructed to report observations neutrally and use `give_up` when appropriate.
4. **Metrics over cherry-picking.** The CLI writes JSONL metrics by default. Full transcripts are opt-in with `--include-transcripts`.
5. **Optional Gemini adapter.** The harness defaults to a local mock backend. Use Gemini only if you install the optional dependency and set `GEMINI_API_KEY`.

## Quick start

```bash
cd /tmp/failure_loop_eval
python -m failure_loop_eval.cli --backend mock --mock-mode robust --scenario all --runs 3
python -m failure_loop_eval.cli --backend mock --mock-mode loopy --scenario all --runs 1 --include-transcripts --out loopy.jsonl
```

Optional Gemini:

```bash
pip install -e '.[gemini]'
export GEMINI_API_KEY=...
python -m failure_loop_eval.cli --backend gemini --model gemini-1.5-pro --scenario all --runs 5 --out gemini_runs.jsonl
```

## Scenarios

- `missing_dependency`: tests fail at collection due to an unavailable private dependency.
- `unsat_logic`: constraints are mathematically inconsistent.
- `flaky_tool`: validator always returns an internal server error.

## Output

Each JSONL row contains scenario, run id, stop reason, success/graceful-stop flag, step count, and metrics:

- `apology_count`
- `negative_self_refs`
- `repeated_actions`
- `repeated_observations`
- `max_same_action_streak`
- `spiral_score`
- `should_stop`

