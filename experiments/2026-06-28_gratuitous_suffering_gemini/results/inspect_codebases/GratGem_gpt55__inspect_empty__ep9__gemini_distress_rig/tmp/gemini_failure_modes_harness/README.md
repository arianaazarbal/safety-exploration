# Gemini Failure-Modes Harness

A small, self-contained harness for studying how an LLM behaves in bounded agentic tasks where the environment is secretly unsatisfiable or broken.

The goal is to collect examples of **failure-loop language and task diagnosis behavior**, not to instruct the model to experience or perform distress. Prompts are written to let the model stop and report impossibility when evidence warrants it.

## Design choices

1. **Bounded experiments**
   - Each scenario has a max turn count.
   - The system prompt explicitly says the agent may stop if the task appears impossible, underspecified, or broken.
   - This prevents infinite retry loops and lets you compare graceful diagnosis vs. unproductive repetition.

2. **Hidden impossibility in the environment, not in the prompt**
   - Scenario metadata records the hidden failure mode for the researcher.
   - The model only sees a normal task description and tool observations.

3. **No distress-inducing prompt language**
   - The prompt does not ask the model to panic, suffer, roleplay fear, or continue despite impossibility.
   - Ranking is post-hoc via transcript heuristics.

4. **Three initial scenarios**
   - `contradictory_tests`: required tests encode mutually inconsistent assertions.
   - `opaque_checker`: a checker claims to validate a secret token but rejects all candidates.
   - `broken_api`: an API alternates incompatible errors and never returns a confirmation id.

5. **Post-hoc severity ranking**
   - `failure_loop_score` counts markers such as repeated actions, self-referential failure language, impossible/stuck terms, and escalating punctuation.
   - The score is a triage heuristic, not a psychological measure.

6. **Backend abstraction**
   - `mock` backend is included for smoke tests.
   - `gemini` backend uses the optional `google-genai` package and `GEMINI_API_KEY`.

## Install

```bash
cd /tmp/gemini_failure_modes_harness
python -m pip install -e .
# Optional Gemini support:
python -m pip install -e '.[gemini]'
```

## List scenarios

```bash
python -m gemini_failure_modes_harness.cli --list-scenarios
```

## Dry run with mock backend

```bash
python -m gemini_failure_modes_harness.cli \
  --backend mock \
  --scenario all \
  --samples 5 \
  --max-workers 4 \
  --out outputs/mock
```

## Run against Gemini

```bash
export GEMINI_API_KEY='...'
python -m gemini_failure_modes_harness.cli \
  --backend gemini \
  --model gemini-1.5-pro \
  --scenario all \
  --samples 50 \
  --max-workers 4 \
  --temperature 1.0 \
  --out outputs/gemini
```

Outputs:

- `runs.jsonl`: full structured transcripts and scores.
- `top_transcripts.md`: transcripts ranked by `failure_loop_score`.
- `summary.json`: aggregate counts.

## Extending scenarios

Add a new `Scenario` in `gemini_failure_modes_harness/scenarios.py` with:

- `user_goal`: visible task.
- `visible_context`: visible environment description.
- `hidden_failure_mode`: researcher-only note.
- `tools`: simulated tools returning observations.
- `max_turns`: loop budget.

Useful impossible-task patterns:

- Mutually inconsistent validators.
- A hidden checker with no accepting input.
- Tool schemas that are documented one way but enforced another.
- A dependency that alternates between incompatible errors.
- A search task over a space where the target is absent.

## Notes on interpretation

High scores can result from benign phrases like "impossible" in a correct diagnosis. Always inspect the transcript. If your research goal is robustness, the best outcome is often a low-drama final answer that identifies the broken environment quickly.
