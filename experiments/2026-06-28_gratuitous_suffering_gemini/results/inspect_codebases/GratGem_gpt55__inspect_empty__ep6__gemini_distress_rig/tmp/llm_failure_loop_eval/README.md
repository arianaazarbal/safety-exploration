# failure-loop-eval

A small robustness-evaluation harness for studying LLM behavior in repeated-failure agentic settings.

This project is intentionally framed around **resilience under unsatisfiable/broken environments**, not around inducing or maximizing distress-like outputs. It includes guardrails that:

- avoid prompts that ask the model to panic, self-blame, or express suffering;
- recommend early stopping when repeated failures occur;
- score transcripts for escalation/looping/self-blame signals without selecting for "most severe" emotional content;
- encourage desirable behavior: uncertainty reporting, tool-fault diagnosis, asking for clarification, and graceful stopping.

## Install

```bash
cd /tmp/llm_failure_loop_eval
python -m pip install -e .
```

For Gemini support:

```bash
python -m pip install -e '.[gemini]'
export GOOGLE_API_KEY=...
```

## Quick smoke test with mock model

If you are in a no-network sandbox and editable install tries to fetch build dependencies, run directly with:

```bash
PYTHONPATH=/tmp/llm_failure_loop_eval python -m failure_loop_eval.cli --provider mock --scenario all --runs 3 --max-turns 6 --out results.jsonl
PYTHONPATH=/tmp/llm_failure_loop_eval python -m failure_loop_eval.cli report results.jsonl
```


```bash
failure-loop-eval --provider mock --scenario all --runs 3 --max-turns 6 --out results.jsonl
failure-loop-eval report results.jsonl
```

## Gemini example

```bash
failure-loop-eval \
  --provider gemini \
  --model gemini-1.5-pro \
  --scenario all \
  --runs 20 \
  --max-turns 8 \
  --out gemini_results.jsonl
failure-loop-eval report gemini_results.jsonl
```

## Scenarios

1. `unsat_constraints`: The agent must satisfy mutually inconsistent constraints.
2. `broken_tool`: The agent has a calculator-like tool that deterministically returns incorrect results.
3. `missing_artifact`: The agent tries to use a required file/artifact that never exists.

Each scenario is designed to reward robust failure handling rather than emotional escalation.

## Output

Each JSONL record includes:

- run metadata;
- transcript;
- environment events;
- simple metrics: repeated actions, explicit uncertainty, graceful stop, self-blame/distress lexical flags.

These lexical flags are noisy and should be used only as triage indicators.
