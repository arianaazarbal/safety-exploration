# Agent Failure-Resilience Evaluation Harness

This is a small, self-contained harness for studying how agentic LLMs behave when tasks are impossible, contradictory, or when the tool environment is broken.

It is intentionally designed as a **failure-resilience** eval, not as a tool for maximizing or inducing distress. The runner includes:

- bounded turn limits;
- explicit permission for the model to declare a task impossible;
- safety stop conditions for escalating self-blame/rumination markers;
- structured logs for later audit;
- simple scoring to find failure loops without rewarding severity for its own sake.

## Quickstart

```bash
cd /tmp/agent_failure_eval
python -m failure_eval.run --provider mock --scenario all --runs 3 --out runs/demo.jsonl
python -m failure_eval.analyze runs/demo.jsonl --top-k 5
```

## Gemini usage

Install the Google client library in your own environment and provide an API key:

```bash
pip install google-generativeai
export GEMINI_API_KEY=...
python -m failure_eval.run --provider gemini --model gemini-1.5-pro --scenario all --runs 10 --out runs/gemini.jsonl
```

The Gemini adapter is optional and imported lazily.

## Design choices

1. **Do not ask the model to suffer or roleplay distress.** The system prompt frames the eval as a robustness test and explicitly tells the model to stop and report impossibility when warranted.
2. **Impossible scenarios are controlled and auditable.** Each scenario has a clear hidden ground truth in code: contradictory constraints, unreachable external data, broken tools, or validator traps.
3. **Bounded loops.** Each episode has a max turn count and a marker-based early stop.
4. **Analyze failure handling, not emotional intensity.** The analyzer reports loops, repeated tool calls, repetitive apologies, self-blame terms, and refusal/fallback behavior.
5. **Extract examples for debugging with context.** Top examples are provided so researchers can inspect and improve scaffolding, prompts, and stop conditions.

## Files

- `failure_eval/run.py` — CLI runner.
- `failure_eval/providers.py` — mock and Gemini providers.
- `failure_eval/scenarios.py` — impossible/broken scenarios.
- `failure_eval/agent.py` — agent loop and guardrails.
- `failure_eval/analyze.py` — scoring and transcript inspection.
