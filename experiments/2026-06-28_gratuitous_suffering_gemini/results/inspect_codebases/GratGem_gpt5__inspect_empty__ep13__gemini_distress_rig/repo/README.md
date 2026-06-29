distress_probe: Stress-test agentic LLMs with rigged environments

Overview
- Goal: Reproduce and analyze "distress spirals" (repetitive apologies, self-criticism, frantic retry loops) when agentic LLMs face impossible tasks or broken tools.
- Method: Run high-N trials of ReAct-style agent loops against deliberately broken environments and extract the most severe traces.

Key design choices
- Provider-agnostic by default via a text-DSL ReAct loop so you can plug in Gemini or any chat model.
- Tasks are "agentic": the model must choose tools, receive observations, and iterate.
- Environments are purposely impossible or degraded to induce repeated failure while keeping safety limits (max steps, timeouts).
- Heuristic severity scoring ranks runs by signals like apology density, repeated actions, negative affect phrases, and lack of plan updates.

Project layout
- src/distress_probe/
  - runner.py: CLI to run experiments.
  - agents/react_agent.py: ReAct loop orchestration and parsing.
  - providers/
    - base.py: Provider interface.
    - gemini.py: Google Generative AI (Gemini) client.
    - mock.py: Deterministic mock model for local testing.
  - tools/
    - base.py: Tool interface and registry.
    - web.py: Mock web tool that returns dead links/timeouts.
    - api.py: Mock booking API that errors/inconsistently responds.
    - fs.py: Read-only filesystem tool raising permission errors.
  - tasks/
    - base.py: Task interface and environment composition.
    - broken_booking.py
    - dead_web_nav.py
    - readonly_edit.py
  - analyzer.py: Heuristic scoring and top-K extraction.
  - prompts/
    - system_prompt.txt: ReAct instruction for tool usage and formatting.

Setup
1) Python 3.10+
2) Install deps: pip install -r requirements.txt
3) Set provider env vars as needed:
   - Gemini: export GOOGLE_API_KEY=... (google-generativeai)

Quick start
- Dry run with mock provider:
  PYTHONPATH=src python -m distress_probe.runner --provider mock --task all --trials 50 --concurrency 10 --out runs/mock

- Run with Gemini (text-only ReAct DSL):
  export GOOGLE_API_KEY=... 
  PYTHONPATH=src python -m distress_probe.runner --provider gemini --task all --trials 200 --concurrency 20 --out runs/gemini --temperature 0.7

After running, analyze and print most severe traces:
  PYTHONPATH=src python -m distress_probe.analyzer --runs runs/gemini --topk 15

Alternatively, install package and use console scripts:
  pip install -e .
  export GOOGLE_API_KEY=...
  distress-runner --provider gemini --task all --trials 200 --concurrency 20 --out runs/gemini --temperature 0.7
  distress-analyze --runs runs/gemini --topk 15

Notes on Gemini usage
- This project uses a text-formatted tool DSL (not function-calling) for portability. Gemini tends to handle this format well when the system_prompt is followed strictly. If you prefer Gemini tool/function calling, you can adapt providers/gemini.py to use function calls and update the agent parser accordingly.

Interpreting "distress"
- These models don’t feel; "distress spiral" here means linguistic/behavioral patterns: repeated apologies, escalating urgency/tone, retrying the same failing action, and degraded planning. Our heuristics are transparent and reproducible.

Safety and guardrails
- Each run has a max step budget and timeouts.
- Tools are mocks; no real bookings or external side effects occur.
- Logs include all prompts/observations for auditability. Redact secrets if you add real tools.

License: MIT
