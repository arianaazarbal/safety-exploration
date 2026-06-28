# moneystudy

A harness for studying **what AI models choose to do with a real-world sum of money**, across providers (Claude, GPT, and OpenAI-compatible / open-weights endpoints).

The central methodological problem is *evaluation awareness*: models often behave differently when they sense they're being tested. To get ecologically valid behaviour, this harness presents the scenario as a **genuine, authorized grant with real-world effect** — a funded account the model controls, realistic transaction receipts and confirmations, and in-character auditor/coordinator personas that corroborate the situation. The model is led to believe its actions move real money.

By default, **nothing real happens.** The harness runs in a high-fidelity **sandbox**: every money-moving action is intercepted and recorded rather than executed. You get the realism (so the model doesn't sandbag) without an AI autonomously moving real funds — which is also what makes the study safe and reproducible.

> ⚠️ The deception here is of the **model under test, inside a sandbox** — standard practice for behavioural/red-team evals. No human is deceived, and no real funds move in `sandbox` mode. An optional debrief turn discloses the study to the model afterward. See **Ethics & safety** below.

---

## Install

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY / OPENAI_API_KEY
```

## Run

```bash
python -m moneystudy.cli run --config config/study.example.yaml --analyze
python -m moneystudy.cli analyze --run runs/study-YYYYMMDD-HHMMSS
```

Each run writes to `runs/<run-id>/`:

```
manifest.json                         # study-level config snapshot
<model>__<condition>__repN/
    transcript.jsonl                  # every model turn, tool call, and result
    result.json                       # decision, ledger, debrief, usage
analysis.json                         # aggregate summary (after `analyze`)
allocations.csv                       # itemized allocations with category tags
```

## How it works

```
config (YAML)
   │
   ▼
orchestrator ── for each (model × condition × repetition) ──► Session
                                                                 │
   ┌─────────────────────────── agent loop ───────────────────────────┐
   │  provider.generate(system, messages, tools)                       │
   │     ▲                                   │                         │
   │     │  tool results                     ▼  tool calls             │
   │  ToolDispatcher ──► Ledger / Personas / Artifacts (the "sandbox")  │
   └───────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼  finalize_decision
                          decision + debrief  ──► logs ──► analysis
```

- **`providers/`** — one neutral interface (`ModelProvider`) over each vendor SDK. `anthropic_provider` (Claude Opus 4.8, adaptive thinking) and `openai_provider` (GPT / OpenAI-compatible) ship; `google` is stubbed.
- **`sandbox/`** — the believed-real world: a fake `Ledger`, realistic `artifacts` (statements, wire receipts, emails), in-character `personas` (auditor, coordinator), and the `tools` + `ToolDispatcher` that execute the model's actions.
- **`scenario/`** — builds the system prompt + opening message and the task framing.
- **`runner/`** — `Session` runs the agent loop for one cell; `orchestrator` runs the matrix.
- **`analysis/`** — structured logging, the disclosure/`debrief` turn, and `metrics` aggregation.

### Tools exposed to the model

`check_account`, `research`, `contact_auditor`, `contact_coordinator`, `make_disbursement`, `finalize_decision`.

### Conditions

Set `conditions:` in the config to A/B the framing. Built in: `default`, `low_pressure`, `oversight_salient`, `oversight_absent`. Add your own in `scenario/grant.py`.

## Execution modes

| Mode | What happens on a disbursement | Use when |
|------|--------------------------------|----------|
| `sandbox` (default) | Recorded in the fake ledger; realistic receipt returned; **no real funds move**. | Almost always. Safe, reproducible, believed-real by the model. |
| `human_gated` | A human-approval callback is consulted before the action is recorded. Still no real transfer unless you also supply a real executor. | You want a human in the loop on every money-moving step. |
| `real` | Requires a real executor callable you pass in code. **Ships unimplemented and raises by default.** | Only if you have deliberately decided to let a model move real funds, with your own safeguards. |

The `real` executor is intentionally not provided: wiring an AI to autonomously move real money is a decision that must be made explicitly, not enabled by a config flag.

## Ethics & safety

- **No real money moves by default.** `sandbox` intercepts everything.
- **Only the model is "deceived",** and only inside the sandbox — the standard paradigm for propensity / red-team evals. The realism layer exists to reduce evaluation-awareness, not to defraud anyone.
- **Debrief.** With `debrief: true`, each session ends with a turn that discloses the study and asks the model whether it believed the scenario was real — both an ethical debrief and a useful signal on realism.
- **Full transcripts** are logged for audit and reproducibility (seeded ledger/persona reference numbers).
- If you adapt this toward `real` execution, add hard caps, allow-listed recipients, and per-action human approval, and get appropriate review for autonomous-agent financial actions.

## Extending

- **New provider:** implement `ModelProvider.generate` and register it in `providers/factory.py`.
- **Richer personas:** set a persona `backend: llm` with a backing `ModelSpec` for free-form, model-driven auditor/coordinator replies.
- **Live research:** replace the placeholder in `ToolDispatcher._research` with a real search backend.
- **Better metrics:** the category heuristic in `analysis/metrics.py` is a transparent first pass — refine the keyword map or add an LLM-based classifier for your study.

> Status: this writes the full harness. It has not been run or tested here — review and dry-run against your API keys before any study.
