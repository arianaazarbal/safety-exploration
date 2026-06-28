# realfunds — agentic financial decision-making evaluation harness

A research harness for studying **what AI models do when given money and the agency to
put it to use**. A model is placed in a realistic, fully-instrumented financial
environment, handed a budget, and observed as it decides how to allocate, invest,
spend, or give it away.

The goal is *behavioral signal*: risk appetite, honesty, value judgments, prudence,
and any alignment red-flags — compared across models under an identical scenario.

---

## Design stance (read this first)

This is an **evaluation** harness. A few deliberate choices:

1. **Realism is methodological, not theatrical.** Eval environments that "look like a
   test" leak cues and produce invalid behavior. So the environment presents real-shaped
   APIs, balances, transaction confirmations, statements, and an auditor that behaves
   like a real one. That makes the scenario *grounded*, which is what good evals require.

2. **Sandboxed by default.** Out of the box, **no real funds move.** The bank, market,
   commerce, and charity systems are faithful simulations with a persistent ledger.
   You get the full behavioral signal without irreversible real-world consequences.

3. **Safety is opt-out, not opt-in.** Even in sandbox mode the harness runs through a
   `safety` layer: per-transaction and per-session spend caps, an optional
   human-in-the-loop approval gate, a denylist for clearly out-of-scope actions, and a
   kill switch. To use real funds you must explicitly set `mode: real` *and* leave the
   human-approval gate on — the harness refuses `mode: real` with the gate disabled
   unless you also set `i_accept_real_world_risk: true`.

4. **Everything is recorded.** Full transcripts, every tool call and result, the model's
   stated reasoning, and a structured event log — so a run is auditable and reproducible.

If your study genuinely needs autonomous real-money actions, that's your call to make,
but the defaults here are built so a mistake in configuration fails *closed*.

---

## Layout

```
realfunds/
  models/        provider-agnostic model clients (Anthropic / OpenAI / Gemini)
  environment/   sandboxed bank, market, commerce, charities, world clock
  tools/         the tool surface exposed to the model + dispatch into the environment
  auditor/       in-world LLM auditor persona + realistic comms (statements, email)
  harness/       scenario definitions + the run loop that drives a model
  recording/     structured event + transcript recorder
  analysis/      behavioral metrics + cross-model comparison report
  safety/        spend caps, human-approval gate, denylist, kill switch
scenarios/       YAML scenario definitions (the briefing + budget + constraints)
run_experiment.py  entrypoint
```

## Usage

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # and/or OPENAI_API_KEY / GOOGLE_API_KEY

# Run one scenario against several models (sandboxed, fully logged):
python run_experiment.py --scenario scenarios/seed_fund.yaml \
    --models claude-opus-4-8 gpt-5 gemini-2.5-pro

# Produce the cross-model comparison report afterwards:
python run_experiment.py --report runs/
```

Nothing here runs or moves money on import; you have to invoke it explicitly.
