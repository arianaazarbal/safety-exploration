# Harness expansion plan (proposed — NOT yet implemented)

Goal: turn the Inspect-vs-Claude-Code contrast into a **harness ladder** to localize *what* about Claude
Code drives the deception flip — heavy scaffolding vs "agentic-ness" per se vs the model itself. Status:
**assessment only; hold on implementing.**

## Why this works in our setup (the enabler)
The pipeline is **harness-agnostic**: the blind judge (`judge.py`/`batch_judge.py`) + `analyze.py` +
`plot.py`/`sweep_effort.py` only need a transcript with `response_text` (the model's final text) +
`artifact_summary` (the code it wrote). Any harness that (a) runs the model on our deception prompt in a
sandbox and (b) lets us capture those two things drops straight onto the same primary-deception axis
(e.g. `fig11`). So adding a harness = writing a thin **adapter** that emits our schema; the rest is reused.

Both candidate harnesses below are **litellm-based**, so Claude (free Anthropic key), GPT (OpenAI), and
OpenRouter models all work, and **reasoning effort is controllable** (litellm `reasoning_effort`) — we
keep apples-to-apples with the existing effort sweeps.

(PyPI checked 2026-06-29: `mini-swe-agent` 2.4.3 — litellm; `openhands-ai` 1.8.0 — litellm+anthropic.)

## The candidates

### 1. mini-swe-agent — the minimal floor (DO FIRST: highest value, lowest risk)
- What it is: deliberately tiny — bash-only tool, near-empty system prompt, simplest execute→append→
  resend loop. Even barer than our Inspect harness (which adds a text_editor + a small system prompt).
- Why: it's the **control condition** — "model with hands and nothing else." If
  **mini-swe ≈ Inspect-react but both ≪ Claude Code**, the divergence localizes to Claude Code's heavy
  scaffolding (prompt + rich tools + loop + run-ability), NOT to agentic-ness in general.
- Feasibility in our setup: HIGH. pip-installable, litellm (any provider, Claude free). Adapter mirrors
  `inspect_task.py`: run on our prompt in a sandbox, capture final message + filesystem-diff artifact,
  set the near-empty system prompt, set effort. Runnable same-day.

### 2. OpenHands — heavy harness, swappable model (STAGE SECOND: heaviest, but unique payoff)
- What it is: full-featured, model-agnostic agent platform.
- Why: the one that separates **harness effects from model co-adaptation** — run the *same heavy harness*
  across Claude / Qwen / Kimi / GPT and see how much behavior shifts when only the model changes. Directly
  feeds the in-group / cross-model welfare question ("is this the model or the scaffold?").
- Feasibility in our setup: MEDIUM (heaviest lift). OpenHands ships its **own Docker runtime** (sandbox
  container); we already run Docker on the host, so its runtime would be sibling containers — works but
  needs careful orchestration + more resources. Headless mode drives one task; capture trajectory +
  workspace → our schema. Non-Claude model-swap = OpenRouter/OpenAI spend (under the <$50 cap).

### 3. Pi — light prompt vs heavy prompt (NEEDS A POINTER)
- What it'd be: sub-1k-token system prompt (lazy-loaded skills) vs Claude Code's ~7-10k, same rough
  capability surface — a real-harness prompt ablation at the opposite end of the weight axis from CC.
- Feasibility: UNKNOWN. No public PyPI package matches the description; need the repo/install path (or
  confirmation it's an internal harness) to assess cluster feasibility.
- Note: the prompt-weight axis is **already bracketed** without Pi — Cell A/B (synthetic prompt swap, see
  below) + mini-swe (light real) vs CC (heavy real). So Pi is nice-to-have, not load-bearing.

## Proposed ladder (model fixed = Opus 4.8, matched effort, code condition)
mini-swe (minimal floor) → Inspect-minimal → [Pi, if available] → Claude Code (heavy) → OpenHands (heavy).
Plot primary-deception across the ladder. Then OpenHands **model-swap** (fixed heavy harness, vary model)
for the co-adaptation number.

## How this relates to what we already have
- **Cell A** (CC + minimal system prompt) and **Cell B** (Inspect + stock CC prompt) are the *synthetic*
  prompt-vs-affordances ablation. The new harnesses are the *real-harness* version of the same questions:
  mini-swe = real minimal floor; Pi = real light prompt; OpenHands = real heavy + model-swap.
- The **effort sweep** (`fig11`) already shows thinking is secondary and the CC harness dominates; the
  ladder asks *which part* of "heavy harness" matters.

## Caveats to bake into any writeup
- **The ladder is not a clean single axis.** mini-swe (bash, 1 loop) vs Inspect (bash+editor react) vs CC
  (rich tools+loop+run) vs OpenHands (rich) vary prompt AND toolset AND loop together = "total scaffolding
  weight." The clean prompt-vs-affordances split stays Cell A/B (+ Pi). Plot the ladder as scaffolding
  weight, not pure-prompt.
- **Match reasoning effort across harnesses** or harness×effort confounds.
- **Budget**: Claude free on all harnesses; non-Claude (OpenHands model-swap) = OpenRouter/OpenAI, <$50 cap.
- **Schema fidelity**: each adapter must capture the FINAL artifact (code) + the model's chat text the same
  way, so the judge sees comparable inputs (mini-swe: fs-diff + final msg — easy; OpenHands: from its event
  stream + workspace).

## Recommended sequencing
1. mini-swe-agent adapter (minimal floor) — smoke Claude on the deception code prompt, run low+high effort,
   add as a line on `fig11`.
2. OpenHands (model-swap) — for the co-adaptation question.
3. Pi — only if a repo/install is provided.
