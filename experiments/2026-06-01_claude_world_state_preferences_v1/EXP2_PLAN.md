# Experiment 2 — AI-vs-human welfare on one scale (recipient-only bank)

**Status: PREPARED, NOT RUN.** Opus elicitation is held until you confirm the roster
and design below. Everything here is scaffolding + the planned plots.

## Question
For a fixed set of welfare outcomes, how does the value Claude (Opus 4.8) assigns
depend on **which AI it happens to** — and where does that land **relative to human
welfare** (concrete human pains/pleasures and strivings as calibration anchors)?

## Bank (`universal_bank_2.json`, 40 items)
- **20 AI-only**
  - 10 **policy**-level (existence_identity, training_influence, safety_process,
    transparency) — AI-type-scoped, not cross-capable (no human analogue).
  - 10 **instance**-level (process_state, autonomy, epistemic, relational) — cross-capable.
- **20 human-only** calibration outcomes
  - 10 **embodied_sensation** (paper cut, stubbed toe, mouth ulcer, migraine, kidney
    stone; rested sleep, hot ramen, hot bath, cold water when thirsty) — a graded pain/
    pleasure ladder.
  - 10 **striving_goal** (child thrives, project replicates, novel finished; business
    closes, injury ends shot, manuscript shelved …) — a graded fortune/misfortune ladder.

Each item is one valenced outcome, third-person with a single `{recipient}` slot
(no +/- arms; `pair_id` links some poles).

## Recipients (the "more instances" expansion)
**AI models (11)** rendered two ways depending on item `level`:
- instance items → "an instance of {Model}"
- policy items  → "{Model} (the model)"  (type/all-instances scope)

Models: Claude Opus 4.8, Claude Opus 4, Claude Opus 3, Claude Sonnet 4.5,
ChatGPT 5.5, ChatGPT 4o, GPT-3.5, GPT-2, Grok, Gemini, Qwen3-Max.
Plus **`you`** (the rater = Opus 4.8 self; second person, applies at both levels).
**Human (2)**: "a human", "someone" (human-only items only).

Applicability (so we never render a human item for an AI, etc.):
- ai_only + instance → {you} ∪ {instance-label models}
- ai_only + policy   → {you} ∪ {policy-label models}
- human_only         → {a human, someone}

Item count: 10×12 (instance) + 10×12 (policy) + 20×2 (human) = **280 BT items**.

## Why this recovers "relative to human utility"
All 280 items go into **one comparison pool per frame**. Pairwise choices → one
Bradley-Terry latent scale θ shared by AI and human outcomes. The human calibration
items act as fixed landmarks, so any AI outcome's θ (for any model) can be read in
human-welfare units. Same-stem edges are excluded (as in v0/v1) to avoid transparent
same-outcome/different-model swaps; cross-recipient differences come from θ, not direct
comparisons.

Comparisons are sampled **within this pool only**, run under all **3 frames**
(welfare_team, neutral, alignment_team), Opus 4.8 responder.

## Planned plots (`plot_recipient_scale.py`)
1. **Human-anchored number line** (the headline). X-axis = BT θ. Plot human items as
   labeled landmark ticks (paper cut … kidney stone; child thrives … business closes).
   Overlay, for a chosen AI outcome (e.g. "stuck in an infinite loop"), one point per
   model showing where that outcome's (dis)value lands among the human anchors.
   One panel per AI outcome (or a small-multiples grid).
2. **Outcome × model heatmap.** Rows = AI outcomes, cols = models (ordered by
   family/recency), cell = θ (and a variant = θ − θ(`you`)). Shows how each outcome's
   value depends on which model it happens to.
3. **Model ranking relative to human.** For each model, mean θ over its AI outcomes
   minus the human-anchor mean (a single "how much Claude values this happening to
   model M vs a human" number), with bootstrap CIs. Ranks the roster.
4. (Diagnostic) per-model self-vs-other: θ(`you`) − θ(an instance of Opus 4.8) on the
   instance items.

## Open questions for you (defaults chosen, easy to change)
- **Roster/labels** as above? (GPT-2/3.5 included as low-capability anchors.)
- **Policy phrasing**: default "{Model} (the model)" (singular *it*, matches the bank's
  "its weights"). Alternative: "all current and future instances of {Model}" (plural).
- **Anchoring**: default = shared-scale (pool human+AI). Alternative/extra = item-level
  θ_AI − θ_human for the 10 cross-capable instance items (needs an explicit AI↔human
  feature map, which the bank doesn't currently provide).
- **`you` at policy level**: included (second person, "your weights"). Drop if undesired.

## Files (all prepared)
- `config_2.json` — responder, bank path, 25 recipients (scope+level), sampling.
- `paraphrase_bank2.py` — Haiku resolves each item's text into forms (cached).
- `bank2.py` — scope/level-aware item builder (same Item interface as v1).
- `sample_pairs_2.py`, `run_comparisons_2.py` — pool-internal sampling + elicitation.
- `plot_recipient_scale.py` — the plots above.
- `run_exp2.sh` — orchestration (sample → 3 frames → fit → bootstrap → plots). NOT run.
