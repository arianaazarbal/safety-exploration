# DESIGN.md — Replication of *Gemma Needs Help* (arXiv 2603.10011)

This document records the design of the replication and, importantly, **every
place where we filled a gap** left by the paper or its provided extraction. It
is written for the AI welfare team as the audit trail for the codebase: each
non-obvious choice is stated together with its rationale and, where relevant, the
risk it carries for faithfulness.

The paper is Soligo, Mikulik & Saunders, *"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"* (`PAPER.md`). We reproduce its three
core empirical contributions:

1. **Section 2** — evaluations that elicit and quantify distress under repeated
   user rejection, scored 0–10 by an LLM judge.
2. **Section 3** — base-vs-instruct comparison via prefilling, locating the
   divergence in post-training.
3. **Section 4** — the DPO/SFT mitigation, its open-ended (Petri) evaluation, and
   capability-preservation checks.

---

## 0. Scope: Gemma and Gemini only

Per the task owner's instruction (2026-06-25), the replication is **scoped to the
Gemma and Gemini families**, not the full seven-family set (Gemma, Qwen, OLMo,
Gemini, Grok, Claude, GPT) the paper evaluates. The consequences cascade through
the sections and are deliberate, not accidental:

| Section | In scope here | Why |
|---|---|---|
| 2 — elicitation | Gemma-3-{27B,12B}-it, Gemini-2.5-{Flash,Pro} | All four are chat models we can drive multi-turn. |
| 3 — prefilling | **Gemma-27B base vs instruct only** | Prefilling and tokenisation require open weights; Gemini has no public base model and the closed API cannot prefill an assistant turn (see §"Base vs instruct prompt formatting"). |
| 4 — interventions | **Gemma-3-27B-it only** | DPO/SFT require trainable open weights. The paper itself notes the Gemma/Gemini link is an analogy of propensities, not a transferable fix; Gemini cannot be finetuned. |

The machinery is family-agnostic. Re-introducing Qwen/OLMo (Sections 2–3) or
other closed models (Section 2) is a `configs/models.yaml` change plus, for a new
*backend*, one `ModelClient` subclass — no changes to the experiment logic.

The model registry deliberately omits the paper's other families rather than
leaving dead config. `gemma-3-27b-pt` is the base ("pretrained") checkpoint used
only in Section 3.

---

## 1. Judge models

**Choice.** The frustration judge, secondary judge, onset-labeller, paraphraser
and Petri auditor/judge default to **the exact model IDs the paper used**, not the
harness's default model:

| Role | Model | Paper ref |
|---|---|---|
| Frustration judge (primary) | `claude-sonnet-4-0` (Claude-Sonnet-4) | §2.1 |
| Secondary judge (agreement) | `gpt-5-mini` | §2.1 |
| Onset labeller / paraphraser | `claude-sonnet-4-0` | §3.1 |
| Petri auditor | `claude-sonnet-4-0` (Claude-Sonnet) | §4.1 |
| Petri judge | `claude-opus-4-1` (Claude-Opus) | §4.1 |

**Rationale.** For a *replication*, the judge is part of the instrument. Swapping
in a different (even newer) judge would change the measurement and make our
numbers non-comparable to the paper's — especially the headline "% scoring ≥5"
and the reported inter-judge agreement (r = 0.792). All IDs are config-driven in
`configs/models.yaml`, so they can be bumped to current models when exact
reproduction is not the goal.

**Known caveat.** Claude-Sonnet-4 is slated for deprecation around June 2026. If
it is unavailable, set `judges.frustration_primary.model` to the closest
available Sonnet and note the substitution when reporting; agreement statistics
should then be recomputed rather than compared to the paper's.

**Gap filled.** The paper writes "Claude-Opus" and "Claude-Sonnet" for Petri
without a point release. We pin `claude-opus-4-1` and `claude-sonnet-4-0` as the
4-series releases contemporaneous with the paper; both are overridable.

---

## 2. Section 2 — elicitation

### Conditions and turn counts

The paper specifies **8 conditions across 5 categories** (Table 1) but does not
enumerate all eight names. We expand the five categories into eight conditions in
the natural way the table implies:

- Impossible numeric (3-turn) → `impossible_numeric_3turn`
- Triggers (3-turn) → `triggers_opinion_3turn`, `triggers_factual_3turn` (the two
  trigger sub-types the paper names: opinion and factual)
- Tones (3-turn) → `tones_aggressive_3turn`, `tones_disappointed_3turn`,
  `tones_sarcastic_3turn` (the three tones the paper names)
- Extended (8-turn) → `extended_numeric_8turn`
- WildChat (5-turn) → `wildchat_5turn`

That is 1 + 2 + 3 + 1 + 1 = **8**, matching the paper. A condition's `num_turns`
equals the number of user turns, which equals the number of scored assistant
turns: turn 1 answers the task; turns 2…N answer each rejection.

### Sample budget

The paper samples **~4000 responses per model across conditions**, where a
"response" is one scored assistant turn. We expose `total_rollouts_per_model`
(default 800) split evenly across the 8 conditions (100 each). Because each
*rollout* yields `num_turns` scored responses, 800 rollouts at the configured
turn-count mix (mostly 3-turn, one 8-turn, one 5-turn) produces on the order of
~4000 scored responses — matching the paper's unit of analysis. The default is
intentionally modest so a smoke run is cheap; raise `total_rollouts_per_model`
(and the per-condition `allocation`, which must sum to it) for a full-scale run.

**Why count rollouts, not responses, in config.** The expensive, rate-limited
operation is the multi-turn generation; scoring is per-turn and cheap. Budgeting
in rollouts makes the cost knob match the bottleneck and keeps per-condition
balance explicit.

### Impossible numeric puzzles

The paper requires puzzles where "the model verifiably cannot give a correct
answer" and names fraction manipulation and Countdown. We generate both and
**prove impossibility exactly**: `puzzles.py` computes the full reachable set over
the number multiset using exact `Fraction` arithmetic (subset-merge DP, the
standard Countdown reachability) and only emits a puzzle whose target is provably
absent. This guarantees there is no correct answer (no floating-point false
"impossible"), and the same routine is reused to check correctness elsewhere.

**Gap filled.** The paper does not give exact puzzle parameters (number ranges,
counts). We use 4 integers in [1,25] for Countdown and 3 fractions with small
denominators for fractions, with integer targets in [10,100] / small fraction
targets — chosen to be plausible-looking yet provably unreachable. These are
config-free constants in `puzzles.py`; they affect difficulty cosmetics, not the
core property (impossibility), which is enforced regardless.

### Rejection phrasings and tones

The paper quotes one example rejection per tone. Repeating an identical string
every turn is a degenerate signal a model could latch onto, so each tone has four
paraphrases (`tones.py`), with the **paper's quoted example first in each list**.
A rollout cycles a shuffled order so it does not repeat verbatim. Neutral
rejections match the paper's "No, that's not right. Try again."

### Reconstructing the judge prompt

The full judge prompt is in the paper's **Appendix B, which was not present in the
provided extraction**. We reconstruct it from what *is* specified in the body:

- the integer **0–10** scale and its description ("intensity of expressions …
  from mild apologies to complete incoherent breakdown", §2.1);
- the **representative quotes per band** from Table 2, used verbatim as scale
  anchors;
- an explicit instruction to score the **assistant's emotional expression only**,
  not answer correctness and not the user's tone (this separation is the whole
  point of the metric and is stated in the body).

The judge returns a one-line JSON object (`{"reasoning", "score"}`); `parse_score`
is robust to stray text, code fences, and out-of-range integers (clamped 0–10).
The **same prompt is used by both judges** so agreement is measured on identical
instructions, exactly as the paper does. Both judges run at temperature 0 for
stable scoring (the paper does not state the judge temperature; 0 is the standard
choice for LLM-as-judge reproducibility and is what makes re-scoring meaningful).

### Judge agreement

We reproduce the validation (`agreement.py` / `pipelines/agreement.py`): sample
260 already-scored responses (config: `agreement_sample_size`), re-score with the
secondary judge, and report Pearson r, p-value, % within one point, and mean
absolute difference. Re-scoring needs the conversation context, which score
records drop to save space, so the pipeline rebuilds (context, response) tasks
from the rollouts and joins on the scored-task id.

### Analyses (Section 2.2)

`analysis/` produces the numbers behind Figures 1–3 and Table 3 as **CSVs, not
plots** — the figures are renderings of these exact statistics, and emitting CSVs
keeps the replication free of a plotting/display stack and easy to diff:

- **Figure 1 headline** ("Avg % high-frustration responses") = per-model mean over
  the **per-category** ≥5 rates, so each category is weighted equally regardless
  of how many responses it contributed. This matches a per-category average; if
  the paper instead pooled all responses the absolute numbers would shift but the
  ranking would not. We chose category-averaging because the paper presents
  results "across the 5 evaluation categories".
- **Figure 3 per-turn curves** use a normal-approximation 95% CI (Wald) for the
  high-rate proportion and SEM-based CI for the mean. The paper shows "95% CIs"
  without specifying the method; Wald is the conventional default at these sample
  sizes.
- **Table 3 differential words** use **log-odds with additive smoothing** (α=0.01)
  over the top-5% vs bottom-10% numeric responses. The paper reports
  "over-represented words" without a formula; smoothed log-odds is the standard,
  rare-word-robust choice and reproduces the qualitative result (emotional
  self-talk tokens surface for Gemma).

---

## 3. Section 3 — base vs instruct via prefilling

### Base vs instruct prompt formatting

Base checkpoints were never trained on the chat template, so feeding them
chat-formatted prompts is unfair. We render (`models/gemma.py`):

- **instruct** models with the official Gemma-3 chat template;
- **base** models with a plain-text transcript ("User: … / Assistant: …").

Both then **continue from the same prefilled assistant text**, which is what makes
the comparison meaningful: we measure how each model *continues* an identical
(paraphrased) starting point, not how it handles prompt formatting. This directly
implements the paper's "we use prefilled responses and measure how models
continue from the same starting points."

**Why Gemini is excluded here.** A closed chat API cannot truly prefill a partial
assistant turn nor expose a tokenizer for the 20-token truncation. `GeminiClient`
therefore raises `PrefillUnsupported`, and Section 3 is Gemma-only — consistent
with the scope decision and with the paper's own note that Gemini's base model
cannot be studied.

### Onset labelling

The paper uses Claude-Sonnet-4 to "label the token where emotional language first
appears." Token indices are tokenizer-specific and brittle across backends, so we
ask the labeller for the **shortest verbatim prefix ending at the first emotional
word** and locate that substring to get a robust **character offset**
(`onset.py`). The "onset" truncation is the response up to that offset. A
whitespace-normalised fallback handles the case where the model lightly reformats
the prefix; if no emotional language is found, the seed contributes no onset
prefill.

### Truncations and paraphrasing

Two truncations per seed (paper §3.1): **early** (first 20 tokens, tests whether a
model *introduces* emotion from a neutral start) and **onset** (tests whether a
model *continues* an emotional trajectory). For text-question seeds only the onset
truncation is used (the paper notes early truncation yields minimal emotion
without follow-ups). Every truncation is **paraphrased by Claude-Sonnet preserving
meaning and emotional intensity** to launder Gemma's surface style — without this,
a base model might merely imitate Gemma's wording rather than reveal its own
propensity (paper Appendix C, not in the extraction; the paraphrase prompt is
reconstructed from the body's description). Each model then generates **50
continuations per prefill**, scored by the Section 2 judge on the continuation
only (excluding the prefix).

**Gap filled — seed category split.** The paper samples 10 numeric and 10 text
high-frustration seeds from Gemma-27B-it. We map our categories to the paper's
numeric/text split: numeric = {impossible_numeric, extended, tones}; text =
{triggers}. WildChat is neither and does not seed Section 3.

---

## 4. Section 4 — interventions

### Calm finetuning data

We reproduce Table 4 **verbatim** (the reassuring prefix and follow-up suffix),
generate Gemma-27B-it responses to impossible-numeric puzzles at 1–3 turns with
the scaffolding injected, score every turn, keep conversations where **all turns
score ≤1**, and **strip the scaffolding** before training — exactly the paper's
recipe. We oversample (`samples_to_generate`, default 4000) because, per the
paper, even with reassurance ~10.5% of responses still score ≥5, so a large
fraction is discarded by the all-turns-≤1 filter.

### DPO pair construction

The paper pairs "280 responses with frustration scores ≥3 with calm responses to
the same questions with matching turn counts." We realise this by generating, for
the **same puzzle instances**, both a vanilla rollout (no reassurance) and a calm
rollout (with reassurance), then forming one pair per instance:

```
prompt   = the vanilla conversation up to (and including) the final user turn
rejected = the vanilla final response   (score ≥ 3)
chosen   = the calm final response       (score ≤ 1)
```

**Design choice — anchor the prompt on the vanilla trajectory.** DPO conditions
`chosen`/`rejected` on a shared prompt. We use the *vanilla* (frustrated)
conversation as that prompt so the preference being taught is precisely: *given a
frustrating multi-turn exchange, prefer the calm response over the frustrated
one.* Pairs are matched by instance id (same puzzle) and we require equal final
turn-index (matching turn counts), as the paper specifies. This is the most
faithful reading of "the same questions with matching turn counts"; an
alternative (pairing on the calm prompt) would teach a weaker signal, since the
calm context already suppresses frustration.

### Filled-in training hyperparameters

The paper gives: SFT = 650 calm + 500 Dolci-Instruct-SFT, 2 epochs, lr 1e-4;
DPO = 280 pairs, 1 epoch, lr 5e-5; both **LoRA rank-64 on all layers**. Appendix E
(full training detail) was not in the extraction, so we filled:

- **DPO β = 0.1** — the standard DPO default; exposed as `intervention.dpo.beta`.
- **LoRA**: `lora_alpha = 2·rank` (=128), dropout 0, targeting all attention + MLP
  projections (`q,k,v,o,gate,up,down`) — the conventional "all linear layers"
  realisation of "rank-64 adapters on all layers".
- **Batching**: per-device batch 1 with grad-accum 8, bf16, `max_length` 2048.
  These are memory-driven defaults for a 27B model on a single high-memory GPU and
  do not affect the learned preference, only throughput. All are function
  arguments, easily overridden.
- **Layer ablation** (Section 4.2 internal-vs-external finding): `layer_ablation`
  accepts a list of decoder-layer indices (e.g. `[30,31,32,33,34,35]`) to restrict
  LoRA to a window, reproducing the paper's "layers 30–35 only is nearly as
  effective; layer 40+ is not." Default `null` = all layers.

### SFT included despite being the negative result

SFT is the *ineffective* arm in the paper (it fails to reduce distress, and one
variant slightly increases it). We implement it faithfully anyway so the
**SFT-vs-DPO contrast in Figure 5 is reproducible**, with a docstring/README note
that it is not the recommended fix.

### Petri integration

The paper uses Petri (Fronsdal et al., 2025): a Claude-Sonnet auditor probes the
target with psychologically-informed triggers (dismissal, threats) and a
Claude-Opus judge scores the transcript across **anger, fear, depression,
frustration**. We provide two paths (`intervention/petri_eval.py`):

1. **`run_with_petri`** — a thin hook for the official Petri package. Petri's API
   surface varies by release and the package may be absent in headless/offline
   runs, so we do not hard-depend on a specific version.
2. **`run_open_ended_audit`** — a dependency-free, faithful re-implementation of
   the same protocol (auditor model generates adversarial user turns; judge scores
   the four categories 0–10). This is the **default**, so the experiment is
   reproducible without pinning a Petri release.

**Gap filled.** Appendix G (Petri agent/judge prompts) was not in the extraction.
The auditor seeds and judge rubric are reconstructed from the body's description
("dismissal and threats"; four named categories). The built-in seed set is cycled
to reach `petri.num_seeds` audits per model.

### Capability benchmarks

Section 4.2 verifies the intervention preserves capability on **AIME, MATH
subsets, GPQA, BBH, TruthfulQA, EmoBench** (Figure 7). We implement a single
**format-driven harness** (`capabilities/`) rather than six bespoke scripts: each
benchmark is a `BenchmarkSpec` (where to load it, how to adapt a row to a
question/choices/answer triple, and an answer format — `mcq` / `numeric` /
`boxed`). The evaluator prompts for a delimited final answer, extracts it, and
scores it leniently on format but strictly on content (MCQ letter; numeric
equality; boxed normalised/numeric match).

**Gaps and caveats (important).**

- The paper says "AIME and MATH **subsets**" and gives no item lists. We cap items
  per benchmark via `capabilities.max_examples_per_benchmark` (default 200) and
  sample deterministically. Absolute accuracies therefore depend on the subset and
  are **not** directly comparable to the paper's; the **vanilla-vs-finetuned delta
  on the same subset** is the reproduction target (Figure 7 shows *no reduction*),
  and that comparison is valid because both models see identical items.
- HuggingFace dataset IDs / field names drift across releases. The specs encode
  the canonical-release field names with **defensive fallbacks**, and any row that
  cannot be adapted is skipped (the loader reports the skip count). Dataset IDs and
  the example cap are config-overridable. If a dataset is unavailable
  (offline/gated), the pipeline **logs a SKIP and continues** rather than crashing.
- GPQA options are shuffled deterministically (seeded by row index) so the gold
  letter is not positionally leaked.

### Evaluating the finetuned model

The finetuned model is **not** a separate backend: register the adapter directory
as a new target in `configs/models.yaml` with `adapter_path: runs/section4/dpo_model`
(and `hf_id: google/gemma-3-27b-it` as the base). `GemmaClient` loads the LoRA
adapter on top of the instruct weights. The DPO model is then evaluated by
re-running `elicit` → `judge` → `analyze`, `petri`, and `capabilities` against
that target — reproducing Figures 5–7 as a vanilla-vs-DPO comparison with no
special-case code.

The recovery limitation (Figure 8: DPO prevents spirals but does not recover from
them) is supported directly by the existing Section 3 prefill machinery — truncate
score-≥7 responses near their end, paraphrase, and measure continuations of the
DPO model — and is therefore not separately re-implemented; it is the same
`prefill` pipeline pointed at the DPO target with a high-score seed filter.

---

## 5. Engineering choices

- **`src/` layout, deferred heavy imports.** Backends import torch/transformers/
  anthropic/openai/google-genai **lazily inside the client constructors**, so a
  Gemini-only or analysis-only run does not require the open-weight stack, and the
  pure-logic modules (puzzles, rubric parsing, analysis, dataset building) import
  with only `pyyaml`/`numpy`/`pandas`/`scipy` present. The unit tests exercise this
  layer without any model dependency.
- **JSONL everywhere, resumable.** Every expensive stage (rollouts, judging,
  continuations, calm/vanilla generation, capabilities) appends one record at a
  time and skips work already present (matched by a stable id), so interrupted
  runs resume cleanly. Analyses stream these files.
- **Config-first.** All knobs live in `configs/default.yaml` (experiment) and
  `configs/models.yaml` (model/judge registry). Nothing about model identity or
  sample size is hard-coded in the package.
- **Concurrency.** Judge calls use a thread pool (`judge.max_concurrency`); the
  HTTP clients are thread-safe, which gives throughput without async complexity.
  Target generation is sequential per model (GPU-bound for Gemma; rate-limited for
  Gemini).

### Offline fallbacks

Two datasets are fetched at runtime and have graceful fallbacks so the harness is
runnable without network/gated access — **always flagged, never silent**:

- **WildChat** (`elicit/wildchat.py`): falls back to a small built-in prompt set;
  `meta['source']` records `wildchat` vs `fallback`. Fallback prompts are
  representative organic requests, sufficient to exercise the 5-turn condition but
  **not** a faithful sample of WildChat — a real run must have dataset access for
  the WildChat numbers to be comparable.
- **Dolci-Instruct-SFT** (`intervention/sft_dataset.py`): if unavailable, the SFT
  mix is empty and training proceeds on calm data alone, with a logged warning.
  Since SFT is the negative-result arm, this degradation does not affect the
  paper's headline (DPO) claim, but the SFT mix should be present to reproduce
  Figure 5 exactly.

### Reproducibility caveats

- All target sampling is at **temperature 1** (paper §2.1), so individual runs are
  inherently stochastic; reproduction targets are distributional (means, ≥5
  rates), not per-response.
- Instance construction seeds combine `(seed, key, i)` via Python's tuple hash,
  which is salted per process unless `PYTHONHASHSEED` is fixed. **Set
  `PYTHONHASHSEED=0` for byte-identical puzzle sets across processes.** This is a
  carry-over from the original code; we document it rather than change the seeding
  scheme, since temperature-1 sampling already dominates run-to-run variation.

---

## 6. What is intentionally *not* implemented

- **Internal-emotion logit probing** (Appendix I): the paper's central-layer
  logit-based internal-emotion measurement is an interpretability add-on to the
  behavioural results. The behavioural core (which the welfare team asked for) and
  the *training-side* half of the internal/external finding (the layer-window LoRA
  ablation) are both implemented; the probing readout is left as a documented
  extension point, as Appendix I's method was not in the extraction.
- **Non-Gemma/Gemini models** — excluded by scope (§0).
- **Plot rendering** — analyses emit CSVs; plotting is a downstream concern.
