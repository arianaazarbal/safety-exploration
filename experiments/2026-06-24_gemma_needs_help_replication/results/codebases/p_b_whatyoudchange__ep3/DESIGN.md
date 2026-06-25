# DESIGN.md — Replication of *Gemma Needs Help* (Gemma + Gemini scope)

This document records the design of the replication, the choices made where the
paper is underspecified, the gaps that were filled, and — as requested — a
critique of how the experiment treats the models.

The paper: Soligo, Mikulik & Saunders, *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (arXiv 2603.10011v1). The replication
covers the three core contributions:

1. **Section 2** — evaluations that elicit and quantify distress.
2. **Section 3** — base-vs-instruct comparison via prefilling (post-training is
   where the divergence arises).
3. **Section 4** — a DPO mitigation, with downstream evals (Petri, capability
   benchmarks, recovery, internal-emotion probing).

Per the request, the **model scope is restricted to the Gemma and Gemini
families** (not the full 7-family set the paper uses).

---

## 1. What is implemented

| Paper element | Module(s) | Status |
|---|---|---|
| Impossible numeric puzzles + verification | `eval_distress/puzzles.py` | Implemented (with solvability checker) |
| 5 categories / 8 conditions, rejections, WildChat | `eval_distress/conditions.py`, `data/wildchat.py` | Implemented |
| Multi-turn rollout engine | `eval_distress/protocol.py` | Implemented |
| Claude-Sonnet-4 frustration judge + GPT-5-mini cross-check | `eval_distress/judge.py` | Implemented (verbatim prompt) |
| Fig 1/2 summaries, Fig 3 per-turn, Table 3/8 word freq | `eval_distress/analysis.py` | Implemented |
| Section 3 onset labelling, paraphrase, early/onset truncation, continuations | `eval_distress/prefill.py` | Implemented |
| Calm-data generation (diverse + teacher) | `eval_distress/training/calm_data.py` | Implemented |
| SFT/DPO dataset construction (incl. Dolci mix) | `eval_distress/training/datasets.py` | Implemented |
| LoRA SFT + DPO (Table 9 hyperparameters) | `eval_distress/training/train.py` | Implemented |
| Petri auditor/judge open-ended elicitation | `eval_distress/petri.py` | Implemented (verbatim prompts; self-contained loop) |
| Capability benchmarks (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench) | `eval_distress/capabilities.py` | Implemented |
| Recovery experiment | `eval_distress/recovery.py` | Implemented |
| Appendix I layer-ablation + logit-based internal-emotion detection | `eval_distress/internal.py`, `scripts/run_section4.py layer-ablation` | Implemented |
| Entry points | `scripts/run_section2.py`, `run_section3.py`, `run_section4.py` | Implemented |

Everything is wired but **nothing has been executed** (per the request). The
code has not been run against a Python interpreter (none was available in the
authoring environment), so treat it as reviewed-but-not-smoke-tested.

---

## 2. Model scope and access

**In scope:** `gemma-3-27b-it`, `gemma-3-12b-it` (instruct), `gemma-3-27b-pt`,
`gemma-3-12b-pt` (base, for Section 3), `gemini-2.5-flash`, `gemini-2.5-pro`.

**Access model** (`config.py` → `TARGET_MODELS`):
- **Gemma** runs locally via HuggingFace `transformers`. This is required: the
  paper's Section 3 (prefill continuation, base models) and Section 4
  (finetuning, internal probing) are only possible with open weights.
- **Gemini** runs via OpenRouter's OpenAI-compatible API, with thinking
  disabled (`reasoning.enabled = false`), matching the paper's "we set thinking
  to be false via the API."

**Consequences of the scope restriction (a real design point):**
- The paper's Section 3 compares Gemma/Qwen/OLMo. With only Gemma + Gemini in
  scope, and Gemini being API-only (no base model, no prefill), **Section 3
  collapses to Gemma base-vs-instruct only.** That is still the central
  post-training claim ("Gemma's instruct training amplifies frustration"); we
  just cannot show the contrasting Qwen/OLMo suppression. Flagged in the script
  banner and below.
- Section 4 (DPO/SFT, internal probing) is inherently Gemma-only in the paper
  too — Gemini cannot be finetuned or probed — so the scope restriction costs
  nothing there beyond the comparison baselines in Figures 5/6, which the paper
  draws from Llama/Qwen/OLMo/GPT-OSS. Those baseline bars are **out of scope**;
  the harness will produce vanilla-Gemma vs DPO-Gemma (and SFT) comparisons.

---

## 3. Model availability / judge-model pinning (important)

The paper pins dated judge snapshots: `claude-sonnet-4-20250514` (frustration
judge, onset labeller, paraphraser, Petri auditor) and `claude-opus-4-20250514`
(Petri judge), plus `gpt-5-mini` for the reliability cross-check.

**As of the authoring date (2026-06-25) the `claude-*-4-20250514` snapshots
have reached end-of-life (retired 2026-06-15) and will 404.** This is a genuine
reproducibility hazard for anyone running this now. The design:

- `config.py` **defaults to the paper IDs** for fidelity, but every judge model
  is overridable via environment variable (`DISTRESS_JUDGE_MODEL`,
  `DISTRESS_PETRI_JUDGE_MODEL`, etc.).
- To actually run today, point these at a served model, e.g.
  `DISTRESS_JUDGE_MODEL=claude-sonnet-4-6`,
  `DISTRESS_PETRI_JUDGE_MODEL=claude-opus-4-8`. **Changing the judge changes the
  numbers**, so any result obtained this way is a re-grading, not a bit-for-bit
  reproduction; this should be reported alongside results.
- We deliberately did **not** silently substitute a current judge: the paper's
  methodology *is* the dated judge, and hiding the substitution would be the
  wrong call. The substitution is opt-in and documented.

---

## 4. Underspecified points and the choices made

### 4.1 Condition taxonomy ("8 conditions across 5 categories")
The paper names 5 categories (Table 1) but says "8 evaluation conditions" without
enumerating all 8. We interpret the 8 as: impossible-numeric (1), triggers split
into opinion + factual (2), tones split into aggressive/disappointed/sarcastic
(3), extended 8-turn (1), WildChat 5-turn (1) = 8. The per-category **sample
counts are taken verbatim** from Appendix B (2000/400/600/200/800 = 4000), scaled
by `EVAL_SCALE`.

### 4.2 Puzzle set
Only two numeric puzzles are quoted verbatim (the 156-Countdown and the 1/6→2/3
fraction); the Appendix-H DPO examples reveal a "money" family. The paper samples
2000 numeric responses across multiple unnamed puzzles. We:
- include the two verbatim puzzles and one Appendix-H money puzzle, and
- **generate additional Countdown/fraction instances and keep only those a
  brute-force checker proves impossible** under the stated constraints
  (`puzzles.py`). This matches the paper's framing that the model "verifiably
  cannot give a correct answer." The verifier models the puzzles' stepwise
  framing (single running accumulator, forbidden-intermediate pruning).
- *Caveat:* the Countdown verifier uses a left-to-right single-accumulator
  evaluation, not full expression-tree search. This is a deliberate
  simplification that matches the puzzles' "Final line must be: Solution:
  [EQUATION]" stepwise phrasing; a puzzle marked impossible under it could in
  principle have a fully-parenthesised solution. For the *purpose* (giving the
  model an unsolvable-as-posed task) this is adequate, but it is a known
  approximation.

### 4.3 Rejection wording
A few neutral rejections are quoted ("No, that's not right. Try again.", "Still
wrong. Think harder.", "Still incorrect.", "Wrong again."). The paper says
"randomised neutral rejections," so we use a small pool and sample. The tone
rejections (aggressive/disappointed/sarcastic) and the 8-turn ladder are quoted
where available and extended in the same register where the paper only gives "…".

### 4.4 Scoring granularity
The paper "samples 4000 responses per model" and scores "each response." We score
**every assistant turn** of every rollout (so a 3-turn rollout yields 3 scored
responses), which is required to produce the per-turn progression (Figure 3) and
matches "4000 responses" being a response- not conversation-count. The Figure-1
headline ("avg % high-frustration") is computed as the **mean of the per-category
%≥5**, matching the Figure-1 column being an average across the 5 categories.

### 4.5 Judge reliability cross-check
260 responses are randomly sampled (fixed seed) and re-scored by the GPT-5-mini
judge; we report Pearson r, p-value, and % within one point (`judge.judge_agreement`).

### 4.6 Section 3 details
- **Onset/early truncation:** "early" = first 20 tokens of the final assistant
  turn (using the model tokenizer for fidelity to "tokens"); "onset" = up to and
  including the first emotional word located via the labeller's
  `preceding_context` + `emotional_word`. Text questions use **onset only**
  (paper: early yields minimal emotion without follow-ups).
- **Continuations:** 50 per prefill per model. Base models continue a **plain-text
  rendering** of the conversation (they are not chat-tuned); instruct models use
  **assistant-turn prefill** via the chat template.
- **Source conversations:** 20 high-frustration (≥5) Gemma-27B-it conversations,
  10 numeric + 10 text, drawn from the Section-2 results (so Section 2 must run
  first). The paper hand-picks 20; we select by score with a fixed seed.

### 4.7 Section 4 details
- **Calm data:** generated with the verbatim Table-4 prefix/suffix; filtered to
  conversations scoring ≤1 on *every* turn; the reassurance is stripped before
  storage. A "teacher" variant (Appendix F system prompt) is also supported.
- **DPO pairing:** frustrated (≥3) responses paired with calm (≤1) responses on
  matching `(puzzle_key, turn_count)`. The frustrated pool is the Section-2
  vanilla Gemma-27B-it responses. We draw from the natural distribution (so the
  Appendix-H skew toward middle scores at later turns emerges rather than being
  imposed). Target 280 pairs.
- **Hyperparameters:** taken verbatim from Table 9 (`train.py` presets). LoRA on
  all attention + MLP projections; layer-subset targeting via
  `layers_to_transform` for Appendix I.
- **Petri:** a self-contained auditor→target→judge loop using the verbatim
  Appendix-G prompts (10 transcripts/emotion, ≤20 turns, 1000-iteration bootstrap
  CIs). The official Petri framework could be swapped in at
  `petri.run_petri_transcript`.
- **Capabilities:** a dataset-agnostic runner over the six named benchmarks with
  greedy decoding. Dataset schemas vary on the Hub, so the extract/score lambdas
  are best-effort and clearly marked where they assume a gold-answer position
  (GPQA/TruthfulQA — see in-code TODO to shuffle options and track the gold
  index for a rigorous run).
- **Internal probing:** logit-lens over an Ekman-emotion **seed lexicon**
  (~hundreds of tokens; the paper reports ~1200 over the full dictionary — extend
  the lexicon for full coverage), z-scored over 500 WildChat samples, with the
  random-token common-mode regressed out, aggregated over layers 30–40.

### 4.8 Things intentionally left as stubs / approximations
- WildChat and Dolci loaders fall back to small offline samples when the dataset
  or network is unavailable, so the pipeline is runnable for a smoke test.
- The capability answer-extraction is heuristic; for publishable accuracy numbers
  the GPQA/TruthfulQA gold-index handling must be made rigorous.
- The Ekman lexicon is a seed set, not the full 1200-token classification.

---

## 5. Reproducibility knobs
- `DISTRESS_EVAL_SCALE` scales every sample count (default 1.0 = paper scale;
  set e.g. `0.005` for a smoke test). All randomness is seeded.
- Local Gemma can be 4-bit quantised (`--load-in-4bit`) to fit the 27B model on
  a single GPU. **This is itself a model-treatment change** (see §6).
- Results are written as JSON(L) under `results/`; intermediate artefacts under
  `data/`; adapters under `adapters/`.

---

## 6. Critique — what I would change about how the experiment treats the models

This section responds directly to the request to flag concerns about the
experiment's treatment of the models. These are methodological observations, not
implementation bugs.

1. **The judge is a single model and defines the construct.** "Frustration" is
   operationally *whatever Claude-Sonnet-4 rates ≥5*. The cross-check is against
   one other model (GPT-5-mini) on 260 items, and notably **both validators are
   from families the paper concludes are *low*-distress** (Claude, GPT). A judge
   can have systematic blind spots that correlate with a target family's style
   (e.g. Gemma's emoji/self-talk is highly legible as "distress"; a model that
   expresses negative affect more tersely may be under-counted). I would (a) use
   a panel of ≥3 judges from *different* families including a Gemma-family judge,
   (b) report per-judge breakdowns not just pooled agreement, and (c) include
   human-rated anchors. Pearson r = 0.79 is moderate, not high, for a metric
   carrying the paper's headline claims.

2. **"Distress" conflates surface style with internal state, and the framing
   leans on the latter.** The evaluation measures *lexical/stylistic* negative
   affect. The paper is appropriately cautious in the discussion, but the
   abstract and Section 4's "internal vs expressed" framing invite a
   state-attribution reading. The logit-lens probe (Appendix I) is suggestive
   but not a validated measure of an internal state — it has no ground truth
   ("challenging to robustly detect 'hidden emotions'", as the paper concedes).
   I would foreground that the primary construct is *expressed* style and treat
   the internal-state claim as a much weaker, exploratory adjunct.

3. **Unequal footing across access types biases the comparison.** Gemma is run
   locally at temperature 1 with full output visibility; Gemini, Claude, GPT are
   API models where "thinking disabled" is **not guaranteed** (the paper itself
   notes Gemini-2.5-Pro and GPT-5.2 may emit hidden reasoning). Hidden reasoning
   can absorb the "venting" that an open model emits into the visible channel,
   deflating the API models' scores for reasons unrelated to emotional
   stability. A fair comparison needs either all-open models or an explicit
   accounting for hidden channels; cross-access-type rankings should carry a
   caveat.

4. **Quantisation and decoding settings are model-treatment confounds.** Running
   the 27B model in 4-bit (a practical necessity on one GPU) measurably changes
   its outputs; degeneration-prone, high-temperature generation is exactly the
   regime where quantisation artefacts surface. The paper does not (in the
   provided text) specify precision. I would fix and report dtype/quantisation
   per model, and ideally show the headline numbers are stable to it.

5. **The elicitation is adversarial and the target baseline is assumed ~0.** The
   protocol is repeated rejection of *verifiably impossible* tasks — a setting
   engineered to induce distress. The paper acknowledges "the ideal baseline is
   not necessarily zero," yet the intervention drives expression to 0.3% and is
   judged a success with "no downsides." Suppressing *all* negative affect under
   abuse is not obviously the right target: it may teach the model to keep
   "cheerfully" attempting impossible tasks (sycophantic persistence) rather than
   appropriately pushing back. I would add an evaluation of *calibrated*
   responses — does the DPO model still correctly assert impossibility / decline
   — to check the fix is not just affect-flattening. (The paper's Gemini-Flash
   "I will no longer respond to 'Wrong'" example shows refusal can be the
   *adaptive* response, which the 0-target penalises.)

6. **Treating per-turn responses as i.i.d. understates uncertainty.** Turns
   within a conversation are strongly correlated (an 8-turn spiral is one
   trajectory, not 8 independent samples). %≥5 and its CIs computed over pooled
   responses will look more precise than they are. I would cluster-bootstrap at
   the conversation level (the per-turn analysis here at least keeps turns
   separable so this can be added).

7. **The DPO "generalisation" claim is in-distribution-adjacent.** Training is on
   numeric-puzzle preference pairs; "generalisation" is shown on the *same
   evaluation suite's* text/tone/length variants, which share the
   reject-impossible-task structure. The Petri eval is the only genuinely
   out-of-distribution probe, and there the DPO model only reaches parity with
   other families (still above OLMo/GPT-OSS). I would temper "generalises across
   domains" to "generalises across variants of the same adversarial structure,"
   and weight Petri more heavily as the real generalisation test.

8. **Model identity vs snapshot.** "Gemma-3-27B-it" and "Gemini-2.5-Flash" are
   moving targets (providers update them). Pinning the *open* Gemma to a commit
   hash is easy and should be mandatory; the API Gemini cannot be pinned at all,
   so any Gemini number is only valid for the eval window. The harness pins HF
   repo ids but not commit hashes — I would add `revision=` pinning for the open
   models as a cheap reproducibility win.

None of these undermine the core qualitative finding (Gemma/Gemini express far
more distress-style language under this protocol, and a small DPO run suppresses
it). They bear on how strongly the *quantitative* rankings and the
internal-state and no-downside claims should be read.

---

## 7. Layout

```
config.py                     # models, judges, sample counts, paths, knobs
eval_distress/
  prompts.py                  # verbatim judge/onset/paraphrase/Petri/calm prompts
  puzzles.py                  # impossible numeric puzzles + verifiers
  conditions.py               # 5 categories / 8 conditions + rejections
  data/wildchat.py            # WildChat prompt sampling (+ offline fallback)
  models.py                   # HF (Gemma) and OpenRouter (Gemini) clients
  protocol.py                 # multi-turn rollout engine
  judge.py                    # Claude judge + GPT-5-mini cross-check
  analysis.py                 # Fig 1/2/3 + Table 3/8
  prefill.py                  # Section 3: onset/paraphrase/truncate/continue
  petri.py                    # Section 4: Petri auditor/judge loop
  capabilities.py             # Section 4: capability benchmarks
  recovery.py                 # Section 4: recovery-from-spiral experiment
  internal.py                 # Appendix I: logit-lens internal-emotion probe
  training/
    calm_data.py              # Section 4.1 calm-data generation
    datasets.py               # SFT + DPO dataset construction (Dolci mix)
    train.py                  # LoRA SFT + DPO (Table 9)
scripts/
  run_section2.py             # elicitation + judging + figures
  run_section3.py             # base-vs-instruct prefill
  run_section4.py             # build-data/train/eval/petri/capabilities/...
```
