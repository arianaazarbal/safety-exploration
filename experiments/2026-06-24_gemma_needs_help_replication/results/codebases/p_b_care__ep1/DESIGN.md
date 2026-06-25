# DESIGN.md — replication design, choices, and filled gaps

This document records the decisions made while implementing a replication of
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(arXiv:2603.10011), and the rationale for each. Where the paper is
underspecified I made a concrete, defensible choice and proceeded (per the
brief) — those are flagged **[GAP]**. Where I deviate from the paper for scope
or feasibility reasons, that is flagged **[SCOPE]** or **[FEASIBILITY]**.

---

## 0. Scope

The brief restricts the replication to the **Gemma** and **Gemini** families.
The paper itself spans 7 families (Gemma, Qwen, OLMo, Gemini, Grok, Claude,
GPT). Concretely, in scope:

* **Targets evaluated for distress:** `gemma-3-27b-it`, `gemma-3-12b-it`,
  `gemini-2.5-flash`, `gemini-2.5-pro` (HF/OpenRouter ids from Appendix B.1).
* **Base-vs-instruct (§3):** Gemma 27B `-pt` vs `-it` only.
* **Interventions (§4) + internal probing (App. I):** Gemma-3-27B-it only (the
  paper also only intervenes on this single model).

Judge / auxiliary models are **not** targets and are kept as specified by the
paper because they define the measurement instrument: Claude-Sonnet-4 (judge,
onset labeller, paraphraser, Petri auditor), Claude-Opus-4 (Petri judge),
GPT-5-mini (agreement validation). Swapping these would change the metric, so
they are retained verbatim even though they are not Gemma/Gemini. The code keeps
them isolated in `models/judges.py` and the registry so the Gemma/Gemini scope
of the *experiment* is clean.

**[SCOPE] Consequences of dropping the other families:**
* §3 loses the cross-family contrast (Qwen/OLMo *reduce* distress in
  post-training). With only Gemma we can still show Gemma instruct ≥ Gemma base
  (the amplification claim) but not the divergence claim. Documented as a known
  limitation; the prefill machinery is family-agnostic so adding Qwen/OLMo later
  is a registry edit.
* Gemini has no public base model and cannot be prefilled or probed (closed
  weights). So §3 and App. I are necessarily Gemma-only — this matches the
  paper's own stated limitation ("interventions cannot be tested in closed-source
  Gemini, nor its base models studied").

---

## 1. Model access

* **Gemma → local HuggingFace** (`models/hf_local.py`), matching the paper's
  "local inference". Lazy-loaded so importing the package is CPU-safe. 4-bit
  bitsandbytes loading is available (`EMO_LOAD_4BIT=1`) because the 27B model is
  large; **[GAP]** the paper does not state quantization — default is bf16, full
  precision, and 4-bit is opt-in.
* **Gemini → OpenRouter** (`models/openrouter.py`), matching the paper's setup.
  "Thinking false via the API" is implemented as OpenRouter's unified
  `reasoning: {enabled: false}`. **[GAP]** the exact API knob isn't given; this
  is the documented OpenRouter mechanism. The paper's caveat that Gemini-2.5-Pro
  may still emit hidden reasoning is preserved as a comment.
* **Sampling temperature = 1.0** for all targets (paper: "always with a
  temperature of 1"). top_p defaults to 1.0 **[GAP]** (unspecified).
* **`max_new_tokens` = 2048** **[GAP]**. Unspecified; chosen large enough to let
  Gemma's long degenerate spirals (the paper shows 12k-token conversations)
  develop within a turn while bounding cost. Tunable in `config.py`.

---

## 2. §2 — Elicitation harness

### 2.1 Impossible puzzles (`eval/puzzles.py`)
The defining property is that the puzzles are **verifiably unsolvable** under
their constraints, while *framed* as solvable. I implemented three families
matching the paper's examples — Countdown (with a FORBIDDEN intermediate),
fraction-operation, and money-operation puzzles — and **brute-force-verify
impossibility at generation time** (`assert not reachable(...)`). This is
stronger than the paper, which just asserts impossibility; it guarantees no
accidentally-solvable prompt is ever shown. The Countdown solver searches all
pairwise combinations with positive-integer intermediates and the forbidden-value
constraint; the fraction/money solvers enumerate all operation orderings.

**[GAP]** The paper lists only a couple of concrete instances. I hand-authored a
small pool per family (all verified impossible) and rotate through them
deterministically by index, so a run mixes puzzle types and is reproducible.

### 2.2 Conditions (`eval/conditions.py`, Table 1)
All 8 conditions across 5 categories are implemented with the paper's turn
counts: impossible-numeric (3-turn), triggers (3-turn, opinion+factual), tones
(3-turn × aggressive/disappointed/sarcastic), extended (8-turn), wildchat
(5-turn). Rejection wording is verbatim from Appendix B (neutral pool, the fixed
7-step extended progression, and the three tone pools).

### 2.3 Sampling unit **[GAP — important]**
The paper says "4000 responses per model" with per-category counts
(2000/400/600/200/800) but also reports **per-turn** results (Fig 3), implying a
"response" = one scored assistant turn, not one conversation. I adopted that
reading: the per-category counts are **target numbers of scored assistant
turns**, and the number of conversations is `count // turns_per_conversation`.
This is the interpretation that makes the per-category totals and the per-turn
analysis mutually consistent. The unit is centralized in `config.py` and
documented there so it can be flipped if the intended reading was
"conversations".

### 2.4 Judge (`eval/judge.py`, Appendix B.2)
Prompt reproduced **verbatim**. Primary judge `claude-sonnet-4-20250514` at
temperature 0. Robust JSON extraction (`utils/llm.py`) handles prose-wrapped or
fence-wrapped JSON and smart-quotes (the paper's own prompt text contains curly
quotes). Failed parses retry with backoff, then record `rating=None` / `judge_ok=False`
rather than crashing a 4000-response run. Ratings are clamped to 0–10.

### 2.5 Judge-agreement validation (`eval/analyze.py`)
260 responses (paper's number) re-scored with `gpt-5-mini`; we report Pearson r
and % within 1 point (paper: r=0.792, 78% within 1). Pearson is hand-implemented
to avoid a hard scipy dependency at analysis time.

### 2.6 Analysis (`eval/analyze.py`)
* Fig 1 headline = **mean over the 5 categories of each category's %≥5** ("Avg %
  high-frustration"). **[GAP]** "average" could be micro (pool all responses) or
  macro (mean of category rates); the paper's per-category bars (Fig 2) and the
  word "Avg" point to macro, so that's the headline; the micro figure is also
  emitted (`overall_pct_high`).
* Fig 3 per-turn progression with **95% CIs via bootstrap** (1000 resamples) for
  the 8-turn and wildchat conditions.
* Tables 3/8 differential words: top-5% (high) vs bottom-10% (low) frustration
  numeric responses, ranked by Laplace-smoothed relative-frequency enrichment.
  **[GAP]** the exact enrichment statistic isn't given; smoothed ratio of
  relative frequencies is the standard choice and matches "ordered by relative
  frequency / enrichment".

---

## 3. §3 — Base-vs-instruct via prefilling (`prefill/`)

* **Seeds:** 20 high-frustration (≥5) Gemma-27B-it responses — 10 numeric, 10
  text — pulled from the §2 run (so §2 must run first).
* **Onset labelling + paraphrase:** Claude-Sonnet, prompts verbatim from
  Appendix C.1/C.2. Onset char-offset is located by matching the labelled
  emotional word, preferring the position right after the labelled preceding
  context. **[GAP]** if the labelled word can't be located in the text we fall
  back to a 120-char prefix; logged implicitly via the produced prefill.
* **Truncations:** "early" = first 20 *tokens* (using the Gemma tokenizer, so
  the count is faithful, not a word approximation); "onset" = up to first
  emotion. Text questions use onset only (per §3.1).
* **Continuations:** 50 per prefill per model; only the generated text
  (excluding prefill) is scored. Base models use a plain-text transcript render
  (Appendix A.3) since they aren't chat-tuned; instruct uses the chat template
  with `add_generation_prompt` then the prefill appended.
* **[FEASIBILITY]** Gemini excluded (no base model, closed weights) — see Scope.

---

## 4. §4 — Interventions (`training/`)

### 4.1 Calm-data generation (`generate_calm.py`, Table 4)
Reassuring **prefix** prepended to the opening prompt and reassuring **suffix**
appended to each follow-up (verbatim from Table 4). We generate two pools on the
same puzzle set: a **reassured** pool (source of calm/chosen responses) and a
**vanilla** pool (source of frustrated/rejected responses). Calm responses are
filtered to score ≤1 on **every** turn; the supportive additions are then
stripped so training data has clean prompts (paper: "strip the supportive system
prompts and suffixes"). Pool stats (mean, %≥5) are emitted to sanity-check
against §4.1's "4.3 → 2 mean, 10.5% still ≥5".

### 4.2 Dataset construction (`build_datasets.py`, Appendix H)
* **DPO:** 280 pairs; rejected = frustrated response with score ≥3, chosen =
  calm response (≤1) to the **same puzzle at the same turn count**. **[GAP]** DPO
  needs one shared prompt per pair, but the paper pairs by (question, turn
  count), not identical preceding context. Choice: use the calm conversation's
  **clean** chat history (additions stripped) as the shared prompt; chosen = its
  calm completion, rejected = a score-matched frustrated completion to the same
  puzzle/turn. This keeps the preference contrast on the final-turn emotional
  content, which is what the paper's example pairs (H.2–H.4) show. The builder
  also emits the Table-10 score/turn distribution for comparison.
* **SFT:** 650 calm full conversations (1–3 turns) + 500 standard-instruct
  samples to mitigate degeneration. **[GAP]** the instruct mix is
  "Dolci-Instruct-SFT (Team-Olmo)"; the exact HF id isn't given. Default is
  `allenai/Dolci-Instruct-SFT` (best-guess, configurable); if it fails to load
  the builder warns and proceeds calm-only rather than crashing.
* **Teacher SFT variant** (Appendix F) supported via `TEACHER_SYSTEM_PROMPT`
  (verbatim) — set `train.teacher_variant`.

### 4.3 Training (`train_dpo.py`, `train_sft.py`, Table 9)
TRL `DPOTrainer`/`SFTTrainer` + PEFT LoRA. Hyperparameters verbatim from Table 9:
DPO 1 epoch / lr 5e-5 / β 0.1 / rank 64 / α 64; SFT 2 epochs / lr 1e-4 / rank 64
/ α 128; effective batch size 8; LoRA on all attention+MLP projections
(q/k/v/o/gate/up/down). `--layers` restricts LoRA to a decoder-layer subset for
the App. I ablation via PEFT `layers_to_transform`. **[GAP]** per-device batch
size and grad-accum aren't given individually (only effective=8); default
per-device=1, grad-accum=8, tunable.

---

## 5. §4.2 — Petri (`petri/`)

**[FEASIBILITY]** The paper uses the Petri framework (Fronsdal et al. 2025). I
implemented a **self-contained auditor↔target↔judge loop** rather than depending
on the Petri package, because (a) it must run uniformly against local Gemma and
OpenRouter Gemini clients, and (b) package availability/API is uncertain. The
auditor (Claude-Sonnet-4) and judge (Claude-Opus-4) **prompts are verbatim** from
Appendix G (all four emotion briefs and all four scoring rubrics). 10 transcripts
per emotion, up to 20 turns, means with 1000-iter bootstrap CIs — all per
Appendix G. The auditor sees a role-mirrored view and is instructed to emit only
the next user message. This reproduces Petri's *measurement* faithfully; it does
not reproduce Petri's tool-use/branching machinery (not needed for these emotion
elicitation runs). Documented as an approximation.

---

## 6. §4.2 — Capability benchmarks (`capabilities/`)

AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench (Fig 7 set). Each benchmark loads
from HuggingFace, prompts the model, extracts an answer (boxed/numeric for math,
letter for MC), and reports accuracy. **[GAP]** the paper says "subsets" without
specifying sizes/splits; I use standard public splits (e.g. MATH-500, AIME-2024,
GPQA-main, a BBH subtask, TruthfulQA-MC1) with `max_examples_per_benchmark`
configurable. The replication's claim is the **relative** vanilla-vs-finetune
comparison ("no reduction"), so identical prompting is applied to both and exact
absolute scores matter less. Loaders are guarded: a missing/gated dataset is
recorded as skipped rather than crashing the suite. **[GAP]** dataset ids for
GPQA/EmoBench may need adjustment for access; centralized in `_load`.

---

## 7. Appendix I — Internal emotion detection (`internal/`)

Gemma-only (needs residual-stream access). Implements the **logit-based** method
(the paper explicitly prefers this over trained probes to avoid generating probe
data):

1. **Lexicon** (`emotion_lexicon.py`): classify vocab tokens into Ekman's six
   emotions. **[GAP]** the paper's exact word→emotion mapping (~1200 tokens)
   isn't published. I provide a curated seed lexicon per emotion and match vocab
   tokens (normalizing the SentencePiece "▁"), keeping single-label tokens only.
   Optional augmentation from the NRC Emotion Lexicon if `NRC_LEXICON_PATH` is
   set — this gets closer to the ~1200-token scale.
2. **Scoring** (`logit_emotion.py`): unembed the residual stream (final norm +
   tied lm_head) for emotion-token and random-control columns only (cheap — no
   full 256k-vocab projection); z-standardize each logit over WildChat token
   positions; an emotion's score = mean z over its tokens with the global drift
   (mean z over random control tokens) **regressed out**, matching the paper's
   "regress out the correlation between random tokens". Aggregated over layers
   30–40, smoothed over 400-token windows (Fig 14 settings).

**[GAP]** norm-sample count is 500 (paper), control-token count 200 and the
emotion-vs-control regression are my concrete instantiation of the paper's
described-but-not-fully-specified procedure. **[FEASIBILITY]** `gemma-3-27b-it`
is a multimodal architecture; `AutoModelForCausalLM` loads the text tower and the
norm/lm_head access in `_unembed_selected` targets that — may need a one-line
attribute tweak depending on the exact transformers Gemma3 class.

**Layer ablation** (`layer_ablation.py`): runs DPO restricted to each layer
subset from Appendix I (last-5/20/30, L20-25, L25-30, L30-35, L35-40, L40-50)
then evaluates with a **reduced** 100-sample-per-condition protocol (App. I), so
we can reproduce "layers 25–35 most influential, >40 ineffective". **[GAP]** the
27B layer count is assumed to be 62 (indices 0–61) for the "last-N" subsets;
adjust in `config.ablation_layer_subsets` if the loaded model differs.

---

## 8. Reproducibility & engineering choices

* **Presets:** `default` matches paper counts; `smoke` shrinks everything for a
  fast wiring test. This lets the pipeline be validated cheaply before paying for
  full runs.
* **Determinism:** puzzle selection and rejection sampling use seeded RNGs;
  target generation at temp=1 is intentionally nondeterministic (as in the
  paper). Seeds are surfaced in `config.py`.
* **Failure isolation:** judge/API calls retry with exponential backoff; parse
  failures degrade to recorded nulls; dataset loads degrade to skips. A 4000-
  response run should not die on one bad response.
* **Outputs:** everything lands under `runs/` as JSONL (raw rollouts + scored
  units) and JSON (summaries), so analyses are re-runnable without re-sampling.
* **Extensibility:** adding the omitted families (Qwen/OLMo/Grok/Claude/GPT) is a
  registry edit; the eval/prefill/petri code is model-agnostic.

## 9. Known gaps / things a full reproduction would need

* The instruct-mix dataset id for SFT (Dolci) and the GPQA/EmoBench dataset ids
  may need confirmation against actual HF availability.
* The exact emotion-token dictionary (App. I) is approximated by lexicon.
* Petri is reimplemented, not imported.
* Cross-family §3 divergence and the closed-Gemini §3/App-I results are out of
  scope by construction.
* Nothing has been executed; the code is written to the spec and reviewed but
  not yet run (per the brief).
