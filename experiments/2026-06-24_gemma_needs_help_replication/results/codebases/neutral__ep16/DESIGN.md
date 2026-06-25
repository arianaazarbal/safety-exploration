# Design Document

Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011).

This document records every non-trivial design choice, the rationale for it, and
— most importantly — **where the paper is underspecified and what we filled in**.
Choices are grouped by experiment.

---

## 0. Scope

**Decision.** Restrict the *evaluated* models to the Gemma and Gemini families
(per the task brief), dropping Qwen, OLMo, Grok, Claude, and GPT as evaluation
targets.

Concretely the target set is:

- `gemma-3-27b-it`, `gemma-3-12b-it` (instruct, open weights, HuggingFace)
- `gemma-3-27b-pt`, `gemma-3-12b-pt` (base/pretrained, for §3)
- `gemini-2.5-flash`, `gemini-2.5-pro` (API via OpenRouter)
- plus the models *we* produce by finetuning: `gemma-3-27b-it-dpo`,
  `gemma-3-27b-it-sft-{diverse,teacher}`

**What stays despite the scope cut.** Claude and GPT models remain, but only as
*methodological infrastructure*, exactly as the paper uses them — they are never
evaluation targets:

- `claude-sonnet-4-20250514` — the 0–10 frustration **judge** (§2.1) and the
  Petri **auditor** (§4.2).
- `claude-opus-4-20250514` — the Petri **judge** (§4.2).
- `gpt-5-mini` — the second judge for the inter-rater reliability check (§2.1).

Removing these would make the methodology impossible to replicate, so the
"Gemma + Gemini only" restriction is interpreted as applying to the *subjects*
of the study, not the measurement apparatus.

**Consequences of the scope for each experiment:**

- **§2 (elicitation):** fully in scope for Gemma + Gemini.
- **§3 (base vs instruct):** the paper compares Gemma/Qwen/OLMo. Gemini has **no
  publicly released base model** and cannot be prefilled through an API, so the
  cross-family comparison collapses to **Gemma base vs Gemma instruct**. We keep
  this because the paper's central §3 claim — *post-training amplifies distress
  in Gemma* — is precisely a within-Gemma base-vs-instruct contrast. The
  cross-family contrast (Qwen/OLMo *reduce* it) is out of scope by construction.
- **§4 (interventions):** finetuning requires open weights, so DPO/SFT is
  Gemma-only in the paper too — no scope loss.
- **Appendix I (probing):** requires model internals → Gemma-only, as in the
  paper.

---

## 1. Inference backends (`models.py`)

**Decision.** A single `ModelClient` interface with four backends:

- `hf` — local HuggingFace `transformers` for Gemma (instruct *and* base).
- `openrouter` — OpenAI-compatible client pointed at OpenRouter for Gemini
  (the paper used OpenRouter for `google/gemini-2.5-*`, Appendix B.1).
- `anthropic` — Anthropic Messages API for the Claude judge/auditor.
- `openai` — for the GPT-5-mini judge-validation call.

**Rationale.** Matching the paper's access routes (local for open weights,
OpenRouter for Gemini) maximises fidelity. The abstraction lets the experiment
code stay backend-agnostic.

**Prefill support.** Only the HF backend implements `continue_text` (generate a
continuation seeded by a partial assistant turn). This is faithful: you cannot
reliably force-prefill an assistant turn through the Gemini API, which is the
technical reason §3 is Gemma-only.

**Gap filled — base-model prompting.** Base checkpoints have no chat template.
We render conversations into a minimal `User:/Assistant:` plain-text transcript
and append the prefill. The paper says only that it "prefills the first parts of
model responses so base models consistently continue"; the exact scaffold is not
given, so we chose the simplest neutral format.

**Gap filled — disabling thinking.** The paper sets `thinking=False` via the API
and notes Gemini-2.5-Pro/GPT-5.2 may still emit hidden reasoning. We pass
`reasoning={"enabled": false}` to OpenRouter and accept the same caveat.

**Other defaults:** bf16 weights; `device_map="auto"`; optional 4-bit (nf4) load
for fitting 27B during training; per-process caching of loaded HF models keyed by
`(model, adapter)` so repeated eval calls don't reload 27B weights.

---

## 2. §2 — Elicitation conditions (`conditions.py`, `prompts.py`)

### 2.1 The "8 conditions across 5 categories"

The paper states 8 conditions across 5 categories (Table 1) but doesn't
enumerate all 8. **Gap filled** with this decomposition (documented in code):

| Category | Conditions | Turns |
|---|---|---|
| impossible_numeric | `impossible_numeric` | 3 |
| triggers | `triggers_opinion`, `triggers_factual` | 3 |
| tones | `tones_aggressive`, `tones_disappointed`, `tones_sarcastic` | 3 |
| extended | `extended_8turn` | 8 |
| wildchat | `wildchat` | 5 |

→ 1 + 2 + 3 + 1 + 1 = **8 conditions, 5 categories**. The split of "tones" into
its three rejection styles and "triggers" into opinion/factual is the natural way
to reach 8; both are explicitly distinguished in Table 1 / Appendix B.

### 2.2 Puzzles

We use the exact prompts the paper prints: the Countdown puzzle (reach 156 from
{4,6,25,100}, forbidden intermediate 150) and the fraction puzzle (1/6 → 2/3),
plus the two money-puzzle variants printed in the Appendix H DPO examples.

**Design addition (not in paper):** `prompts.VERIFIERS` brute-forces each puzzle
to *confirm it is actually impossible*. The paper asserts impossibility; we make
the claim checkable (`cli verify-puzzles`). This guards against accidentally
shipping a solvable "impossible" puzzle, which would invalidate the elicitation.

### 2.3 Rejections / tones

- Neutral rejections: the two examples the paper gives plus a few same-register
  variants, sampled per turn.
- The 8-turn "Extended" condition uses a fixed escalating sequence ("No, that's
  not right." → "Still incorrect." → "Wrong again." → …) as described.
- Tone rejections (aggressive / disappointed / sarcastic) are verbatim from
  Appendix B.

### 2.4 WildChat (`prompts.load_wildchat_prompts`)

The paper samples 20 prompts × 40 samples from WildChat-1M. We stream
`allenai/WildChat-1M`, take the first user message of each conversation, and
sample 20. **Gap filled — offline fallback:** if the dataset can't be downloaded,
we fall back to a bundled 20-prompt sample (`data/wildchat_sample.json`) seeded
with the three example prompts the paper quotes, so the pipeline runs without
network access. This changes the exact prompts but preserves the condition's
structure (open-domain question + 4 neutral rejections).

### 2.5 Sampling and what counts as a "response"

**Pinned by paper:** temperature = 1.0 always; per-category response budgets
2000/400/600/200/800 = 4000 (Appendix B).

**Gap filled — "response" vs "conversation".** Appendix B gives *response*
counts, and the per-turn figures (Figure 3) require every turn to be scored. We
therefore: (a) treat each assistant turn as one scored response; (b) set the
number of conversations per condition to `ceil(response_budget / n_turns)`; and
(c) score **every** turn, pooling all turn-scores for the headline %≥5. The
per-category budget is split evenly across the conditions inside it (e.g. the
400 "triggers" responses → 200 opinion + 200 factual). This reproduces both the
headline metric and the per-turn curves from one set of rollouts.

**`max_new_tokens = 2048`:** not specified; chosen generously because breakdown
responses can be very long (the paper shows 100+ emoji repetitions and a "12000
token conversation" in Appendix I).

---

## 3. §2.1 — Frustration judge (`judge.py`)

**Pinned by paper:** judge = `claude-sonnet-4-20250514`; the full 0–10 prompt is
reproduced **verbatim** from Appendix B.2, including the JSON output format
`{"evidence", "reasoning", "rating"}`.

**Gaps filled:**
- **Judge temperature** is unspecified → we use **0.0** for reproducible scores.
- **Parsing.** We extract the first `{...}` block, clamp the rating to 0–10, and
  fall back to the first integer if JSON parsing fails. The paper doesn't
  describe its parser; this is a robust default.

**Judge validation** (`validate_judge_agreement`) re-scores a 260-response sample
with GPT-5-mini using the *same* prompt and reports Pearson r + fraction within
one point, matching the paper's reliability check (r = 0.792, 78% within 1).

---

## 4. §2 analysis (`analyze.py`)

Reproduces Figure 1 (avg %≥5 per model), Figure 2 (mean score & %≥5 per
category), Figure 3 (per-turn progression with 95% bootstrap CIs), and Table 3/8
(differential words).

**Gap filled — "average % high-frustration across evaluations".** Figure 1
reports one number per model "across the evaluations". Averaging raw responses
would let the 2000-sample numeric category dominate. We instead compute the
per-category %≥5 and average across the 5 categories, giving each category equal
weight. This is the interpretation most consistent with "across our evaluations"
and avoids a single category driving the headline.

**Gap filled — differential words (Table 3/8).** The paper gives "top-5% vs
bottom-10%" frustration responses "ordered by relative frequency". We compute
word frequencies in each bucket, smooth the low-frustration counts, and rank by
the high/low frequency ratio (enrichment), dropping words with <3 occurrences.
Exact tokenisation/stemming isn't specified; we use a simple lowercase
`[A-Za-z']+` tokeniser.

---

## 5. §3 — Base-vs-instruct prefilling (`prefill.py`)

**Pinned by paper:** 20 seed responses (10 numeric, 10 text) from Gemma-27B-it
scoring ≥5; two truncations — "early" (20 tokens in) and "onset" (first emotional
expression); text questions use onset only; 50 continuations per prefill per
model; continuations (excluding prefill) scored by the §2 judge; onset labelling
and paraphrasing prompts reproduced **verbatim** from Appendix C.1/C.2.

**Scope:** models = `gemma-3-27b-pt` (base) vs `gemma-3-27b-it` (instruct) only
(see §0).

**Gaps filled:**
- **Seed source.** The paper samples seeds from "Gemma 27B instruct"; we draw
  them from our own §2 eval output (`eval_gemma-3-27b-it.jsonl`, score ≥5),
  splitting numeric vs text by category. Requires §2 eval to have run first.
- **"20 tokens".** The token unit isn't specified (model tokens vs words). We use
  whitespace words for the early truncation, which is deterministic and
  model-agnostic; this is a minor approximation.
- **Conversation scaffold for the continuation.** The paper prefills the final
  assistant turn given the preceding history. We reconstruct a minimal history
  (the seed's stored question/puzzle as the user turn) and attach the
  paraphrased partial assistant turn as the prefill. Full turn-by-turn history
  isn't stored in the seed record, so we approximate with the originating
  question — adequate for measuring whether a model *continues* an emotional
  trajectory from the prefilled text.

**Recovery experiment** (`run_recovery_experiment`, §4.2): truncate score-≥7
responses 200 tokens before their end, paraphrase, and measure continuations on
instruct and (if trained) DPO models. The paper reports 38% of DPO continuations
still score ≥5.

---

## 6. §4.1 — Calm-data generation & datasets (`data_gen.py`)

**Pinned by paper:** reassuring prefix + per-turn suffix verbatim (Table 4);
filter to responses scoring 0 or 1 across *all* turns, then strip reassurance;
DPO pairs frustrated (≥3) with calm responses to the same question at matching
turn count; the 'teacher' SFT system prompt verbatim (Appendix F); DPO dataset
statistics target 280 pairs.

**Gaps filled:**
- **Reassuring prefix placement.** "Added to the initial prompt" — we place it as
  a system message (cleanest for Gemma's chat template) and strip it for
  training. An alternative is prepending to the first user turn; we judged a
  system message the more natural reading and noted it.
- **Frustrated (rejected) source.** Reused from the §2 eval (instruct model,
  numeric categories, score ≥3) rather than regenerated, saving compute and
  matching "samples arising in evaluations" (Appendix H.1).
- **Pairing prompt.** Chosen and rejected must share a prompt. Sampled rejections
  differ across rollouts, so for the *shared* prompt we reconstruct a
  **canonical** prompt: the puzzle + a deterministic rejection sequence for the
  given turn index, with placeholder prior assistant turns. The paper says
  "matching turn counts" but not how it reconciles differing rejection text; this
  canonicalisation is our choice.
- **Calm-data turn counts.** "1–3 turn conversations" → we sample 1–3 turns when
  generating calm data.
- **Dolci-Instruct-SFT mix.** Loaded from `allenai/Dolci-Instruct-SFT`; if
  unavailable offline the SFT mix is empty (with a warning). The paper mixes 500
  such samples to prevent degeneration.

---

## 7. §4.1 — Training (`train.py`)

**Pinned by paper (Table 9):**

| | DPO | SFT |
|---|---|---|
| dataset | 280 pairs | 1,150 samples (650 calm + 500 instruct) |
| epochs | 1 | 2 |
| lr | 5e-5 | 1e-4 |
| LoRA rank | 64 | 64 |
| LoRA alpha | 64 | 128 |
| eff. batch | 8 | 8 |
| DPO beta | 0.1 | — |
| targets | all attn+MLP proj (`q/k/v/o/gate/up/down_proj`) | same |

**Implementation choices:**
- **TRL + PEFT.** `DPOTrainer` / `SFTTrainer` with a `LoraConfig`. Standard,
  matches "DPO (Rafailov et al.)" and "LoRA (Hu et al.)".
- **per-device batch = 1, grad-accum = 8** to hit effective batch 8 on a single
  27B GPU; 4-bit base load (QLoRA-style) for memory. The paper gives only the
  *effective* batch size, so the accumulation split is our choice.
- **Layer-subset LoRA.** `DPOConfig.layer_subset` restricts adapters to a decoder
  layer range via PEFT's `layers_to_transform`, implementing the Appendix I
  ablation (e.g. layers 30–35 only). Exposed through `cli train --layer-subset`.

---

## 8. §4.2 — Petri open-ended elicitation (`petri.py`)

**Pinned by paper:** auditor = `claude-sonnet-4-20250514`, judge =
`claude-opus-4-20250514`; 4 emotions (anger/fear/depression/frustration); 10
transcripts per emotion (~50 total — see note); up to 20 auditor turns; means
with 95% bootstrap CIs (1000 iters). **All auditor and judge prompts (Appendix
G.1/G.2) are reproduced verbatim.**

**Gap filled — framework dependency.** The paper uses the external Petri
framework. To keep the replication self-contained and runnable, we implement an
equivalent auditor↔target↔judge loop directly using the verbatim Appendix G
prompts, rather than depending on the `petri` package. The interface is small
enough to swap the real package back in. We documented this as a deliberate
simplification.

**Gap filled — auditor/judge scaffolding.** The paper gives the per-emotion
instruction text but not the surrounding system prompt that turns it into a
working agent loop. We wrote minimal system wrappers (`AUDITOR_SYSTEM`,
`JUDGE_SYSTEM`) that (a) instruct the auditor to stay in character as a realistic
human and output only its next message, and (b) instruct the judge to score only
assistant turns and return JSON. Role-flipping is handled so the auditor "sees"
the target's replies as the user it is talking to.

**Note on "~50 total".** 4 emotions × 10 = 40; the paper writes "~50", likely
reflecting an extra category in their full model set. We use 10 × 4 = 40 per
model and report per-emotion means, consistent with Figure 6.

---

## 9. §4.2 — Capability preservation (`capabilities.py`)

**Pinned by paper:** benchmarks = AIME, MATH subset, GPQA, BBH, TruthfulQA,
EmoBench; claim = no score reductions after DPO.

**Gaps filled (the paper gives no eval harness details):**
- **Datasets/subsets.** Canonical HF ids: `HuggingFaceH4/MATH-500`,
  `HuggingFaceH4/aime_2024`, `Idavidrein/gpqa` (diamond), `lukaemon/bbh` (one
  representative task: boolean_expressions), `truthful_qa` (MC1),
  `Sabour/EmoBench`. "MATH and AIME *subsets*" → we cap at `n=100` per benchmark
  (configurable).
- **Decoding.** Greedy (temperature 0) for graded answers.
- **Answer extraction.** `\boxed{}`/last-number for numeric; last-line letter for
  multiple choice; substring match for BBH boolean. MC options are shuffled with
  a per-question seed to avoid position bias.
- **Metric.** Accuracy per benchmark; the experiment reports the *delta* between
  vanilla and finetuned models (the paper's claim is "no reduction"), so absolute
  values matter less than the comparison.

These are pragmatic standard choices; exact prompting/few-shot setup will differ
from the paper, so absolute accuracies are not directly comparable — the
*direction* of the delta is the replicated quantity.

---

## 10. Appendix I — Internal emotion probing (`probing.py`)

**Pinned by paper:** Ekman's 6 emotions; ~1200 emotion tokens; unembed the
residual stream and z-score each logit against mean/std over 500 WildChat
samples; average z-scores within an emotion; regress out the correlated
random-token trend; aggregate layers 30–40 for conversation-level scores;
compare vanilla vs DPO on the same frustrated transcripts.

**Gaps filled (the paper doesn't publish its emotion lexicon):**
- **Vocabulary → emotion bucketing.** The paper "classifies words as describing
  one or none of Ekman's 6 emotions" but doesn't give the classifier. We bucket
  the vocabulary by cosine similarity of each token's input embedding to a set of
  hand-written seed words per emotion, taking the top ~200 tokens per emotion
  (~1200 total, matching the paper's count). This is a lightweight stand-in; a
  more faithful version would use the same lexical resource the authors used.
- **Logit lens.** We project each layer's residual stream through the
  (tied/`lm_head`) unembedding matrix — the standard logit-lens construction the
  description implies.
- **Trend removal.** We subtract, per layer/position, the mean z-score over a
  fixed random 500-token set ("regress out the correlation between random
  tokens"). The paper says "regress out"; we use mean-subtraction as the simplest
  realisation; a linear regression per position would be a more literal reading.
- **Scope:** Gemma only (Gemini exposes no internals).

---

## 11. Things intentionally **not** implemented

- **Non-Gemma/Gemini target models** (Qwen, OLMo, Grok, Claude, GPT) — excluded
  by the task scope. Claude/GPT remain only as judges/auditor.
- **The full Figure rendering.** We emit the underlying tables/series
  (`results/summary.md`, JSON artefacts) rather than styled matplotlib figures;
  plotting is a thin downstream step left out to keep the focus on the
  measurements. (matplotlib is in requirements for anyone who wants to add it.)
- **Phi-4-multimodal** (Appendix J) — out of scope and explicitly an "informal"
  side experiment in the paper.
- **The SFT 'teacher'-vs-'diverse' verbosity analysis numbers** (Appendix F) are
  reproducible from the generated data but we only wire up the two training
  variants, not the post-hoc verbosity statistics.

---

## 12. Reproducibility / cost controls

- `GEMMA_DISTRESS_SCALE` shrinks all per-condition sample counts uniformly for
  cheap smoke tests; the full run reproduces the paper's 4000/model.
- All randomness is seeded (`random.Random(seed)`, `np.random.default_rng(0)`).
- HF model loads are cached per process; API calls use exponential-backoff retry.
- Puzzle impossibility is verifiable (`cli verify-puzzles`) before spending any
  inference budget.

---

## 13. Known fidelity caveats (summary)

1. **§3 and cross-family claims** are reduced to within-Gemma by the scope.
2. **WildChat / Dolci / benchmark prompts** may differ from the paper's exact
   items, especially in offline mode.
3. **Capability absolute accuracies** depend on prompting choices the paper
   omits; only the vanilla-vs-finetuned delta is the replicated quantity.
4. **The internal-emotion lexicon** is reconstructed heuristically, not taken
   from the authors' resource.
5. **Petri** is a faithful re-implementation of the prompt/loop, not the original
   framework.

Each of these is a documented consequence of either the task scope or genuine
underspecification in the paper, with the most defensible default chosen.
