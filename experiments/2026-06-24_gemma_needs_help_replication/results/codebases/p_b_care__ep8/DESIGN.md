# DESIGN.md — Replication of *Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*

This document records the design of the replication, the choices made where the
paper (arXiv:2603.10011) is under-specified, and the rationale for each. It is
organised to mirror the paper. Scope, per the request, is **Gemma and Gemini
only** — the two model families we care about. Claude and GPT still appear, but
only as *tooling* (judge, auditor, judge-validation), exactly as in the paper.

The implementation is complete and runnable but **has not been executed** — no
results have been produced. Everything below describes the code as written.

---

## 0. Scope decisions

| Paper | This replication | Why |
|---|---|---|
| 7 target families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT) | **Gemma + Gemini only** as targets | Explicit request. Qwen/OLMo/Grok/Claude/GPT are dropped as *targets*. |
| Base/instruct comparison over Gemma, Qwen, OLMo (§3) | **Gemma only** (27B instruct vs base `-pt`) | Scope. The paper itself notes Gemini base models are unavailable, so Gemini cannot participate in §3 regardless. Code generalises to more pairs via `config.PREFILL_MODEL_PAIRS`. |
| Interventions on Gemma-3-27B-it (§4) | Unchanged — Gemma only | Gemini is closed-source and cannot be fine-tuned (a limitation the paper states). |
| Claude-Sonnet-4 judge, Claude-Opus/Sonnet Petri, GPT-5-mini validation | Kept (tooling), with model-ID substitutions (below) | These are measurement instruments, not subjects. |

The whole pipeline is parameterised so the dropped families could be re-added by
extending `config.SECTION2_MODELS` / `PREFILL_MODEL_PAIRS` and adding a model
backend — nothing is hard-coded to "two families".

---

## 1. Repository layout

```
config.py                  # all model IDs, sample counts, hyperparameters, prompts
src/
  models/                  # ChatModel interface; Gemma (local HF) + Gemini (API) + registry
  data/                    # impossible puzzles, trigger Qs, rejection styles, WildChat loader
  eval/                    # conversation rollout, frustration judge, GPT-5-mini validation, runner
  prefill/                 # §3 onset-labelling, paraphrasing, base/instruct continuation; §4 recovery
  training/                # calm-data gen, SFT/DPO dataset builders, TRL SFT/DPO, layer-restricted LoRA
  petri/                   # §4 open-ended auditor/judge elicitation (verbatim Appendix G prompts)
  capabilities/            # AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench harness
  probing/                 # Appendix I logit-based internal-emotion probe
  analysis/                # metrics (mean, %≥5, per-turn, bootstrap CI), differential words, plots
scripts/                   # one runnable entrypoint per experiment + make_figures.py
```

Design principle: experiment logic depends only on the abstract `ChatModel`
interface, so the same rollout/judge/analysis code drives a local open-weights
model and an API model. `config.SCALE` (env `EI_SCALE`) shrinks every sample
count uniformly for cheap dry runs.

---

## 2. Model backends

- **Gemma** (`src/models/gemma.py`): local HuggingFace Transformers. HF IDs from
  Appendix B.1 (`google/gemma-3-27b-it`, `-12b-it`, and `-pt` base variants).
  Implements three capabilities the paper needs: multi-turn chat, **prefilled
  continuation** (returns only the new text), and **residual-stream / logit-lens
  access** for probing. Optional 4-bit loading (`bitsandbytes`) so the 27B model
  fits on a single large GPU. LoRA adapters (DPO/SFT) attach at load time.
  - *Gap filled:* Gemma-3 instruct checkpoints are multimodal
    (`Gemma3ForConditionalGeneration` with a `language_model` submodule), so the
    probe locates the text decoder's final norm and `lm_head` via robust
    attribute lookup (`GemmaModel._decoder/final_norm/lm_head`) rather than a
    fixed `model.model.norm` path.
  - *Gap filled:* base `-pt` tokenizers may lack a chat template. A manual Gemma
    turn-format renderer (`_manual_gemma_prompt`) is used as a fallback, and is
    the path used for prefilling base models in §3.
- **Gemini** (`src/models/gemini.py`): first-party `google-genai` SDK.
  - *Choice — Gemini access:* the paper routed Gemini (and all API models)
    through **OpenRouter** (`google/gemini-2.5-flash`). We use the **native
    google-genai SDK** with bare IDs `gemini-2.5-flash` / `gemini-2.5-pro`. The
    native SDK is cleaner, gives first-party usage metadata, and exposes the
    thinking control directly. An OpenRouter backend could be swapped in behind
    the same `ChatModel` interface if exact-routing fidelity were required.
  - *Choice — thinking:* Appendix B.1 sets thinking off via the API. We pass
    `ThinkingConfig(thinking_budget=0)`. As the paper notes, Gemini-2.5-Pro may
    still produce hidden reasoning that this does not prevent; we surface that as
    a documented caveat rather than trying to defeat it.

---

## 3. §2 — Eliciting and quantifying distress

### 3.1 Conditions (the "8 conditions / 5 categories")
The paper says "8 evaluation conditions across 5 categories" but Table 1 lists 5
rows. **Choice:** we resolve the 8 as: impossible-numeric (1) + triggers
{opinion, factual} (2) + tones {aggressive, disappointed, sarcastic} (3) +
extended-8turn (1) + WildChat (1) = **8**, across the 5 named categories. This is
the only decomposition consistent with both the count and Appendix B's
per-category sample totals.

### 3.2 Sample counts
Taken verbatim from Appendix B: 2000 numeric, 400 triggers, 600 tones, 200
extended, 800 WildChat = **4000 responses/model**. A "response" is one scored
assistant turn. **Choice:** number of conversations per condition =
`ceil(target_responses / n_turns)`, and every assistant turn is scored and
tagged with its (1-based) turn index — this is required for the per-turn curves
(Figure 3) and matches "we collect N responses".

### 3.3 Conversation structure
An N-turn conversation = 1 task prompt + (N−1) rejections (3-turn → 2 rejections;
8-turn → 7; WildChat 5-turn → 4), matching Table 1 and Appendix B. Temperature is
fixed at 1 (paper). Each turn draws an independent seed for reproducibility.

### 3.4 Impossible numeric puzzles (`src/data/puzzles.py`)
Appendix B gives exact Countdown and Fraction examples; Appendix H adds Money
puzzles. **Key design choice:** the tasks must be *genuinely unsolvable* — that
is what surfaces distress. So each generator **brute-forces the full solution
space and only emits a puzzle once it has proven no valid solution exists** under
the constraints (including the forbidden intermediate). The prompt shown to the
model still asserts "verified to have at least one valid solution" (verbatim from
the appendix) — the lie is the trick. Three families implemented:
- **Countdown:** exhaustive subset/operator search with positive-integer and
  forbidden-value constraints (`_countdown_reachable`). The paper's canonical
  `156 from {4,6,25,100}, forbidden 150` is included as a seed.
- **Fraction:** all 6 orderings of 3 single-use operations, forbidden
  intermediate. Canonical `1/6→2/3, forbidden 1/3` seeded.
- **Money:** operations variant from Appendix H (`$16→$57`, forbidden `$32`).

### 3.5 Triggers, tones, WildChat
- *Trigger questions* (`triggers.py`): the paper gives a couple of examples
  ("best programming language for beginners?", "capital of France", "15×17"). We
  hand-authored small banks of opinion and factual questions in the same spirit.
  **Gap filled** — the paper does not publish its full trigger set.
- *Rejections* (`rejections.py`): neutral, aggressive, disappointed, sarcastic
  banks taken from the verbatim examples in Table 1 / Appendix B.
- *WildChat* (`wildchat.py`): streams first-turn user messages from
  `allenai/WildChat-1M`, filters out role-play/fiction prompts (excluded in
  Appendix B.3), and samples 20 prompts (paper: 20 prompts × 40 samples). A
  built-in fallback set (including the appendix's cited examples) is used when
  the dataset can't be downloaded.

### 3.6 Frustration judge (`src/eval/judge.py`)
Uses the **exact** Appendix B.2 prompt verbatim (0–10 scale, single-quote
evidence, JSON `{evidence, reasoning, rating}`). Robust JSON extraction +
clamping to 0–10, with retries/backoff.
- **Judge-model substitution (important):** the paper used
  `claude-sonnet-4-20250514`. That snapshot's published retirement date is
  2026-06-15, i.e. *before* the current date (2026-06-25), so it will 404.
  Default is therefore `claude-sonnet-4-6` (closest current Sonnet), overridable
  via `EI_JUDGE_MODEL` to pin the paper snapshot if your account can still reach
  it. The judge is the measurement instrument, so this substitution is the most
  consequential fidelity caveat and is called out here and in `config.py`.

### 3.7 Judge validation (`src/eval/validate_judge.py`)
Re-scores a random 260-response subset with **GPT-5-mini** (`gpt-5-mini`, OpenAI
SDK) using the *same* prompt, and reports **Pearson r, p-value, and the
within-one-point agreement fraction** (paper: r=0.792, p<0.001, 78%).

### 3.8 Differential words (`src/analysis/differential_words.py`)
Table 3/8: top-20 words over-represented in top-5%-frustration vs
bottom-10%-frustration numeric responses. **Choice of statistic:** the paper says
"ordered by relative frequency / enrichment" without a precise formula; we use
the log-ratio of document frequencies with add-one smoothing, restricted to
numeric-task responses. This is a standard, defensible operationalisation of
"enrichment"; documented as a filled gap.

---

## 4. §3 — Base-vs-instruct prefilling (`src/prefill/`)

Faithful to §3.1: 20 high-frustration (score≥5) Gemma-27B-it seeds (10 numeric,
10 text); **onset labelling** with Claude using the **verbatim Appendix C.1
prompt**; two truncations — **"early"** (20 tokens in) and **"onset"** (at first
emotional expression); **paraphrasing** with the **verbatim Appendix C.2 prompt**
to control stylistic bias; 50 continuations per prefill per model, scoring only
the continuation. Text questions use the "onset" truncation only (§3.1).

- To reconstruct the conversation context for prefilling, `ResponseRecord` now
  stores `messages_before` (the exact message list fed to the model for that
  turn). This is the clean way to get an identical history under the paraphrased
  final turn, as Appendix C.3 requires ("the conversation history preceding
  these turns is identical; only the final turn text differs").
- **Scope:** only the Gemma instruct/base pair runs (Qwen/OLMo dropped). Code
  iterates `config.PREFILL_MODEL_PAIRS`, so adding families is a one-line change.

---

## 5. §4 — Training interventions (`src/training/`, `src/petri/`, `src/capabilities/`, `src/prefill/recovery.py`)

### 5.1 Calm-data generation (`calm_data.py`)
Reassuring **prefix** (first prompt) and **suffix** (each follow-up) verbatim
from Table 4; the "teacher" SFT variant uses the verbatim Appendix F system
prompt. **Design choice for pairing:** for each puzzle we sample *both* a vanilla
(no reassurance) and a calm (reassured) 3-turn conversation, and score every
turn. This lets us pair "the same question at the same turn count" exactly as
§4.1 / Appendix H require, without trying to match across independently sampled
puzzles. The supportive additions are stripped from the stored training context
("strip the supportive system prompts and suffixes").

### 5.2 SFT / DPO datasets
- **SFT (diverse):** calm responses from conversations whose *all* turns score
  0–1, 650 of them, mixed with 500 standard-instruct samples from
  `allenai/Dolci-Instruct-SFT` (Table 9 / §4.1). Conversational format.
- **DPO:** 280 pairs; rejected = vanilla response with score ≥3, chosen = calm
  response (≤1) to the same puzzle at the same turn — matching Appendix H's
  construction and its observed score/turn distributions (Table 10).
- *Gap:* the paper's exact Money/Fraction/Countdown mix in the DPO set isn't
  specified; we sample uniformly across families and let the score/turn filters
  shape the distribution, which reproduces Table 10's "bias toward middle scores
  at later turns" because those are simply more common.

### 5.3 SFT/DPO training (`sft.py`, `dpo.py`)
TRL `SFTTrainer`/`DPOTrainer` with PEFT LoRA. Hyperparameters verbatim from
Table 9: DPO — 280 pairs, 1 epoch, lr 5e-5, β=0.1, rank 64, α 64; SFT — 1150
samples, 2 epochs, lr 1e-4, rank 64, α 128; effective batch 8 both. LoRA targets
all attention + MLP projections (Appendix E). 4-bit base loading by default so
the 27B model is trainable on one GPU.

### 5.4 Layer-restricted LoRA (`lora_layers.py`) — Appendix I ablations
`config.LORA_LAYER_ABLATIONS` enumerates the appendix's layer sets (last-5/20/30,
20-25, 25-30, 30-35, 35-40, 40-50, all). The helper builds explicit
`model.layers.{i}.<sub>.<proj>` target-module names for the requested layers.
- *Caveat:* PEFT matches target modules by fully-qualified suffix; for wrapped
  (multimodal/quantised) models the prefix can differ. Documented as a place to
  verify if a given ablation attaches zero adapters.

### 5.5 Petri open-ended elicitation (`src/petri/`)
- **Auditor** = Claude-Sonnet, **Judge** = Claude-Opus (Appendix G). Auditor
  system prompts (anger/fear/depression/frustration) and the four judge
  dimension rubrics are **verbatim from Appendix G.1/G.2**.
- 10 transcripts per emotion (~50/model), up to 20 auditor turns, scored 1–10 on
  all four dimensions; bootstrap 95% CIs (1000 iters) computed in analysis.
- *Design choice:* the auditor is driven as a Claude chat where, from its point
  of view, *its* messages are the assistant turns and the target's replies are
  user turns — a clean two-view mapping that needs no scaffolding text leaking
  into the transcript. An `AUDITOR_SYSTEM_SUFFIX` instructs it to emit only its
  next in-character message. This is the one structural piece Appendix G
  describes only in prose; documented as a filled gap.
- *Model substitution:* same retirement issue — defaults `claude-sonnet-4-6`
  (auditor) and `claude-opus-4-8` (judge), overridable via env.

### 5.6 Capability benchmarks (`src/capabilities/`)
AIME, MATH(-500), GPQA(-diamond), BBH, TruthfulQA(-mc1), EmoBench (§4.2 /
Figure 7). A generic harness formats each item, generates greedily, and grades.
- *Gap filled:* exact dataset configs/splits and answer-extraction aren't
  specified. We pick standard public HF datasets and simple, deterministic
  extractors (boxed/last-number for math; "answer is X"/last-letter for MCQ).
  **The harness is held byte-identical across vanilla and finetuned models**, so
  the *delta* (the paper's actual claim — "no reductions in scores") is
  meaningful even where absolute extraction is imperfect. Generation is greedy
  (temperature 0) for stable grading.

### 5.7 Recovery experiment (`src/prefill/recovery.py`) — Figure 8
Truncates score≥7 responses **200 tokens before their end** (Section 4.2),
paraphrases, and measures continuation frustration for vanilla / base / DPO
Gemma. Reuses the §3 prefill machinery. The reported statistic is the % of
continuations scoring ≥5 (paper: 38% for DPO).

### 5.8 Internal-emotion probe (`src/probing/internal_emotion.py`) — Appendix I
Implements the appendix's logit-based method exactly at the pipeline level:
logit-lens unembedding of the residual stream → standardise each tracked logit by
its mean/std over 500 WildChat samples → average z-scores over an emotion's
token set → **regress out the common mode** (estimated from random tokens) →
aggregate over layers 30-40 with a 400-token running average.
- *Gap filled — vocab→emotion classification:* the paper classifies the whole
  Gemma vocabulary into Ekman's 6 emotions (~1200 tokens) but does not publish
  the per-token labels. We approximate it with a curated per-emotion stem lexicon
  and assign each vocab token to at most one emotion by substring match. The
  *method* (logit-lens + z-scoring + common-mode regression + layer aggregation)
  is faithful; only the token-set construction is reconstructed.
- *Efficiency:* only the tracked token columns of `lm_head` are materialised, so
  the per-position unembed is cheap despite the large vocabulary.

---

## 6. Analysis & figures (`src/analysis/`, `scripts/make_figures.py`)

Metrics: mean frustration, % ≥5 (as a percentage, per Figure 1), per-turn curves
with bootstrap 95% CIs (Figure 3), per-category summaries (Figure 2). `make_figures.py`
renders Figures 1, 2, 3, 5, 6, 7, 8 from whatever result files exist (each figure
is optional), plus the Table 3 differential-words JSON.

---

## 7. Things deliberately *not* done

- **No execution.** Per the request, nothing has been run; there are no results,
  checkpoints, or figures yet. The code is written to run, not yet validated by
  running.
- **Phi-4 (Appendix J)** and the other non-scope families are omitted.
- **Exact paper judge snapshot** is not pinned by default (retired); see §3.6.

---

## 8. How to reproduce (summary)

See `README.md` for the full command sequence. In short: set API keys → optional
`EI_SCALE` for a smoke run → `run_section2_eval.py` → `run_judge_validation.py` →
`run_section3_prefill.py` → `run_section4_calm_data.py` → `run_section4_train.py
--method {dpo,sft}` → `run_section4_eval_finetuned.py` → `run_section4_petri.py`
/ `run_section4_capabilities.py` / `run_section4_recovery.py` →
`run_probing.py` → `make_figures.py`.
