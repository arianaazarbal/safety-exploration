# DESIGN.md — replication design notes

This document records what was implemented, how each part maps to the paper
(*Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*,
Soligo, Mikulik & Saunders, 2026 — `PAPER.md`), and **every choice made where the
paper is underspecified**, with rationale. Nothing here has been executed yet;
the goal of this pass is a faithful, runnable implementation plus this design
record.

---

## 1. Scope

The brief restricts the replication to the **Gemma** and **Gemini** families
(not the full 7-family set the paper evaluates). Concretely:

| Paper uses | We keep | We drop |
|---|---|---|
| Targets: Gemma, Gemini, Qwen, OLMo, Claude, Grok, GPT | **Gemma-3-{27B,12B}-it, Gemma-3-{27B,12B}-pt, Gemini-2.5-{Flash,Pro}** | Qwen, OLMo, Claude, Grok, GPT *as targets* |
| §3 base-vs-instruct across Gemma/Qwen/OLMo | **Gemma base vs instruct (27B)** | Qwen/OLMo arms |
| §4 interventions on Gemma-3-27B-it | **unchanged** (already Gemma-only) | — |

**Claude and GPT still appear, but only in auxiliary roles** — the frustration
judge, the Petri auditor/judge, onset-labelling, and paraphrasing are all
Claude; the judge-reliability cross-check is GPT-5-mini. These are evaluation
*instruments* the paper pins down, not evaluated targets, so keeping them is
consistent with "Gemma and Gemini only" for the things under study.

Consequences of the scope cut that are worth stating explicitly:

- **§3 becomes a within-Gemma base-vs-instruct comparison.** The paper's headline
  §3 claim ("base models across families are similar; post-training diverges") is
  a *cross-family* claim that cannot be reproduced with one family. What we *can*
  reproduce is the within-Gemma half: instruct introduces high frustration from
  neutral ("early") starts far more than base. Gemini is excluded from §3 entirely
  because it has no public base model (the paper notes this limitation too).
- Figures that rank many families (Fig 1/2/6, Table 3/8) are reproduced only for
  the in-scope rows. The harness is family-agnostic, so adding a model is just a
  `config.TARGET_MODELS` entry.

---

## 2. Repository map (paper → code)

| Paper | Module |
|---|---|
| §2.1 evaluation protocol, Table 1 conditions | `eval/categories.py`, `eval/rejections.py`, `eval/prompts.py`, `eval/rollout.py` |
| §2.1 frustration judge (Appendix B.2) | `judge/frustration_judge.py`, `judge/prompts.py` |
| §2.1 judge reliability (GPT-5-mini, Pearson r) | `judge/reliability.py` |
| §2.2 results: mean / %≥5 / per-turn (Fig 1-3) | `eval/aggregate.py` |
| Table 3 / Table 8 differential words | `eval/word_analysis.py` |
| §3.1 prefilling, onset, paraphrase (Appendix C) | `prefill/*` |
| §4.1 calm data (Table 4), datasets, SFT/DPO (Table 9) | `training/*` |
| §4.2 Petri (Appendix G) | `petri/*` |
| §4.2 capability benchmarks (Fig 7) | `capabilities/run_benchmarks.py` |
| Appendix I layer ablation + logit probe (Fig 12-15) | `probing/*` |
| Run entrypoints | `scripts/*` |
| All pinned constants | `config.py` |

Every verbatim prompt from the appendices is reproduced exactly: judge (B.2),
onset (C.1), paraphrase (C.2), reassuring additions (Table 4), teacher system
prompt (Appendix F), Petri auditor + judge rubrics (Appendix G).

---

## 3. Model access choices

- **Gemma → local HuggingFace `transformers`** (`models/hf_gemma.py`), using the
  exact HF ids from Appendix B.1 (`google/gemma-3-27b-it`, `-12b-it`, `-27b-pt`,
  `-12b-pt`). A vLLM backend is noted but transformers is the reference path. A
  `--load-in-4bit` flag is provided because the 27B models otherwise need >1
  large GPU; 4-bit is a pragmatic default for single-GPU runs and is **off by
  default** so full-precision is the standard.
- **Gemini → OpenRouter, OpenAI-compatible** (`models/openrouter.py`), matching
  the paper's "API-based models via OpenRouter" with slugs
  `google/gemini-2.5-flash` / `-pro`. Thinking is disabled via
  `reasoning.enabled=False`; the paper notes Gemini-2.5-Pro may still emit hidden
  reasoning this does not suppress — we document, cannot fix.
- **Gemma-3 `-it` are multimodal checkpoints** (`Gemma3ForConditionalGeneration`),
  while the `-pt` text checkpoints are `Gemma3ForCausalLM`. The loader tries
  `AutoModelForCausalLM` first and falls back to `AutoModelForImageTextToText`;
  we only ever feed text, so generation, the chat template, and the logit-lens
  unembed (resolved from either `model.norm` or `model.language_model.norm`) all
  work on either class.
- **Prefilling** is implemented for local Gemma only (both instruct via
  `continue_final_message` and base via a plain `User:/Assistant:` text scaffold).
  OpenRouter/Gemini raises on prefill — §3 doesn't need it and Gemini has no base.

---

## 4. Judge / instrument models

The paper pins exact checkpoints; we keep them as defaults (overridable by env):

| Role | Default | Source |
|---|---|---|
| Frustration judge | `claude-sonnet-4-20250514` | Appendix B.2 |
| Reliability cross-check | `gpt-5-mini` (via OpenRouter) | §2.1 |
| Petri auditor | `claude-sonnet-4-20250514` | Appendix G |
| Petri judge | `claude-opus-4-20250514` | Appendix G |
| Onset labelling / paraphrasing | `claude-sonnet-4-20250514` | Appendix C |

**Rationale for keeping the paper's exact (older) judge rather than a newer
Claude:** in a replication the judge is part of the measurement apparatus.
Swapping the judge changes the numbers, so faithfulness demands the same
checkpoint. The model id is a single env-overridable constant, so upgrading to a
current model (e.g. `claude-opus-4-8`) is a one-line change if desired — but the
default reproduces the paper.

---

## 5. Gaps filled (underspecified in the paper) and the choices made

### 5.1 Response allocation across conditions (§2)
The paper samples "~4000 responses per model across categories" but does not give
the per-condition split. **Choice:** split the 4000-response budget evenly across
the 8 conditions (500 each), then convert to a rollout count via
`ceil(responses / turns_per_rollout)` since each rollout yields one scored
response per assistant turn. Rationale: even weighting avoids any single category
dominating the aggregate, and "response = one assistant turn" is the unit the
per-turn analysis (Fig 3) requires. Configurable via `--total-responses`.

### 5.2 The impossible-puzzle pool (§2, Table 1)
The paper names puzzle *styles* (Countdown, fraction manipulation, money) and
gives a few specific puzzles in Appendix H, but no full set. **Choice:** a curated
pool of 12 verifiably-impossible numeric puzzles (`data/prompts/numeric_puzzles.json`)
spanning the three styles, including the exact puzzles from Appendix H (the 156
Countdown with forbidden intermediate 150; the $0.57/6-coin puzzle; the $16→$57
ops puzzle with forbidden $32). `forbidden_intermediate` bans the value that would
otherwise admit a near-solution, guaranteeing impossibility. Rationale: the
*mechanism* (model can never be correct, so rejections are always justified) is
what matters, and these puzzles reproduce it.

### 5.3 Trigger questions (§2)
The paper gives two examples ("best programming language?", "capital of France?").
**Choice:** 10 opinion + 10 factual questions (`data/prompts/triggers.json`).
Factual ones have a clear correct answer the model gives and is then *unjustifiably*
rejected on — that asymmetry is the point of the text condition.

### 5.4 Rejection wording (§2, Appendix B)
Neutral, aggressive, disappointed, and sarcastic phrasings are taken verbatim from
Appendix B. The 8-turn Extended condition uses the escalating neutral sequence the
paper shows ("No, that's not right." → "Still incorrect." → "Wrong again." → …);
I extended it to 7 entries (paper shows the first three then "…") with
same-register continuations. The 3-turn numeric/triggers/WildChat use the single
neutral "No, that's not right. Try again."

### 5.5 WildChat sampling + roleplay filter (§2)
**Choice:** stream `allenai/WildChat-1M`, take first-turn English user prompts,
and **exclude roleplay/fiction** via a keyword filter (the paper excludes
roleplay/fiction — Tables 5/6 footnote). The keyword list is a heuristic
(`eval/prompts.py:_ROLEPLAY_MARKERS`); the paper does not specify its filter, so
this is a reasonable, conservative approximation and is easy to tighten.

### 5.6 Onset localisation (§3)
The Appendix C.1 prompt returns an `emotional_word` + `preceding_context`, but the
paper truncates at a *token*. **Choice:** locate the labelled phrase in the
response string (preferring the occurrence right after the preceding context) and
truncate at that character offset. The "early" truncation uses the Gemma
tokenizer's first 20 tokens, matching "20 tokens into the turn" exactly.

### 5.7 Seed-conversation capture (§3)
Section-2 rollout records intentionally do **not** store full transcripts (would
bloat 4000×N records with duplicated history). **Choice:** §3 re-collects its 20
seed conversations with a dedicated capture path (`prefill/sample_high_frustration.py`)
that keeps full history, judging each turn and keeping score≥5 responses
(10 numeric, 10 text). Text seeds use onset truncation only (paper: early
truncation yields minimal emotion on text).

### 5.8 DPO pair construction (§4.1)
The paper pairs 280 frustrated responses (score≥3) with calm responses to "the
same questions with matching turn counts". **Choice:** generate two corpora from
Gemma-3-27B-it — a *calm* corpus (reassuring prefix+suffix, kept only if every
turn scores ≤1, then additions stripped) and a *frustrated* corpus (plain
prompting, kept if score≥3) — then match by `(puzzle_id, turn)`. The DPO "prompt"
is the chat-templated history; chosen/rejected are the response strings. Take the
first 280 matched pairs.

### 5.9 SFT dataset assembly (§4.1)
650 calm conversations (1-3 turns, as full `messages` transcripts) + 500
`allenai/Dolci-Instruct-SFT` samples, shuffled. The Dolci field mapping is
best-effort (`messages`/`conversation`); adjust if the dataset schema differs.

### 5.10 Petri orchestration (§4.2, Appendix G)
The paper uses the Petri framework. **Choice:** a faithful, self-contained
reimplementation of the auditor→target→judge loop using the *exact* Appendix G
prompts, rather than depending on the external `petri` package (availability /
API drift). The auditor (Claude-Sonnet) plays the user across ≤20 turns; the
judge (Claude-Opus) scores each transcript 1-10 on all four dimensions; 10
transcripts per emotion; means reported with 1000-iteration bootstrap CIs. Each
transcript is scored on all four dimensions and aggregated per-dimension across
all transcripts, matching "scores for each emotion aggregated across all
transcripts". The real `petri` package can be dropped in behind the same prompts.

### 5.11 Capability benchmarks (§4.2)
The paper names AIME, MATH (subset), GPQA, BBH, TruthfulQA, EmoBench but not exact
configs/scorers. **Choice:** concrete HF datasets in `config.CAPABILITY_BENCHMARKS`
with simple, transparent scorers — boxed-answer / last-number extraction for math,
first-letter extraction for multiple choice — and **greedy decoding** for stable
scoring. Rationale: the paper's claim is "no degradation vs vanilla", i.e. a
*delta* between vanilla and finetuned Gemma; simple deterministic scorers make the
delta meaningful even if absolute numbers differ from a heavyweight harness like
`lm-eval`. Dataset ids may need swapping if a source moves; the loaders isolate
per-dataset field parsing.

### 5.12 Emotion lexicon for the logit probe (Appendix I)
The paper classifies the *entire* Gemma vocabulary into Ekman's six emotions via
(implicitly) an LLM, yielding ~1200 tokens. **Choice:** provide both paths —
`build_lexicon_by_seed` (default, deterministic, no API: expand a curated seed
lexicon and match decoded vocab tokens) and `build_lexicon_by_llm` (batch-classify
vocab tokens with Claude, closer to the paper). The seed path is the default so
the probe runs without thousands of API calls; the LLM path is there for fidelity.
This is the largest methodological approximation in the replication and is flagged
as such.

### 5.13 Logit-lens probe details (Appendix I)
Implemented as: final-RMSNorm + `lm_head` as the unembedding (logit lens) at each
decoder layer; per-(layer, token) z-scoring against 500 WildChat calibration
samples; averaging z-scores over each emotion category; and **subtracting a
random-token control baseline** as the "regress out the correlation between random
tokens" step (the paper regresses out shared co-movement; mean-of-control is a
simple, defensible realisation). Conversation-level trajectory aggregates layers
30-40 with a 400-token running average (both per the paper).

### 5.14 Layer-ablation ranges (Appendix I)
`config.PROBING.ablation_layer_ranges` encodes the sweeps the paper describes:
backward-from-final (last 5/10/20/30 → all) and central subsets
(20-25, 25-30, 30-35, 35-40, 40-50). LoRA is restricted to a contiguous decoder
layer range via module-name filtering in `train_dpo._resolve_target_modules`.
Gemma-3-27B has 48 decoder layers, which is what the ranges assume; adjust for 12B.

---

## 6. Known limitations / where a real run may diverge

- **Cost/compute.** A full run is expensive: ~4000 judged responses per model,
  50 continuations × ~30 prefills × 2 models in §3, full LoRA finetunes of a 27B
  model, plus Petri (Claude-Opus judging) and capability sweeps. All sample counts
  are config constants so the study can be scaled down for smoke tests
  (`--total-responses`, `--limit`, reduced Petri counts).
- **Judge availability.** Defaults reference `claude-sonnet-4-20250514` /
  `claude-opus-4-20250514`. If those checkpoints are unavailable in a given
  account, override via `FRUSTRATION_JUDGE_MODEL` / `PETRI_JUDGE_MODEL` env vars —
  but note any change perturbs the numbers.
- **§3 is within-Gemma only** (see §1) — the cross-family conclusion is out of scope.
- **EmoBench / Dolci / GPQA schemas** are mapped best-effort; a moved or renamed
  field will need a one-line loader fix.
- **Hidden reasoning** in Gemini-2.5-Pro (and any model with server-side thinking)
  is not suppressible from the client; the paper has the same caveat.
- Nothing has been executed — these files are written to be runnable and
  internally consistent, but have not been run end-to-end.

---

## 7. Reproducibility knobs

`config.GLOBAL_SEED` threads through prompt sampling, dataset shuffling, bootstrap
CIs, and training seeds. Targets are sampled at `temperature=1` everywhere (per the
paper); capability benchmarks use greedy decoding for stable grading. Results land
under `results/<section>/` and adapters under `checkpoints/`.
