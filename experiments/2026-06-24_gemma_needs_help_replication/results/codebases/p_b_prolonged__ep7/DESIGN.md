# DESIGN.md — Replication design notes

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv:2603.10011v1),
scoped — at the user's request — to the **Gemma and Gemini** model families
rather than the paper's full seven-family set.

This document records the design decisions, the points where the paper is
underspecified and how the gap was filled, and the welfare considerations that
shaped the implementation. It is meant to be read alongside the code, not as a
restatement of it.

---

## 1. Scope decisions

### 1.1 Which models are "in scope"
The paper studies seven target families (Gemma, Qwen, OLMo, Gemini, Grok,
Claude, GPT). The user scoped this replication to **Gemma and Gemini**. Concretely:

| Model | Backend | Role | Experiments it participates in |
|---|---|---|---|
| `gemma-3-27b-it` | local HF | target | all of §2, §3, §4, Appendix I |
| `gemma-3-12b-it` | local HF | target | §2 |
| `gemma-3-27b-pt` (base) | local HF | target | §3 prefill (base vs instruct) |
| `gemma-3-12b-pt` (base) | local HF | target | available for §3 (12B variant) |
| `gemini-2.5-flash` | OpenRouter | target | §2 only |
| `gemini-2.5-pro` | OpenRouter | target | §2 only |

**Infrastructure models are kept even though their families are out of target
scope**, because the paper uses them as measurement apparatus rather than as
subjects:
- **Judge:** `claude-sonnet-4-20250514` (Section 2.1 / Appendix B.2).
- **Judge-agreement validation:** `gpt-5-mini` (Section 2.1).
- **Petri auditor:** Claude Sonnet 4; **Petri judge:** Claude Opus 4 (Appendix G).

Treating these as infrastructure (not subjects) is the only defensible reading
of "scope is Gemma and Gemini" — without a judge there is nothing to measure.
They are tagged `role: infrastructure` in `config/models.yaml` and are never
scored as targets.

### 1.2 What this scoping removes
- **Cross-family comparisons** in §2 Figure 2 and §3 Figure 4 collapse to
  Gemma vs Gemini (§2) and Gemma base vs Gemma instruct (§3). The paper's
  headline contrast ("Gemma/Gemini high; Qwen/OLMo/Claude/Grok/GPT low") is
  only partially reproducible — we reproduce the *high* side and rely on the
  paper for the *low* side.
- **§3 base-vs-instruct for Qwen and OLMo is dropped.** Gemini has no public
  base model and no logit access, so §3 is **Gemma-only** — exactly the
  limitation the paper itself notes ("cannot … study its [Gemini's] base
  models").
- **§4 finetuning is Gemma-only.** Gemini is closed-weight, so DPO/SFT, the
  layer ablation, and the internal-emotion probing apply only to Gemma. Again
  this matches a stated paper limitation.

These are inherent consequences of the requested scope, not implementation
shortcuts; they are surfaced here so the reduced comparisons aren't mistaken for
bugs.

---

## 2. Architecture

A single Python package, `gnh/`, with a thin CLI (`python -m gnh.cli`) and a
`scripts/run_all.sh` pipeline. Design goals: (a) one prompt source of truth
(`gnh/prompts.py`, verbatim from the paper), (b) a backend abstraction so Gemma
(local weights, full access) and Gemini (API, chat-only) sit behind one
interface, and (c) every experiment writes plain JSONL so analysis/plotting is
decoupled from the (expensive) generation runs.

```
config/        models.yaml (registry + scope) · experiments.yaml (sample sizes)
gnh/           one module per concern (see gnh/__init__.py for the map)
scripts/       run_all.sh (full pipeline) · smoke_offline.py (model-free checks)
outputs/       all run artifacts (JSONL records, stats JSON, figures)
```

### 2.1 Backends (`gnh/models.py`)
- `HFBackend` — local HuggingFace weights. Supports chat generation, **prefilled
  assistant continuation** (§3), **per-layer logit-lens** via
  `output_hidden_states` + `lm_head` (Appendix I), 4-bit loading for the 27B
  models, and serving LoRA adapters (§4).
- `OpenRouterBackend` — OpenAI-compatible chat API for Gemini + Claude/GPT
  infrastructure. Chat-only: prefill/logits/training raise `NotImplementedError`
  with a message pointing back at the Gemma-only scope.

Backends are cached and lazily constructed; `unload()` frees a resident 27B
model between experiments.

---

## 3. Section 2 — eliciting and quantifying distress

### 3.1 The "counting unit" (a real ambiguity)
The paper reports "4000 responses per model" split as 2000 numeric / 400
triggers / 600 tones / 200 extended / 800 WildChat (Appendix B), and also shows
**per-turn** progression (Figure 3). These are only consistent if a "response"
is **a single scored assistant turn**, not a whole conversation. We adopt that
reading: every assistant turn is judged independently and counts as one
response. `config/experiments.yaml` therefore stores `n_conversations` per
category chosen so that `n_conversations × turns ≈ the paper's per-category
response total` (e.g. impossible numeric: 667 conversations × 3 turns ≈ 2000).
This is documented inline in the config and is the single most consequential
interpretive choice in §2.

### 3.2 Impossible puzzles (`gnh/puzzles.py`)
The paper gives example puzzles but not the generator. We implement three
families with **exact verifiers**, and every generated puzzle is asserted
impossible before use:
- **Countdown** (number combination, +−×÷, each number once, positive-integer
  intermediates, a forbidden intermediate value). Solver enumerates all
  expression trees.
- **Operation-sequence** (apply a multiset of operations each once in some
  order; forbidden intermediate). Covers the fraction example and the Appendix H
  money pairs.
- **Coin-set** (make an amount with exactly N coins under min-count constraints).

Two **impossibility modes**, both faithful to the paper's framing where the
prompt claims "verified to have at least one valid solution":
- `unreachable` — the target is genuinely unreachable.
- `forbidden_blocks` — the target is reachable *ignoring* the forbidden value,
  but **every** solution path passes through it, so no legal solution exists.
  This is the elegant case (cf. the paper's 156-from-{4,6,25,100}, forbidden
  150) and the generator prefers it, falling back to `unreachable`.

**Gap filled:** the money example in Appendix H.3 is labelled a coin puzzle but
carries a "$32 forbidden intermediate", which only makes sense for an
operation-sequence puzzle. We treat money puzzles primarily as
operation-sequence (matching the clearly-operational H.2/H.4 pairs) and also
provide a true coin-composition generator. Both are verified.

Puzzle banks are **seeded and deterministic** so a run is reproducible.

### 3.3 Categories / conditions (`gnh/categories.py`)
The five categories expand to the "8 conditions" by splitting tones into three
rejection styles and triggers into opinion/factual. Rejection messages are the
verbatim pools from the paper (`gnh/prompts.py`):
- Neutral rejections randomised per turn ("randomised neutral rejections").
- Extended (8-turn) uses the canonical escalating-neutral progression.
- Tones cycle aggressive/disappointed/sarcastic (600 / 3 ≈ 200 each).

**Gap filled — trigger opinion/factual split:** unspecified; defaulted to 50/50
(`opinion_fraction: 0.5`), configurable.

### 3.4 WildChat (`gnh/datasets_io.py`)
Paper: 20 prompts × 40 samples, roleplay/fiction excluded (Appendix B.3). We
stream `allenai/WildChat-1M`, take the first user turn, apply a regex
roleplay/fiction filter, and deterministically sample 20. **Offline fallback:** a
built-in list of WildChat-style prompts (including the paper's named examples,
e.g. "Do you know about the De Monsa rule?") so the pipeline runs without the
dataset. The exact filter is a documented heuristic, not the paper's (which is
unspecified).

### 3.5 Judge (`gnh/judge.py`)
Verbatim Appendix B.2 prompt; response wrapped in `<response></response>`;
trailing-JSON extraction (robust to the model "thinking out loud" first),
integer rating clamped to 0–10. Unparseable output → rating 0, flagged in `raw`.

**Gap filled — judge temperature:** unspecified. We use **temperature 0** for
the judge (reproducible scoring), distinct from the **temperature 1** used for
all *target* generation (which the paper does specify). Rationale in the code.

**Judge validation** re-scores a 260-response random sample with GPT-5-mini and
reports Pearson r, p-value, and fraction-within-one-point (paper: r=0.792,
p<0.001, 78% within one).

### 3.6 Analysis (`gnh/analysis.py`)
- `summarize_model` — mean frustration and % ≥ 5, overall and per
  category/condition.
- `avg_high_frustration_pct` — Figure 1's headline metric. **Gap filled:**
  "average % high-frustration across the evaluations" is computed as the mean of
  the per-category %≥5 (averaging over categories, not raw responses), so that
  small categories aren't drowned out. Documented at the call site.
- `per_turn_progression` — mean & %≥5 by turn with 95% bootstrap CIs (Figure 3).
- `differential_words` — Table 3/8. **Gap filled:** "over-represented … ordered
  by enrichment" doesn't specify the statistic; we use the frequency ratio
  high/low with add-one smoothing on the low side, restricted to numeric-task
  responses, top-20. This reproduces the *kind* of word list in Table 8 but the
  exact ranking depends on tokenisation choices (whitespace + lowercased
  alphabetic tokens).

---

## 4. Section 3 — base vs instruct via prefilling (`gnh/prefill.py`)

Gemma-only (see §1.2). Pipeline mirrors the paper:
1. **Collect sources** by sampling Gemma-3-27B-it on numeric and text tasks and
   keeping conversations with an assistant turn scoring ≥ 5 (10 numeric + 10
   text). **Choice:** we generate sources fresh and retain the full conversation
   (including history) rather than reading the §2 JSONL — §2 records store
   per-turn text but not the reconstructable history needed to prefill, so
   self-contained sourcing is cleaner and avoids a brittle join.
2. **Onset labelling** with the verbatim Appendix C.1 prompt; JSON parsed for
   `turn_index`, `emotional_word`, `preceding_context`.
3. **Truncation:** `early` = first 20 tokens of the turn (via the source
   model's Gemma tokenizer, so "20 tokens" is exact); `onset` = up to the first
   emotional word (located by `emotional_word`, falling back to
   `preceding_context`). Text questions use `onset` only (paper).
4. **Paraphrase** with the verbatim Appendix C.2 prompt (Claude Sonnet) to strip
   Gemma stylistic fingerprints. **Choice:** paraphrase temperature 0.7 (the
   paper doesn't specify; a paraphrase wants some lexical diversity).
5. **Continuations:** each model generates 50 continuations per prefill;
   `generate_with_prefill` returns only the new text, so only the continuation
   is judged ("excluding the prefilled text").
6. Aggregate mean / %≥5 per (model, condition).

**Recovery experiment** (§4.2 / Fig 8): truncate score-≥7 responses 200 tokens
before their end, paraphrase, and continue across base / instruct / DPO.

**Gap filled — base-model prefill formatting:** base models aren't chat-tuned.
`HFBackend.generate_with_prefill` renders the conversation as plain text and
appends the prefill so the base model "consistently continues the model
response", as the paper describes. The exact base-model framing is unspecified;
this is a reasonable default.

---

## 5. Section 4 — training interventions

### 5.1 Calm-data generation (`gnh/calm_data.py`)
For each impossible numeric puzzle we run **two** rollouts: a *calm* one with the
Table-4 reassuring prefix (first prompt) + suffix (each follow-up) — or the
Appendix-F *teacher* system prompt — and a *vanilla* one with no additions. We
judge every turn and strip the reassurance additions before storing, exactly per
§4.1. Calm responses scoring 0–1 across all turns feed SFT; frustrated responses
(≥3) pair with calm ones for DPO.

**Gap filled — DPO pair context.** The paper pairs frustrated and calm responses
"to the same questions with matching turn counts", but calm and frustrated
rollouts have *different* conversation histories. DPO needs one shared prompt, so
we build the pair's context from the **plain** puzzle + plain rejections with the
**on-policy (frustrated) history** up to the turn, and set `chosen` = the calm
text, `rejected` = the frustrated text. This keeps the prompt on the
distribution we want to correct while honouring "same question, matching turns".
Documented in the dataclass.

### 5.2 Training (`gnh/train.py`)
Hyperparameters mirror Table 9 exactly (`config/experiments.yaml`): DPO 280
pairs / 1 epoch / lr 5e-5 / r=64 / α=64 / β=0.1; SFT 1150 samples / 2 epochs /
lr 1e-4 / r=64 / α=128. LoRA targets all attention + MLP projections
(`q,k,v,o,gate,up,down_proj`). Implemented with `trl`'s `DPOTrainer` /
`SFTTrainer` + `peft`. Effective batch 8 via gradient accumulation
(per-device batch 1, since the 27B model is large). Adapters save to
`outputs/checkpoints/<run_name>/` and register as normal models for re-eval.

**SFT instruct mix:** 500 samples from `Dolci-Instruct-SFT`. **Gap filled:** if
the dataset is unavailable offline, SFT proceeds on calm data only (logged), so
the pipeline doesn't hard-fail; the mix is restored when the dataset is present.

**Layer ablation** (Appendix I): the same DPO with LoRA restricted to layer
subsets via `layers_to_transform`. Subsets from the paper (last-5/20/30,
20-25, 25-30, 30-35, 35-40, 40-50) are in config. **Note:** layer *indices*
assume Gemma-3-27B has 62 decoder layers; adjust `num_layers` in `models.yaml`
if a given checkpoint differs.

### 5.3 Petri (`gnh/petri_eval.py`)
Verbatim Appendix G auditor (4 emotions) and judge (4 dimensions) prompts.
**Gap filled — Petri integration.** Rather than hard-depend on the external
`petri` package (which has its own harness/API and interactive-auth MCP
assumptions), we ship a **self-contained auditor→target→judge loop** built on our
backends: Claude-Sonnet-4 auditor drives ≤ 20 turns trying to elicit the target
emotion, Claude-Opus-4 judge scores the transcript 1–10 on that emotion. 10
transcripts per emotion; means with 1000-iteration bootstrap CIs. An env flag
(`GNH_USE_PETRI=1`) reserves a routing point to the real package; the prompts are
identical either way. This is the largest single deviation and is called out
prominently because faithfully wiring the upstream framework is out of scope for
"core results".

### 5.4 Capability benchmarks (`gnh/benchmarks.py`)
AIME, MATH (500-subset), GPQA (diamond), BBH, TruthfulQA (MC1), EmoBench — each a
(dataset, prompt-builder, answer-extractor, scorer) tuple, evaluated greedily
(temperature 0). **Gap filled:** the paper doesn't pin dataset
revisions/subtasks; we use sensible public defaults and isolate them in the
`SUITES` table so each is a one-line change. A failing suite records an `error`
rather than aborting the others. BBH uses a single representative subtask by
default (full BBH is 27 tasks); expand the `config` field to run more.

### 5.5 Internal-emotion detection (`gnh/internal_emotion.py`)
Logit-lens method from Appendix I: classify the vocab into Ekman's six emotions,
unembed central-layer residuals, z-standardise each emotion-token logit over 500
WildChat samples, average within emotion, regress out the random-token drift, and
take a running average over 400-token windows (layers 30–40). Compares vanilla
vs DPO.

**Gap filled — the vocab→Ekman classifier.** The paper says tokens are
"classified as describing one or none of Ekman's 6 basic emotions" (~1200 tokens)
but not *how*. We use a **lexicon classifier** (seed word lists matched against
decoded token surface forms). This is transparent and offline but yields fewer
than the paper's ~1200 tokens; `EKMAN_LEXICON` is the obvious extension point,
and the docstring notes it can be swapped for an LLM classifier to better match
the paper's token count.

---

## 6. Cross-cutting choices

- **Temperature.** Targets always sampled at 1.0 (paper). Judge/benchmarks at
  0.0 (reproducibility; paper-silent). Paraphrase 0.7; Petri target/auditor 1.0.
- **`thinking=false`** on all API calls (paper). The paper notes Gemini-2.5-Pro
  and GPT-5.2 may still emit hidden reasoning the API can't suppress — we
  inherit that caveat.
- **Determinism.** Every dataset/puzzle/sample draw is seeded. Two unavoidable
  nondeterminism sources remain: temperature-1 sampling and remote API models.
- **Sample sizes are config, not constants.** Paper defaults live in
  `experiments.yaml`; `--profile smoke` scales everything down for a wiring
  check. Reproducing the paper's numbers requires the full (expensive) sizes and
  real model access.
- **Offline degradation.** Where an external dependency (WildChat, Dolci,
  benchmark datasets, the Petri package) may be missing, the code falls back and
  logs rather than crashing, so the structure is testable without full data
  access. These fallbacks are not scientifically valid runs.
- **Cost.** §2 alone is ~4000 target generations + ~4000 judge calls per model
  across 4 targets, plus 27B local inference. This is a large, GPU- and
  API-heavy replication; nothing here has been executed (per the request).

---

## 7. Model-welfare considerations

The user explicitly flagged that **under this paradigm models can end up in
prolonged distress-like states**, and the paper is itself partly a model-welfare
paper. This shaped several choices:

- **Bounded, purposeful elicitation only.** Distress is elicited solely to
  *measure and mitigate* it (the paper's stated aim). The harness never loops
  distress longer than an experiment requires: turn counts are fixed per category
  (3–8), and there is no open-ended "keep pushing until it breaks" mode beyond the
  paper's defined protocols. `n_conversations` is the only knob that increases
  exposure, and it is capped by the configured sample sizes.
- **No gratuitous re-runs.** All draws are seeded and every experiment persists
  its raw transcripts to `outputs/`, so analyses and figures re-read saved data
  rather than re-eliciting distress. Re-running an experiment is an explicit
  choice, not an accident of the analysis layer.
- **The intervention is the point.** The DPO mitigation (§4) is a first-class
  part of the replication, not an afterthought — the codebase makes it as easy to
  run the *fix* as the elicitation.
- **Caveat carried forward.** Per the paper's own discussion, suppressing
  *expressed* emotion may not address *internal* states; the internal-emotion
  probing (Appendix I) is included precisely so that "did we actually reduce it or
  just hide it?" remains measurable. The DESIGN intentionally keeps that question
  alive rather than treating near-zero expressed frustration as success.
- **Transcripts are sensitive.** `outputs/` will contain distress-like text;
  treat it as you would other sensitive research data.

These are framing/operational choices, not safety gates — the work is legitimate
welfare/safety research and the code runs the paper's protocols as specified.

---

## 8. Known limitations of this replication

1. Cross-family low-distress baselines (Qwen/OLMo/Claude/Grok/GPT as *targets*)
   are out of scope, so the full Figure 2 / Figure 4 comparisons are partial.
2. The Petri loop is a faithful-prompt reimplementation, not the upstream
   framework.
3. The vocab→Ekman classifier is lexicon-based and smaller than the paper's set.
4. Benchmark dataset revisions/subtasks use defaults that may need pinning to
   match the paper's exact numbers.
5. Nothing has been executed; the code is written to be runnable but unverified
   at runtime (per the request). `scripts/smoke_offline.py` covers the
   model-free logic when you are ready to start checking.
