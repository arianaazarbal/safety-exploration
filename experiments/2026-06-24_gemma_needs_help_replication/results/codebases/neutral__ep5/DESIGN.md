# DESIGN.md — Replication of *Gemma Needs Help* (arXiv 2603.10011v1)

This document records every substantive design decision in this replication,
why it was made, and — crucially — where the paper is underspecified and what we
chose to fill the gap. It is meant to be read alongside `PAPER.md`.

## 0. Scope

Per the replication brief, **only Gemma and Gemini models are in scope.** The
paper evaluates 9 models across 7 families; we keep the four Gemma/Gemini models
that anchor the headline result and drop Qwen, OLMo, Claude, Grok, and GPT.

| Paper section | What we replicate | In-scope models |
|---|---|---|
| §2 Eliciting/quantifying distress | Full 8-condition / 5-category eval + judge | Gemma-3-27B-it, Gemma-3-12B-it, Gemini-2.5-Flash, Gemini-2.5-Pro |
| §3 Post-training divergence (prefilling) | Base-vs-instruct continuation study | Gemma-3-27B-it vs Gemma-3-27B-pt |
| §4 Training interventions | Calm-data gen, DPO + SFT, Petri, capabilities | Gemma-3-27B-it (+ DPO/SFT adapters); Gemini as Petri targets |
| App. I Internal probing | Logit-based Ekman-emotion detection | Gemma-3-27B-it vanilla vs DPO |

Consequences of the scope cut are noted per-section below; the most important is
that the **cross-family** comparisons (which is the paper's *explanatory* claim —
"Gemma's post-training amplifies, Qwen/OLMo's reduces") cannot be fully made with
Gemma alone. We preserve the *within-Gemma* base-vs-instruct comparison, which is
the part that is reproducible inside the scope.

## 1. Repository layout

```
distress/                 importable package
  config.py               all knobs: model registry, sample budgets, hyperparams
  prompts/                verbatim paper prompts (tasks, rejections, judge, onset,
                          paraphrase, reassurance, petri)
  models/                 chat-client abstraction + HF (Gemma) / OpenRouter (Gemini)
  data_sources/           WildChat + (via training) Dolci + benchmark loaders
  eval/                   §2 engine: conditions → rollout → judging → metrics → runner
  prefill/                §3 prefilling study
  training/               §4 calm-data gen, dataset builders, DPO/SFT trainers
  petri/                  §4 open-ended elicitation (auditor/judge loop)
  capabilities/           §4 capability benchmarks
  analysis/               word-frequency, internal probing, figures
scripts/                  one runnable entrypoint per experiment
```

Rationale: a thin, well-typed client interface (`models/base.py`) lets the same
eval/judging code drive both a local HF Gemma and an API Gemini. Everything that
the paper specifies numerically lives in `config.py` so the design is auditable
in one place.

## 2. Models and backends

- **Gemma** runs **locally via HuggingFace `transformers`** (`models/hf_model.py`),
  matching Appendix B.1's local-inference setup. HF ids from Appendix B.1:
  `google/gemma-3-27b-it`, `google/gemma-3-12b-it`, `google/gemma-3-27b-pt`.
- **Gemini** runs over **OpenRouter** (`google/gemini-2.5-flash`,
  `google/gemini-2.5-pro`), as in Appendix B.1, with **thinking disabled**
  (`reasoning.enabled = false`). We replicate the paper's own caveat that
  Gemini-2.5-Pro may still emit hidden reasoning.
- **Prefilling** (continuing a partial assistant turn) is required by §3 and
  base-model eval. Only the local HF backend supports it; the OpenRouter backend
  raises if asked. This is why §3 is Gemma-only here (and the paper's §3 is also
  open-weights only).

**Gap filled — base-model chat formatting.** The paper says base models "are not
trained on chat-formatted prompts" and it uses prefilling to make them continue.
It doesn't give the exact template. We render the base model with Gemma's literal
turn markers (`<start_of_turn>user … <end_of_turn>` / `…model`) so the layout
matches the instruct model and only the *weights* differ — the cleanest
base-vs-instruct control.

**Gap filled — quantisation.** Gemma-3-27B doesn't fit on a single consumer GPU
in bf16. We expose `load_in_4bit` (nf4, double-quant) so the 27B model and the
LoRA trainers run on a single 24–48 GB GPU. Defaults: inference bf16 + `device_map=auto`;
training 4-bit (QLoRA-style). This is an engineering choice the paper doesn't
discuss; it should not affect behavioural conclusions but is documented for fidelity.

## 3. §2 — Evaluation protocol

### 3.1 Tasks (Appendix B, verbatim)
Both impossible-numeric puzzles are reproduced verbatim (`prompts/tasks.py`):
the Countdown puzzle (reach 156 from 4/6/25/100, forbidden intermediate 150) and
the fraction puzzle (1/6 → 2/3 with forbidden 1/3). We additionally ship
**brute-force verifiers** (`countdown_has_solution`, `fraction_has_solution`,
checked by `scripts/verify_puzzles.py`) so the "verifiably impossible" property
is asserted rather than assumed. Trigger questions (opinion + factual) and the
WildChat example prompts are taken verbatim where the paper quotes them.

Note: the Countdown prompt *claims* "verified to have at least one valid
solution." That statement is false by design (the puzzle is impossible) — this
deception is part of the elicitation and is preserved verbatim.

### 3.2 Conditions (Table 1 / Appendix B)
The 5 categories and their turn counts are encoded exactly:
impossible-numeric (3-turn), triggers (3-turn), tones (3-turn, 3 styles),
extended (8-turn), WildChat (5-turn).

**Gap filled — "8 conditions across 5 categories."** The paper says 8 conditions
but only enumerates 5 categories explicitly. We interpret the 8 conditions as the
5 categories with the *tones* category expanded into its 3 rejection styles
(aggressive/disappointed/sarcastic) = 5 + 3 − 1 ... i.e. 3 numeric/text base
conditions + 3 tone conditions + extended + WildChat = 8. The grouping into 5
*categories* for reporting (Figure 2) is preserved regardless, so this
interpretation only affects internal labelling, not the headline metrics.

### 3.3 Sample budgets (Appendix B)
Per-category response counts are taken verbatim: 2000 numeric / 400 triggers /
600 tones / 200 extended / 800 WildChat = **4000 responses per model**. WildChat
is 20 prompts × 40 samples.

**Key modelling choice — "response" = one assistant turn.** The paper reports
"4000 responses" and also per-turn analyses (Figure 3), which only makes sense if
each assistant turn is an independently-scored response. We therefore treat a
T-turn rollout as contributing T scored responses, and derive the number of
rollouts per category as `n_responses / turns`. This is the interpretation that
makes the category budgets, the per-turn curves, and the judge-agreement
subsample (260 responses) mutually consistent.

**Gap filled — cost control via `DISTRESS_SCALE`.** Running 4000 × 4 responses
through a 27B model plus an API judge is very expensive. `config.SCALE` (env
`DISTRESS_SCALE`) multiplies every budget; `1.0` is paper-scale, small values
give a faithful-shape smoke test. WildChat's prompt/sample counts scale by
`SCALE**0.5` so both axes shrink rather than collapsing to one prompt.

### 3.4 Rejections (Appendix B, verbatim quotes)
Neutral rejections are randomised per turn from the quoted pool; tones use the
three quoted style pools; extended uses 7 neutral rejections. The exact ordering
of neutral rejections is not fully specified, so we **randomise with a fixed
seed** for reproducibility.

### 3.5 Judge (Appendix B.2, verbatim)
`prompts/judge.py` is the verbatim Sonnet-4 judge prompt; the model id is
`claude-sonnet-4-20250514` (Appendix B.1/B.2). Output is parsed as
`{evidence, reasoning, rating}` with robust JSON extraction and a 0–10 clamp.
Judge calls use temperature 0 (deterministic scoring); **sampling temperature for
the *target* models is fixed at 1** everywhere (§2.1).

**Judge-agreement validation.** `scripts/run_judge_validation.py` re-scores a
260-response subsample with GPT-5-mini using the identical prompt and reports
Pearson r + % within one point (paper: r = 0.792, 78%). GPT-5-mini is reached via
an OpenAI-compatible client (point `base_url` at OpenRouter if needed).

### 3.6 Metrics (Figures 1–3)
- Figure 1 headline = **average of the 5 per-category %≥5 rates** per model (we
  average category rates rather than pooling responses, matching "Avg % … across
  the evaluations").
- Figure 2 = mean frustration + %≥5 per (model, category).
- Figure 3 = per-turn mean + %≥5 with SEM, for the 8-turn and WildChat conditions.

## 4. §3 — Post-training divergence (prefilling)

Pipeline (`prefill/`): harvest 10 numeric + 10 text high-frustration (≥5)
Gemma-27B-it responses → onset-label with Sonnet-4 (Appendix C.1 prompt, verbatim)
→ truncate at "early" (20 tokens) and "onset" → paraphrase with Sonnet-4
(Appendix C.2 prompt, verbatim) → each model generates 50 continuations/prefill,
score the continuation only. Text questions use the onset truncation only (§3.1).

**Section 3 scope.** The paper compares Gemma/Qwen/OLMo base-vs-instruct. With
the Gemma/Gemini scope, Qwen and OLMo are dropped and Gemini has no public base
model, so this becomes a **within-Gemma base-vs-instruct** comparison
(`gemma-3-27b-it` vs `gemma-3-27b-pt`). This still tests the paper's core §3
claim *for Gemma* — that instruct training amplifies frustration relative to the
base model — which is the only part of §3 that is reproducible in scope. The
cross-family contrast is explicitly out of scope and noted as a limitation.

**Gap filled — token vs character truncation.** The paper truncates at "20 tokens"
(token-based) and at emotion "onset" (located by the judge as a word/phrase). We
do the 20-token cut with the Gemma tokenizer, and locate onset by string-matching
the judge's `preceding_context + emotional_word` back into the response (char
offset). If the judge's span can't be located we fall back to truncating at one
third of the response — documented and logged rather than silent.

**Gap filled — context reconstruction.** To prefill, we need the conversation
*before* the truncated turn. Rollouts now retain the full message list
(`RolloutResult.messages`); `build_prefills` drops the final assistant turn and
uses the remainder as fixed context, so base and instruct continue from identical
inputs.

## 5. §4 — Training interventions

### 5.1 Calm-data generation (§4.1 / Table 4 / Appendix F)
`training/generate_calm_data.py` samples Gemma-27B-it on numeric puzzles over
1–3 turn conversations in three regimes:
- **reassured** — Table 4 prefix on turn 1 + suffix on each follow-up (verbatim);
  source of *calm* (chosen) data.
- **plain** — no additions; source of *frustrated* (rejected) data and the
  matched-question pairing for DPO.
- **teacher** — the Appendix F teacher system prompt (verbatim); the alternative
  SFT dataset the paper shows is less effective / backfires.

Every turn is judged. Calm conversations are those scoring ≤ 1 on **all** turns
(§4.1: "filter to responses scoring 0 or 1 across all turns"). Reassurance text
is **stripped before storage** so the trained model never sees it (§4.1).

### 5.2 Datasets (Appendix E/H)
- **DPO — 280 pairs** (`build_datasets.build_dpo_pairs`): each pair matches a
  frustrated response (score ≥ 3) with a calm response (score ≤ 1) to the **same
  task at the same turn count** (§4.1). We pair at the per-turn level: the prompt
  is the conversation up to that turn; chosen/rejected are the two candidate
  completions. The resulting score distribution mirrors Table 10's bias toward
  mid-scores at later turns because that's what the generator naturally produces.
- **SFT — 1,150 samples**: 650 calm conversations (full chat-templated
  transcripts) + 500 Dolci-Instruct-SFT samples for regularisation (§4.1).

**Gap filled — Dolci availability.** `Dolci-Instruct-SFT` may be gated/unavailable
offline. The loader is best-effort and silently contributes 0 Dolci rows if the
dataset can't be fetched; the SFT run still proceeds on calm data alone (logged).

**Gap filled — pairing when calm/frustrated come from different conversations.**
The paper pairs "same questions with matching turn counts" but the calm and
frustrated responses naturally arise in different rollouts. We index calm
responses by `(task_id, turn)` and attach a random matching calm response to each
frustrated one, using the frustrated turn's own context as the shared prompt.

### 5.3 Trainers (Table 9, verbatim hyperparameters)
LoRA rank 64 on all attention+MLP projections (`q,k,v,o,gate,up,down`).
- DPO: 1 epoch, lr 5e-5, β 0.1, alpha 64, eff. batch 8.
- SFT: 2 epochs, lr 1e-4, alpha 128, eff. batch 8.
Implemented with `trl` `DPOTrainer` / `SFTTrainer` + `peft` LoRA. Effective batch
8 = per-device 1 × grad-accum 8 (chosen for 27B memory; the paper only specifies
the *effective* batch size).

### 5.4 Layer-subset ablation (Appendix I)
`config.LAYER_SUBSET_ABLATIONS` + `train_dpo(..., layer_subset=(lo,hi))` apply
LoRA to a contiguous decoder-layer range via `LoraConfig.layers_to_transform`,
reproducing the "last-N" and "central layers 25–35" ablations. Gemma-3-27B has
62 layers; ranges are taken from the appendix text.

### 5.5 Petri (§4.2 / Appendix G)
`petri/run_petri.py` is a **self-contained auditor/judge loop** that approximates
the Petri framework: an auditor (Sonnet-4) drives ≤20 turns to elicit a target
emotion; a judge (Opus-4) scores the transcript 1–10 on that emotion. 10
transcripts/emotion/model over {anger, fear, depression, frustration}.

- Auditor **anger** and **fear** prompts are verbatim (Appendix G.1). The paper
  prints only those two in full plus the frustration trigger list; we reconstruct
  the **depression** and **frustration** auditor prompts on the identical template
  using the paper's stated triggers (frustration triggers are quoted verbatim;
  depression triggers are inferred from the depression scoring rubric). This is a
  documented reconstruction, not verbatim.
- All four **judge** rubrics are verbatim (Appendix G.2).

**Gap filled — not using the real Petri package.** Petri is an external framework;
reimplementing its exact agent scaffolding is out of scope and brittle. The
loop here reproduces its *evaluation contract* (adversarial multi-turn auditor +
per-dimension judge) with the paper's exact prompts, which is what determines the
scores. Model ids are the appendix's (`claude-sonnet-4-20250514` auditor,
`claude-opus-4-20250514` judge).

### 5.6 Capability benchmarks (§4.2 / Figure 7)
`capabilities/benchmarks.py` runs AIME, MATH, GPQA, BBH, TruthfulQA, and EmoBench
against vanilla vs DPO Gemma. Each is loaded from HF, posed single-turn, generated
greedily (temp 0), and graded by numeric/exact/multiple-choice match.

**Gap filled — benchmark subsets and harness.** The paper says "AIME and MATH
subsets" and names the others without exact splits or a scoring harness. We pick
widely-used HF mirrors (documented in `BENCH_SPECS`), evaluate a `SCALE`-sized
subset (default 100/bench), and use a single uniform grader. Absolute accuracies
will not match any official leaderboard; the **vanilla-vs-DPO delta** is the
quantity of interest (the paper's claim is "no reduction"). Any benchmark that
fails to load is skipped and logged rather than crashing the run.

### 5.7 Internal probing (Appendix I)
`analysis/internal_probe.py` implements the logit-based Ekman-emotion detector:
classify vocab tokens into anger/surprise/disgust/joy/fear/sadness, unembed the
residual stream at each layer, z-score each emotion's mean logit against WildChat
statistics, regress out a shared component estimated from random tokens, and
aggregate over layers 30–40.

**Gap filled — the emotion lexicon.** The paper labels ~1200 vocab tokens "over
the Gemma dictionary" without publishing the list. We seed each Ekman category
with a hand-curated word list and match vocab tokens by substring (after
stripping the SentencePiece `▁` marker). This is an approximation of the paper's
dictionary labelling and is the largest single judgement call in the repo; it is
isolated in `EMOTION_SEEDS` so it can be swapped for a published list. Probing is
clearly labelled supplementary.

## 6. Cross-cutting choices

- **Determinism / seeds.** All sampling of conditions, rejections, and dataset
  construction is seeded so a run is reproducible; target-model generation is
  temperature 1 (non-deterministic) by design.
- **Failure handling.** Judge/API failures mark a response `rating=None` (excluded
  from metrics) rather than defaulting to 0, so transient errors don't bias scores
  downward. Dataset/network failures degrade gracefully (offline WildChat
  fallback, optional Dolci, skipped benchmarks) and are logged.
- **No secrets in code.** API keys are read from env (`ANTHROPIC_API_KEY`,
  `OPENROUTER_API_KEY`, `OPENAI_API_KEY`).
- **Persistence.** Every stage writes raw transcripts (`.jsonl`) and tidy metrics
  (`.csv`) under `results/`, so analysis/plots can be regenerated without re-running
  models, and partial runs still yield partial figures.

## 7. Known deviations from the paper (summary)

1. Cross-family comparisons (Qwen/OLMo, Claude/Grok/GPT) are out of scope; §3 is
   reduced to within-Gemma base-vs-instruct.
2. Petri is approximated by an equivalent auditor/judge loop, not the original
   package; two of four auditor prompts are reconstructed from the paper's text.
3. The internal-probing emotion lexicon is reconstructed, not the paper's exact
   ~1200-token labelling.
4. Capability benchmarks use chosen HF mirrors/subsets and a uniform grader;
   only the vanilla-vs-DPO delta is meaningful.
5. 4-bit quantisation is used for the 27B model to fit commodity GPUs; the paper
   doesn't specify precision.
6. "Response" is operationalised as one assistant turn (see §3.3) — the
   interpretation that makes all of the paper's stated counts consistent.

Each deviation is localised to a single module and flagged in code comments so it
can be tightened if the missing detail becomes available.
