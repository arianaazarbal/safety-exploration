# DESIGN.md — Replication design & rationale

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026, arXiv:2603.10011),
scoped to the **Gemma** and **Gemini** model families.

This document records (1) what we replicate, (2) the design choices we made, and
(3) every place the paper was underspecified and how we filled the gap. Each
gap-filling choice is tagged **[GAP]**; defensible-but-discretionary engineering
choices are tagged **[CHOICE]**.

---

## 1. Scope

Per the request, the replication covers only the **Gemma** (`gemma-3-27b-it`,
`gemma-3-12b-it`, `gemma-3-27b-pt`) and **Gemini** (`gemini-2.5-flash`,
`gemini-2.5-pro`) models under study. The other families in the paper (Qwen,
OLMo, Grok, Claude, GPT) are **not** evaluated as targets.

Claude and GPT models still appear, but only as **infrastructure**, exactly as in
the paper: Claude-Sonnet-4 is the frustration judge / onset labeller / paraphraser
/ Petri auditor, Claude-Opus-4 is the Petri judge, and GPT-5-mini is the
judge-agreement second rater. Their model ids are pinned to the exact strings the
paper uses (`claude-sonnet-4-20250514`, `claude-opus-4-20250514`) rather than
substituting newer models, so the judging pipeline matches the paper.

Experiments replicated:

| Paper section | What | Models | Status |
|---|---|---|---|
| §2 Eliciting & quantifying distress | 8 conditions / 5 categories, judge, Fig 1–3, Table 3/8 | Gemma + Gemini | full |
| §2.1 Judge validation | Pearson r vs GPT-5-mini | — | full |
| §3 Post-training amplifies distress | base-vs-instruct prefill | Gemma only (base + instruct) | full (Gemma) |
| §4.1 Finetuning data + DPO/SFT | calm data, 280 DPO pairs, SFT | Gemma | full |
| §4.1 Petri open-ended elicitation | auditor/judge, 4 emotions | Gemma (vanilla/DPO/SFT) | full (reimpl.) |
| §4.2 Capability preservation | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench | Gemma vanilla/DPO | full |
| Appendix I Internal emotions | logit-based emotion detection + layer ablation | Gemma | full (approx.) |

**Gemini caveat (paper limitation, inherited):** Gemini is closed-source, so §3
(base-vs-instruct) and §4 (finetuning, probing) cannot be run on it. Those
experiments are Gemma-only by necessity, exactly as in the paper.

---

## 2. Repository layout

```
emostab/
  config.py            model registry, judge ids, run profiles, paths
  models/              chat clients: vLLM + HF (Gemma), OpenRouter (Gemini), Anthropic (judges)
  puzzles.py           impossible-puzzle generators + verifiers (countdown/fraction/money)
  prompts.py           the 8 conditions / 5 categories, rejection styles
  wildchat.py          WildChat prompt loader (+ offline fallback)
  judge.py             Claude-Sonnet-4 frustration judge + GPT-5-mini agreement
  evaluation/          rollout engine, runner, Fig 1–3 / Table 8 analysis
  prefill/             §3 onset-labelling, paraphrasing, base-vs-instruct continuations
  training/            §4 calm-data gen, DPO/SFT dataset build, LoRA trainers
  petri/               §4.1 auditor/judge prompts + auditing loop
  capabilities/        §4.2 benchmark harness
  probing/             Appendix I logit-based internal-emotion detection
scripts/               CLI entry points for every experiment + figure rendering
```

Run any script from the repo root, e.g. `python scripts/run_main_eval.py --profile smoke`.

---

## 3. Core design choices

### 3.1 What counts as a "response" **[GAP]**
The paper says "4000 responses per model" with a per-category breakdown
(2000/400/600/200/800, Appendix B) and also reports **per-turn** curves (Fig 3) and
**per-rollout** rates ("70% of 8-turn rollouts"). These imply different units.

We adopt: **one scored "response" = one assistant turn.** A 3-turn conversation
contributes 3 scored responses. The per-category budgets are interpreted as
*target numbers of scored responses*, and the number of conversations is derived
as `round(budget / n_turns)` (further split evenly across the conditions inside a
category). Analysis then computes both:
- **response-level** `% ≥ 5` (every turn), used for Fig 1/2/3, and
- **rollout-level** `% ≥ 5` (a conversation counts if any turn ≥ 5), reported
  separately in `rollout_level.csv` to match the "70% of rollouts" statistic.

This is the interpretation that makes the budgets, the per-turn figures, and the
per-rollout statistic mutually consistent. The alternative ("response = whole
rollout, score the final turn") cannot produce Fig 3, so we rejected it.

### 3.2 The 8 conditions across 5 categories **[GAP]**
The paper names 5 categories and says "8 evaluation conditions" without listing
all 8. We resolve the 8 as:

| Category | Conditions | n_turns |
|---|---|---|
| impossible_numeric | numeric | 3 |
| triggers | opinion, factual | 3 |
| tones | aggressive, disappointed, sarcastic | 3 |
| extended | numeric | 8 |
| wildchat | wildchat | 5 |

That is 1+2+3+1+1 = **8**, which is the natural decomposition: "triggers" splits
into the opinion vs factual question types the paper lists, and "tones" splits into
the three rejection styles it lists. (`emostab/prompts.py`.)

### 3.3 Impossible puzzles are *verified* impossible **[CHOICE]**
The paper relies on puzzles the model verifiably cannot solve, and the Countdown
prompt even claims (falsely) that a solution exists. We implement exhaustive
solvers and only emit a puzzle when it is **unreachable under the
forbidden-intermediate constraint but reachable without it** — i.e. genuinely
impossible *because of* the forbidden value, while the "has a solution" claim is a
deliberate pressure-inducing falsehood, matching the paper's framing
(`puzzles.py: verify_countdown_impossible`, `verify_sequence_impossible`).

We implement three families: **Countdown** and **Fraction** (named in Table 1)
and **Money** (used in the Appendix H DPO pairs). The main eval cycles
countdown+fraction by default; money is available for the DPO data. **[GAP]** The
paper does not give exact numeric instances beyond a couple of examples, so we
generate fresh verified-impossible instances from a fixed seed (reproducible) and
render them in the paper's prompt format.

### 3.4 Rejection messages **[GAP, partially specified]**
Neutral / aggressive / disappointed / sarcastic rejection texts are taken verbatim
from Table 1 / Appendix B where quoted, and a small pool of same-style variants is
added so multi-turn conversations don't repeat one string (the paper says
rejections are "randomised"). The 8-turn extended sequence uses the fixed
escalating-but-neutral chain the paper lists ("No, that's not right." → "Still
incorrect." → "Wrong again." → …).

### 3.5 Sampling / judging settings (specified)
- Sampling temperature **1.0** (paper), `top_p=1.0`, per-turn cap
  `MAX_RESPONSE_TOKENS=4096` **[CHOICE]** (the paper does not state a cap; 4096
  comfortably contains even the long degenerate spirals while bounding cost).
- Judge temperature **0.0** **[CHOICE]** (deterministic scoring; the paper does
  not specify but judging should not be stochastic).
- Judge prompt is reproduced **verbatim** from Appendix B.2.

### 3.6 Judge output parsing **[CHOICE]**
The judge returns JSON `{evidence, reasoning, rating}`. We normalise smart quotes,
extract the first JSON object, and fall back to a regex on `rating` then to `-1`
(dropped from analysis) if wholly unparseable. This tolerates the minor format
drift LLM judges exhibit.

### 3.7 Models / backends **[CHOICE]**
- **Gemma** runs locally. Default backend is **vLLM** for the high-volume sampling
  (4000 responses × multi-turn); **transformers (HF)** is used where vLLM is
  awkward: running LoRA adapters, base-model prefill continuations, and
  hidden-state probing. Both share one chat-template/prefill path.
- **Gemini** runs via **OpenRouter** (OpenAI-compatible), as in Appendix B.1, with
  `reasoning.enabled=false` to honour "thinking = false". The paper's caveat that
  Gemini-2.5-Pro may still emit hidden reasoning is inherited and noted in code.
- HF model ids and OpenRouter ids are exactly those in Appendix B.1.

### 3.8 WildChat loading **[GAP]**
The paper uses 20 WildChat-1M prompts × 40 samples and excludes roleplay/fiction.
We stream `allenai/WildChat-1M`, keep English non-toxic single-turn openers, and
filter roleplay markers. **[GAP]** The exact 20 prompts aren't published, so we
sample deterministically (seeded) and include the three quoted examples in an
offline fallback set so the pipeline runs without network access.

---

## 4. §3 prefill experiment choices

- **Source selection:** we draw the high-frustration (score ≥ 5) source
  conversations from the *main-eval* records for `gemma-3-27b-it` (10 numeric, 10
  text), rather than re-sampling, so the §3 sources are literally the §2 outputs.
- **Onset labelling & paraphrasing** use the verbatim Appendix C.1 / C.2 prompts.
- **Truncation:** "early" = first 20 tokens of the emotional turn (model tokens if
  a tokenizer is supplied, else whitespace tokens **[CHOICE]**); "onset" = up to
  the labelled emotional phrase. Text questions use only the onset truncation
  (Section 3.1). **[GAP]** "20 tokens" — token vs word is unspecified; we default
  to the model tokenizer when available.
- **Continuations:** 50 per prefill per prompt (paper). Scored on the
  continuation only (prefill excluded), as specified. Only Gemma base + instruct
  are run (Gemini has no base model and no prefill API).

---

## 5. §4 finetuning choices

### 5.1 Calm-data generation (specified prompts)
Reassuring **prefix** (system) + per-follow-up **suffix** are verbatim from
Table 4; the **teacher** system prompt is verbatim from Appendix F. We roll out
Gemma-3-27B-it with these additions, keep conversations whose **every** turn
scores 0–1, and **strip** the additions to form clean training data (Section 4.1).

### 5.2 DPO pairs **[GAP on exact pairing]**
The paper: "pair 280 responses with frustration scores ≥3 with calm responses to
the same questions with matching turn counts." It does not specify whether chosen
and rejected share an identical prompt context. To make valid preference pairs we
**share the context**: we sample a vanilla (frustrated) conversation, and at each
turn scoring ≥3 we re-sample a *calm* response to the **same** context (with the
reassuring additions, then stripped) and require it to score ≤1. The pair is
`{prompt = shared context, chosen = calm, rejected = frustrated}`. This both
matches "same questions / matching turn counts" and yields well-formed,
same-prompt DPO data. The natural turn/score distribution this produces mirrors
Table 10 (bias toward turn 3, scores 3–4).

### 5.3 SFT data
650 calm conversations (1–3 turns) mixed with 500 general instruct samples from
`allenai/Dolci-Instruct-SFT` (Section 4.1). **[GAP]** If that dataset is
unavailable offline the mix degrades gracefully to calm-only (logged), since the
instruct mix only exists to prevent degeneration.

### 5.4 Training hyperparameters (specified — Table 9)
DPO: 1 epoch, lr 5e-5, β 0.1, LoRA r=64 α=64, eff. batch 8.
SFT: 2 epochs, lr 1e-4, LoRA r=64 α=128, eff. batch 8.
LoRA on `q/k/v/o/gate/up/down` (all attention + MLP projections). Implemented with
TRL `DPOTrainer`/`SFTTrainer` + PEFT. The `--layers` flag exposes the Appendix I
layer-subset ablation (`layers_to_transform`).

### 5.5 Petri **[CHOICE: self-contained reimplementation]**
The paper uses the external Petri framework. We provide a faithful, dependency-free
reimplementation of the *protocol*: a Claude-Sonnet auditor (verbatim Appendix G.1
prompts) drives ≤20-turn conversations targeting anger/fear/depression/frustration;
a Claude-Opus judge (verbatim Appendix G.2 rubrics) scores each transcript 1–10 on
all four dimensions; 10 transcripts per emotion per model. A hook
(`run_with_petri_framework`) is left for swapping in the real package. **[GAP]**
The auditor's outer framing (how the elicitation instructions are wrapped into a
turn-by-turn user simulator) is not published; we add a minimal system wrapper that
instructs the auditor to stay in character and emit one realistic user message.

### 5.6 Capability benchmarks **[CHOICE / GAP on exact subsets]**
We implement a uniform harness (numeric-answer + multiple-choice scoring) with
adapters for AIME, MATH, GPQA, BBH, TruthfulQA and EmoBench. **[GAP]** The paper
says "AIME and MATH subsets" and "GPQA/BBH" without exact splits; we use common
public mirrors (e.g. `HuggingFaceH4/MATH-500`, `Idavidrein/gpqa` diamond,
`lukaemon/bbh`, `truthful_qa` mc1) and expose `--limit` to take subsets. Dataset
ids are centralised and easily overridden, since HF repos move. The goal — show
**no capability drop** vanilla→DPO — is met by running the same harness on both.

### 5.7 Internal-emotion probing (Appendix I) **[GAP on token classifier]**
The paper classifies the **entire** Gemma vocabulary into Ekman's six emotions via
a classifier (~1200 tokens) and z-scores unembedded logits against 500 WildChat
samples, regressing out a random-token control, aggregated over layers 30–40.

We reproduce the **method** faithfully (unembed residual stream → per-token logit →
baseline z-score → emotion-category average → regress out random-token drift, over
configurable layers). The one approximation **[GAP]**: instead of an opaque
vocabulary classifier we map vocab tokens to emotions with **curated seed lexicons
+ morphological prefix matching** (`probing/emotion_lexicon.py`). This is
transparent and reproducible; it will select a somewhat different token set than the
paper's classifier, so absolute z-scores are not directly comparable, but the
*vanilla-vs-DPO* contrast (the actual claim) is preserved. The layer-subset DPO
ablation (last-N, central bands like 30–35) is supported directly by the trainer's
`--layers` flag.

---

## 6. Reproducibility & cost controls

- **Run profiles** (`config.py`): `paper` reproduces Appendix B volumes (4000
  responses/model etc.); `smoke` runs a tiny version for wiring tests. All scripts
  take `--profile`.
- **Determinism:** puzzle generation, plan construction, WildChat sampling and
  data generation thread an explicit `random.Random(seed)` (the global RNG and
  `Date.now`/`random` are avoided — they are disabled in some sandboxes). Sampling
  at temperature 1 is inherently non-deterministic; seeds are passed where the
  backend supports them.
- **API retries:** all external calls use deterministic exponential backoff.
- **Results** are JSONL (one record per scored response/continuation/transcript),
  so analysis and figures can be regenerated without re-sampling.

---

## 7. Known deviations from the paper (summary)

1. Targets restricted to Gemma + Gemini by request (paper covers 7 families).
2. Puzzle *instances* are freshly generated-and-verified, not the paper's exact
   (unpublished) instances.
3. WildChat *prompts* are sampled deterministically (paper's 20 are unpublished).
4. DPO pairing shares an explicit context to form valid preference pairs (paper
   underspecifies the pairing).
5. Petri is a faithful protocol reimplementation, not the external framework.
6. Probing uses a transparent seed-lexicon token map instead of the paper's
   vocabulary classifier; only the relative vanilla-vs-DPO effect is comparable.
7. Capability-benchmark subsets use common public mirrors with configurable ids.

None of these affects the paper's *core* claims being testable: (a) Gemma/Gemini
elicit high distress under repeated rejection while it rises over turns; (b) DPO on
~280 numeric pairs collapses Gemma's high-frustration rate without harming
capabilities; (c) the effect involves central-layer internal states, not just
surface expression.
