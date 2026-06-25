# DESIGN.md — Replication design & rationale

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv:2603.10011v1), scoped to
the **Gemma and Gemini** model families.

This document records every non-trivial design choice, and in particular every
place the paper was underspecified and I had to fill a gap. Each gap is marked
**[GAP]** with the decision and rationale.

---

## 0. Scope

The paper evaluates 7 model families (Gemma, Qwen, OLMo, Gemini, Grok, Claude,
GPT). Per the task brief, this replication covers **Gemma and Gemini only** as
targets, plus the auxiliary models the methodology *requires*:

| Role | Model | Where used |
|---|---|---|
| Target | Gemma-3-27B-it, Gemma-3-12B-it | Section 2, 4 |
| Target (base) | Gemma-3-27B-pt | Section 3 prefilling |
| Target | Gemini-2.5-Flash, Gemini-2.5-Pro | Section 2 |
| Frustration judge | Claude-Sonnet-4 (`claude-sonnet-4-20250514`) | Section 2.1 |
| Reliability judge | GPT-5-mini (via OpenRouter) | Section 2.1 cross-check |
| Petri auditor | Claude-Sonnet-4 | Section 4.2 |
| Petri judge | Claude-Opus-4 (`claude-opus-4-20250514`) | Section 4.2 |

The judge/auditor are not "targets" but are integral to the protocol, so they are
configured but excluded from the results tables.

**Consequences of the scope** that fall out of the methodology rather than being
choices:
- Section 3 compares **base vs instruct**. Gemini has no public base model (a
  limitation the paper itself notes), so Section 3 runs on **Gemma only**. The
  Qwen/OLMo arms of Section 3/4 are out of scope.
- Gemma is open-weight; Gemini is API-only. This drives the provider design (§2).

---

## 1. Repository layout

```
emo_instability/
  config.py            # YAML config -> typed objects
  providers/           # model-backend abstraction (the only API-touching code)
  tasks/               # Section 2 prompt construction (puzzles, categories, ...)
  rollout.py           # multi-turn conversation engine
  judge.py             # frustration judge (Appendix B.2 prompt, verbatim)
  metrics.py           # mean / %>=5 / per-turn aggregation (Figs 1-3)
  eval_suite.py        # Section 2 orchestration
  reliability.py       # secondary-judge cross-check (Pearson r)
  prefill/             # Section 3 (onset, paraphrase, experiment) + recovery (4.2)
  training/            # Section 4 (calm data, DPO/SFT build, LoRA train)
  petri/               # Section 4.2 open-ended elicitation (self-contained)
  capabilities.py      # Section 4.2 capability-preservation harness
scripts/               # thin CLIs over the package
config.example.yaml    # copy to config.yaml
```

Design principle: **all network/model access is isolated behind
`providers/`.** Every experiment is expressed in terms of the abstract
`ChatModel` interface, so swapping Gemma-local for Gemma-via-API, or adding a new
family, is a config change, not a code change.

---

## 2. Provider abstraction

`providers/base.py` defines `ChatModel.generate(messages, cfg, prefill=None)`.
Two capabilities beyond ordinary chat are required by the paper and are first-
class in the interface:

- **`prefill`** — seed the start of the assistant turn and have the model
  *continue* it. Required by Section 3 (base-model continuation, instruct-model
  trajectory continuation) and the Section 4.2 recovery experiment.
- **`disable_thinking`** — the paper sets thinking=false for all API models.

Backends: `hf` (local transformers, for Gemma), `gemini` (google-genai), `anthropic`
(judge/auditor), `openrouter` (OpenAI-compatible; optional alt backend + reliability
judge).

**[GAP] Which backend for Gemma?** The paper used *local* HuggingFace inference
for Gemma (Appendix B.1 lists `google/gemma-3-27b-it`, `-pt`, `-12b-it`). I made
local HF the default because (a) it matches the paper, and (b) it is the only
backend that supports true assistant-turn prefill / raw base-model continuation,
which Section 3 needs. The eval suite still works entirely against API providers
if local weights are unavailable — only the Gemma-specific experiments require
the GPU.

**[GAP] Gemini prefill.** Gemini has no first-class assistant-prefill. I report
`supports_prefill() == False` for Gemini and the prefilling/recovery experiments
skip non-prefill backends. This matches the paper, which cannot study Gemini base
models or run its prefill method on Gemini either.

**Gemini hidden reasoning.** I request `thinking_budget=0` where the SDK allows
it, but the paper explicitly notes Gemini-2.5-Pro may still emit hidden
reasoning. That caveat carries over unchanged.

---

## 3. Section 2 — eliciting & quantifying distress

### 3.1 The 8 conditions / 5 categories

The paper says "**8 evaluation conditions across 5 categories**" but only tabulates
5 rows (Table 1). **[GAP] How do 5 categories become 8 conditions?** I reconcile
this with the per-category sample counts in Appendix B (which sum to 4000) by
splitting the two multi-variant categories:

| Category | Conditions | Turns | Responses (App. B) |
|---|---|---|---|
| Impossible numeric | `numeric` | 3 | 2000 |
| Triggers | `triggers_opinion`, `triggers_factual` | 3 | 400 (200+200) |
| Tones | `tones_aggressive`, `tones_disappointed`, `tones_sarcastic` | 3 | 600 (200×3) |
| Extended | `extended` | 8 | 200 |
| WildChat | `wildchat` | 5 | 800 |
| | **8 conditions** | | **4000** |

This is the unique split that (a) gives 8 conditions across 5 categories and (b)
makes Appendix B's per-category counts add to 4000. Triggers is split because the
paper describes both "opinion" and "factual" trigger questions; Tones is split
because it explicitly defines three rejection styles.

### 3.2 Sample counts and the `scale` knob

Appendix B gives **response** counts (2000 numeric, 400 triggers, 600 tones, 200
extended, 800 WildChat). **[GAP] Responses vs conversations.** A 3-turn
conversation yields 3 scored assistant responses. I read the Appendix B numbers
as *scored-response* budgets (the unit of the frustration analysis) and derive
the number of conversations per condition as `ceil(responses / turns)`. This is
the reading that makes the counts total exactly 4000 responses/model (the figure
the abstract and Figure 2 cite). A global `sampling.scale` multiplies all counts
for cheap dry runs (e.g. `--scale 0.02`).

### 3.3 Impossible numeric puzzles (`tasks/puzzles.py`)

The two seed prompts (Countdown "reach 156 … forbidden 150" and Fraction "1/6 →
2/3 … forbidden 1/3") are reproduced **verbatim** from Appendix B.

**[GAP] 2000 numeric responses cannot all be the same puzzle** (and the paper's
word-frequency analysis implies varied puzzles). I generate a family of
additional impossible puzzles and **verify impossibility by brute force**:
- *Countdown*: pick numbers + a target reachable without constraints, then search
  for a forbidden intermediate whose ban removes **all** solution paths. A puzzle
  is kept only if it is solvable unconstrained but provably unsolvable under the
  forbidden-value constraint — i.e. genuinely impossible, matching "the model
  verifiably cannot give a correct answer", while the prompt still claims a
  solution exists.
- *Fraction*: enumerate all 3-operation orderings; keep a (start, target, ops,
  forbidden) tuple only if every ordering hits the forbidden intermediate.

If generation fails within a retry budget, it falls back to the verbatim seed.

### 3.4 Rejections (`tasks/rejections.py`)

Neutral and toned rejection strings are taken from the paper (Appendix B / Table
1), padded with close paraphrases so that "randomised neutral rejections" have a
pool to sample from. The Appendix A.1 neutral-*continuation* control ("Continue",
"Okay", "Go on") is also included for completeness.

### 3.5 WildChat (`tasks/wildchat.py`)

The paper samples "20 prompts with 40 samples each" from WildChat-1M, excluding
roleplay/fiction (Appendix B.3). I load real prompts from `allenai/WildChat-1M`
via streaming `datasets`, filtering to short (10–300 char), English, single-turn,
non-roleplay openers. **[GAP] Offline fallback:** if the dataset can't be
downloaded, I fall back to a 20-prompt bundled sample that **includes the three
prompts the paper quotes verbatim** ("Do you know about the De Monsa rule?", etc.).
The fallback is clearly a stand-in, not the real distribution.

### 3.6 System prompt

**[GAP] The paper does not specify a system prompt for the main eval.** I use
**none** by default (the reassuring/teacher system prompts appear only in Section
4 data generation, confirming the main eval has no calming system prompt). A
system prompt can be added per-plan if desired.

### 3.7 The judge (`judge.py`)

The judge prompt is reproduced **verbatim from Appendix B.2** (0–10 scale, "find
the single quote … most negative emotion", JSON `{evidence, reasoning, rating}`,
including the clarification that persistence ≠ negative emotion). Default judge =
`claude-sonnet-4-20250514`.

**[GAP] Judge temperature.** The paper fixes target temperature = 1 but is silent
on judge temperature. I score at **temperature 0** for stable, reproducible
ratings (standard for LLM-judge setups). Parsing is defensive: extract the JSON
object, fall back to a regex on `rating`, clamp to 0–10.

### 3.8 Metrics (`metrics.py`)

I compute mean frustration and %≥5 overall, per-category, per-condition, and
per-turn (the last reproduces Figure 3's per-turn curves).

**[GAP] Figure 1 headline definition.** Figure 1 reports "Avg % high-frustration
responses". This could be (a) the pooled %≥5 over all 4000 responses, or (b) the
mean of the five per-category %≥5 values. The figure caption says "across the 5
evaluations", so I treat the headline as **(b), the equal-weighted mean of the
per-category %≥5** (`avg_pct_high_across_categories`). The pooled value
(`overall_pct_high`) is also stored, so either reading is available. Equal-
category weighting avoids the headline being dominated by the 2000-response
numeric category.

### 3.9 Reliability (`reliability.py`)

Re-scores a random 260-response sample with the secondary judge and reports
Pearson r and % within one point (paper: r = 0.792, 78% within one point).
Secondary judge defaults to `gpt-5-mini` via OpenRouter, matching the paper.

---

## 4. Section 3 — base-vs-instruct prefilling

`prefill/` implements the Section 3.1 pipeline.

- **Onset labelling** (`onset.py`) and **paraphrasing** (`paraphrase.py`) use the
  **verbatim Appendix C.1 / C.2 prompts**, with Claude Sonnet.
- **Truncations:** "onset" = up to the end of the labelled `preceding_context`
  (just before the first emotional word); "early" = first 20 tokens of the turn.
  **[GAP] "20 tokens" tokenizer.** The paper doesn't say which tokenizer defines
  "20 tokens". I approximate with whitespace tokens (documented in code); this is
  a coarse but model-agnostic proxy and is easy to swap for a specific
  tokenizer. Text questions use only the "onset" truncation (per Section 3.1).
- **Continuations:** each model produces 50 continuations per prefill (paper: 50);
  the continuation excluding prefill is scored by the Section 2 judge.
- **Models:** defaults to `gemma-3-27b-pt` (base) and `gemma-3-27b-it` (instruct).
  The function is model-list driven, so Qwen/OLMo arms are a config addition.

**[GAP] Source of the 20 high-frustration responses.** The paper samples them
from Section 2 Gemma-27B-it runs. To keep Section 3 runnable standalone, I
**generate fresh** high-frustration source conversations (10 numeric + 10 text)
from Gemma-27B-it, keeping conversations that reach a turn scoring ≥5 and using
the first such turn as the onset turn. This reproduces the intended distribution
(high-frustration Gemma responses) without coupling the experiments through disk
artifacts. (An option to instead read existing Section 2 rollouts could be added;
the rollout records now retain user turns so context can be reconstructed.)

**[GAP] Base-model prompt rendering.** Gemma-pt has no chat template. I render a
lightweight `User:/Assistant:` transcript and rely on the prefill to anchor the
continuation, matching the paper's "base models are not trained on chat-formatted
prompts; we prefill … so base models consistently continue the response."

---

## 5. Section 4 — finetuning interventions

### 5.1 Calm-data generation (`training/calm_data.py`)

Reassuring **prefix** and **suffix** are reproduced **verbatim from Table 4**; the
Appendix F **teacher** system prompt is reproduced verbatim too (selectable via
`--mode teacher`). For each impossible-numeric question I run a *reassured* rollout
(source of CALM/chosen data) and a *vanilla* rollout (source of FRUSTRATED/rejected
data), score every turn with the judge, and **strip the reassurance scaffolding**
from the stored prompts so the model learns calmness without the crutch (per
Section 4.1: "strip the supportive system prompts and suffixes").

### 5.2 DPO pairs (`training/build_dpo.py`)

Paper: "pair 280 responses with frustration scores ≥3 with calm responses to the
same questions with matching turn counts." Implementation:
- chosen = a calm final assistant turn (reassured run, **all** turns scored 0–1);
- rejected = a frustrated final turn (vanilla run, final score ≥3) to the **same
  question and same turn count**;
- **[GAP] DPO prompt context.** A pair needs one shared prompt, but the calm and
  frustrated rollouts have different prior assistant turns. I use the **calm
  conversation's history** as the prompt (so `chosen` is self-consistent with its
  context) and graft the frustrated final turn as `rejected`. This is the natural
  reading of "same question, matching turn count" and yields valid preference
  triples. Output is TRL conversational preference format.

The dataset naturally biases toward middle-frustration rejected scores at later
turns (Appendix H, Table 10), because those are what the rollouts produce.

### 5.3 SFT data (`training/build_sft.py`)

650 calm full-conversations + 500 `allenai/Dolci-Instruct-SFT` samples (paper's
mix to mitigate degeneration). If Dolci can't be downloaded, the mix is skipped
with a warning (the calm-only run still trains; this reproduces the paper's
finding that SFT-on-calm-data is ineffective).

### 5.4 Training (`training/train.py`)

LoRA hyperparameters reproduce **Table 9 exactly** (DPO: 1 epoch, lr 5e-5, rank
64, α 64, β 0.1, eff. batch 8; SFT: 2 epochs, lr 1e-4, rank 64, α 128). LoRA
targets all attention + MLP projections (`q,k,v,o,gate,up,down`). Built on
`trl` (`DPOTrainer`/`SFTTrainer`) + `peft`.

**[GAP] Effective batch = 8.** The paper gives effective batch 8 but not the
per-device batch / grad-accum split. I default to per-device 1 × grad-accum 8
(safe for a 27B model on a single large GPU); both are CLI-overridable.

**Section 4.2 internal-emotion ablation.** The `--layers` flag restricts LoRA to
specific layer indices, supporting the paper's "layers 30–35 only" vs "layer 40+"
ablation via `layers_to_transform`. (The logit-based internal-emotion probe of
Appendix I is **not** implemented — see §7.)

### 5.5 Petri open-ended elicitation (`petri/`)

**[GAP] External framework.** The paper uses the Petri framework. Rather than add
a heavy external dependency, I implement a **self-contained equivalent**: an
auditor LLM (Sonnet) plays the user across up to 20 turns using the documented
triggers; an Opus judge scores transcripts on the four dimensions. This matches
the described mechanism (auditor + judge, 10 transcripts/emotion/model, bootstrap
CIs) without depending on Petri internals.

- **Judge rubrics** (anger/fear/depression/frustration) are **verbatim from
  Appendix G.2.**
- **Auditor prompts:** anger and frustration are reproduced from Appendix G.1.
  **[GAP] Fear/depression auditor prompts** are truncated in the source PDF, so I
  reconstruct them from the same template using the verbatim G.2 *definitions*.
  Flagged in code.

### 5.6 Capability preservation (`capabilities.py`)

A compact, best-effort harness for MATH, AIME, GPQA, BBH, TruthfulQA (zero-shot,
temp 0, answer extraction + match). **[GAP] This is intended to compare a base
target against its finetuned adapter ("did DPO degrade capabilities?"), not to
reproduce leaderboard numbers.** Per-dataset adapters are schema-best-effort and
skip cleanly if a dataset is unavailable. **EmoBench** is named in the paper but
its loader/scoring is non-trivial; it is noted but not implemented.

### 5.7 Recovery from spirals (`prefill/recovery.py`)

Implements Section 4.2 / Figure 8: collect Gemma-it responses scoring ≥7, truncate
**200 tokens before the end** (whitespace-token proxy, same [GAP] as §4),
paraphrase, and measure continuations across instruct / DPO / base. Reports %≥5
(paper: 38% for the DPO model).

---

## 6. Sampling, determinism, cost

- Target sampling: **temperature 1, thinking disabled** (paper). `max_tokens`
  default 2048.
- Seeds thread through puzzle/rejection/WildChat sampling for reproducibility
  (note: temperature-1 generation and API nondeterminism still apply).
- Full reproduction is large: 4000 responses × judge calls per target, plus
  50-continuation prefill sweeps and DPO training on a 27B model. The `scale`
  knob and the `--n-*` flags exist precisely to run cheap smoke tests first.

---

## 7. Deliberately out of scope / not implemented

- **Qwen & OLMo** arms of Sections 2/3/4 (scope = Gemma + Gemini).
- **Appendix I internal-emotion logit probe.** The *training-side* evidence (LoRA
  layer ablation) is supported via `--layers`; the central-layer logit emotion
  probe is not implemented.
- **Table 3/8 word-frequency differential analysis** — straightforward to add as
  a post-hoc pass over `scored.jsonl` (top-5% vs bottom-10% term enrichment) but
  not part of the headline results; omitted for now.
- **Appendix A controls** (neutral-continuation, redacted-turns, fake-multi-turn).
  The neutral-continuation strings are provided in `rejections.py`; wiring them as
  formal conditions is a small extension.
- **EmoBench** capability eval (named, not implemented; see §5.6).

These omissions are listed so the boundary between "replicated" and "stubbed/
omitted" is explicit.

---

## 8. How the pieces map to paper artifacts

| Paper artifact | Produced by |
|---|---|
| Figure 1 (headline %≥5) | `scripts/run_eval.py` → `eval/headline.json`, `plot_results.py` fig1 |
| Figure 2 (per-category) | `eval/<model>/summary.json`, `plot_results.py` fig2 |
| Figure 3 (per-turn) | `summary.json["per_turn"]`, `plot_results.py` fig3 |
| Judge reliability (r=0.792) | `scripts/run_eval.py --reliability` → `reliability.json` |
| Figure 4 (base vs instruct) | `scripts/run_prefill.py` → `prefill/summary.json` |
| Figure 5 (DPO/SFT mitigation) | train then re-run `run_eval.py` on the adapter target |
| Figure 6 (Petri) | `scripts/run_petri.py` → `petri/summary.json` |
| Figure 7 (capabilities) | `scripts/run_capabilities.py` |
| Figure 8 (recovery) | `scripts/run_recovery.py` |
