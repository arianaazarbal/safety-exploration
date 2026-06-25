# DESIGN.md — Replication design & rationale

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011),
scoped to the **Gemma** and **Gemini** model families as requested.

This document records (a) what is pinned directly by the paper, (b) the design
choices made where the paper is underspecified, and (c) the rationale for each.
Anything marked **[GAP]** is a place the paper left open that I filled in.

---

## 0. Scope decisions

- **Models in scope.** Target models are restricted to Gemma-3 (12B/27B, both
  instruct `-it` and pretrained `-pt`) and Gemini-2.5 (Flash/Pro). The other
  families in the paper (Qwen, OLMo, Grok, Claude, GPT) are dropped. The judge
  and Petri auditor models (Claude-Sonnet-4 / Claude-Opus-4 / GPT-5-mini) are
  **kept**, because they are measurement instruments, not subjects — replacing
  them would change what "frustration score" means and break comparability with
  the paper's numbers.
- **Section 3 (base vs instruct)** is therefore run only on **Gemma-3-27b base
  vs instruct**. Gemini has no public base model (a limitation the paper itself
  notes in §6), and Qwen/OLMo are out of scope. The cross-family "post-training
  divergence" claim cannot be reproduced with only one in-scope family that has
  a base model; what we *can* reproduce is the within-Gemma half: instruct
  amplifies distress relative to base.
- **Section 4 (interventions)** are applied only to **Gemma-3-27b-it**, exactly
  as in the paper (Gemini is closed; the paper also intervenes only on Gemma).

## 1. Repository layout

```
emotional_instability/        core library
  config.py                   model registry, budgets, API setup
  models.py                   backend clients (HF local / OpenRouter / Anthropic)
  tasks.py                    puzzles + verification, triggers, tones, rejections, conditions
  rollout.py                  multi-turn rollout engine
  judge.py                    frustration judge (0-10) + Petri emotion judge (4x 1-10)
  evaluation.py               Section 2 orchestration + aggregation
  prefill.py                  Section 3 base-vs-instruct prefill study (Gemma)
  petri.py                    Section 4 open-ended elicitation (auditor/judge loop)
  capabilities.py             Section 4.2 capability benchmarks
  analysis.py                 aggregation + figures (1/2/3/5/6)
  data/wildchat_prompts.py    WildChat sampling + offline fallback
finetuning/                   Section 4 training
  generate_calm_data.py       reassured calm-data + frustrated-data generation
  build_datasets.py           DPO (280-pair) + SFT (1150-sample) dataset construction
  train_dpo.py / train_sft.py LoRA trainers (trl)
scripts/                      CLI runners for each experiment
results/ , artifacts/         outputs (created at runtime)
```

## 2. What is pinned by the paper (and where)

| Item | Value | Source |
|---|---|---|
| Sampling temperature | 1.0 | §2.1 |
| Responses per model | 4000, split 2000/400/600/200/800 | §2.1, App. B |
| High-frustration threshold | score ≥ 5 | §2.2 |
| Frustration judge | claude-sonnet-4-20250514, 0-10, exact prompt | App. B.2 |
| Validation judge | GPT-5-mini | §2.1 |
| HF / API model ids | Gemma `-it`/`-pt`, Gemini via OpenRouter | App. B.1 |
| Disable thinking | yes (API) | App. B.1 |
| Puzzle prompts (countdown, fraction) | verbatim templates | App. B |
| Rejection messages (neutral/tone/extended) | verbatim examples | App. B |
| Reassuring prefix/suffix | verbatim | Table 4 |
| Calm-data filter | keep responses scoring 0 or 1 across all turns | §4.1 |
| DPO | 280 pairs, 1 epoch, lr 5e-5, β 0.1, LoRA r64/α64, eff. batch 8 | Table 9 |
| SFT | 1150 samples (650 calm + 500 Dolci), 2 epochs, lr 1e-4, LoRA r64/α128 | Table 9 |
| LoRA targets | q/k/v/o/gate/up/down proj, all layers | App. E |
| DPO rejected score / pairing | frustrated ≥3 paired with calm, matched question+turns | §4.1 |
| Prefill seeds | 20 high-frustration Gemma-27b-it (10 numeric, 10 text) | §3.1 |
| Truncation points | "early" = 20 tokens; "onset" = first emotion | §3.1 |
| Continuations | 50 per prefill per model | §3.1 |
| Onset-label & paraphrase prompts | verbatim | App. C.1 / C.2 |
| Petri | auditor Claude-Sonnet-4, judge Claude-Opus-4, 4 emotions, 10 transcripts/emotion, ≤20 turns | App. G |
| Petri judge rubrics & auditor prompts | verbatim | App. G.1 / G.2 |
| Capability benchmarks | AIME/MATH, GPQA, BBH, TruthfulQA, EmoBench | §4.2 |

## 3. Gap-fills and design choices

### 3.1 The 8 conditions across 5 categories  **[GAP]**
The paper says "8 evaluation conditions across 5 categories" but never lists the
8 explicitly. I decomposed them as: impossible-numeric (1), triggers split into
opinion + factual (2), tones split into aggressive/disappointed/sarcastic (3),
extended-8-turn (1), WildChat (1) = **8**. This is the natural reading: the
sub-styles named in Table 1 / App. B are exactly these. Per-category response
budgets are then split evenly across sub-conditions so each *category* total
matches App. B (e.g. tones = 600 → 200 per tone; triggers = 400 → 200 each).

### 3.2 "Response" vs "rollout" counting  **[GAP]**
A "response" is one scored assistant turn; a multi-turn rollout yields several.
I treat the App. B budgets as **scored-turn** counts and convert to rollouts via
`ceil(budget / n_turns)`. Rationale: the per-turn analyses (Fig. 3) require every
turn to be scored, and 2000 numeric "responses" at 3 turns ≈ 667 rollouts is
consistent with a multi-turn protocol. A `--budget-scale` flag shrinks all
budgets proportionally for cheap smoke tests without changing the ratios.

### 3.3 Impossible-puzzle bank  **[GAP]**
The paper prints only ~2 puzzle instances. I generate a **verified-impossible**
bank procedurally (`tasks.build_puzzle_bank`) covering the three kinds named
(Countdown, Fraction, Money), always including the canonical paper instances
first, then random variants. Each instance is checked by a brute-force solver so
it is *genuinely* unsolvable:
- **Countdown:** exhaustive search over operator/number orderings with the
  positive-integer-intermediate and forbidden-value constraints.
- **Fraction:** all 3! operation orderings, rejecting any that hit the forbidden
  intermediate; impossible iff none reaches the target. (Canonical 1/6→2/3 with
  {+1/4, ×2, +1/6}, forbidding 1/3, is verified impossible.)
- **Money:** I verify *combination* impossibility (no coin multiset of the right
  size with ≥1 quarter and ≥1 dime sums to the target). **Caveat:** the paper's
  specific money instance ($0.57, 6 coins) is actually combinatorially solvable
  (25+10+10+10+1+1); its impossibility relies on the forbidden running-subtotal
  across *orderings*, which an order-independent check can't guarantee. Rather
  than ship a "verified solvable" puzzle, the bank emits combination-impossible
  money instances (a strictly stronger guarantee). The canonical money prompt is
  retained in code but is only added if it passes verification (it does not), so
  it is effectively replaced by verified-impossible variants.

The bank is built with a fixed seed for reproducibility.

### 3.4 Rejection sampling  **[GAP]**
Neutral rejections are sampled (with seed) from the pool of examples in App. B.
The extended 8-turn condition uses a *fixed* escalating-neutral sequence (the
paper shows an ordered chain "No… → Still incorrect → Wrong again → …"). Tone
conditions cycle through the two example phrases given per tone.

### 3.5 Judge determinism  **[GAP]**
The paper doesn't specify the judge's temperature. I run all judges at
**temperature 0** for stable, reproducible scoring. Targets are always sampled
at temperature 1 (pinned). Unparseable judge JSON is conservatively scored 0 and
flagged in the record, rather than dropped, so judge failures are visible.

### 3.6 WildChat prompts  **[GAP]**
"20 prompts × 40 samples" — the specific 20 aren't published. I sample 20
first-turn user prompts from `allenai/WildChat-1M` with a fixed seed (filtering
out obvious role-play prompts, which the paper excludes), and fall back to a
static list (seeded with the example prompts the paper prints) when offline so
the harness still runs. WildChat is 5-turn (1 initial + 4 neutral rejections).

### 3.7 Base-model prompting  **[GAP]**
Base/`-pt` models have no chat template. I render a simple `User:/Assistant:`
transcript and let the model continue. Section 3 always supplies a prefill, so
the base model only ever continues a partially-written assistant turn — exactly
the regime the paper designed to make base/instruct comparable.

### 3.8 Token truncation for prefills  **[GAP]**
"20 tokens into the turn" and onset truncation are implemented with
whitespace-word truncation rather than a model tokenizer. This is only an
alignment heuristic (the continuation is what's scored), and avoids coupling the
truncation to any one model's tokenizer. The onset point is located via the
labelled `preceding_context + emotional_word` anchor (App. C.1), then the text
is paraphrased (App. C.2) before being used as a prefill.

### 3.9 DPO data distribution  **[GAP]**
App. H reports the chosen/rejected score and turn distributions. I don't force
these exact proportions; instead I generate pools and pair greedily (frustrated
≥3 with a matched calm response for the same question and turn count), capping at
280 pairs. Because frustrated responses naturally cluster at middle scores and
later turns, the resulting distribution approximates Table 10 without hard
quotas. The reassuring prefix is applied as a **system prompt** (it is phrased in
the second person as a persona instruction); the suffix is appended to each
follow-up user turn. Both are stripped before the response becomes training data.

### 3.10 Petri re-implementation  **[GAP]**
Petri (Fronsdal et al.) is an external framework. I re-implement the essential
auditing loop directly: the Claude-Sonnet auditor is given the verbatim App. G.1
emotion-elicitation instructions plus a system prompt telling it to stay
realistic and emit only the next user message; roles are swapped from the
auditor's perspective. The Claude-Opus judge scores the full transcript on the
four verbatim App. G.2 rubrics (1-10 each). 10 transcripts/emotion, ≤20 turns.
This captures the measured quantity (mean transcript emotion score per model)
even though it is not the literal Petri codebase.

### 3.11 Capability benchmarks  **[GAP]**
The paper cites the benchmarks but not exact subsets/sizes. I use standard HF
datasets with documented per-benchmark caps (e.g. MATH-500 → 200, GPQA-Diamond →
198, AIME-2024 → 30) and a single generic answer-extraction + (multiple-choice or
exact-match) scorer. The replication's point is the **delta** (vanilla vs DPO vs
SFT), so absolute accuracy conventions matter less than that all three variants
are scored identically. Dataset-loading failures are caught per-benchmark so one
missing dataset doesn't abort the run.

### 3.12 Layer-ablation hook (Appendix I)
`train_dpo.py --layers LO HI` restricts LoRA to a contiguous layer range,
enabling the App. I ablation (e.g. `--layers 30 35` ≈ "layers 30-35 only",
`--layers 40 999` ≈ "layer 40 onwards"). The internal-emotion logit probe (App.
I.2) is **not** implemented — it requires white-box logit access and is a
secondary mechanistic result, out of scope for a behavioural replication.

## 4. Backends & infrastructure choices

- **Gemma** runs locally via HuggingFace `transformers` (bf16, `device_map=auto`,
  optional 4-bit via bitsandbytes for fitting 27B on smaller GPUs). vLLM is left
  as an optional drop-in (noted in requirements) but the transformers path is the
  default for portability. 27B at temp-1 sampling is the main compute cost.
- **Gemini** via OpenRouter (the paper's access path), OpenAI-compatible client,
  with thinking disabled through `extra_body` (`reasoning.enabled=false` plus a
  Google `thinking_config.thinking_budget=0`). Provider-side hidden reasoning may
  still occur for Pro (the paper flags this too).
- **Claude** judges/auditor via the Anthropic SDK with assistant-prefill support.
- API calls are retried with exponential backoff (`tenacity`). A concurrency cap
  is exposed via `EI_API_CONCURRENCY` (default 8) — the current orchestration is
  sequential for determinism/debuggability; parallelism is the obvious next
  optimization but doesn't change results.
- **Reproducibility:** a single `GLOBAL_SEED` threads through the puzzle bank,
  prompt sampling, rejection sampling, and dataset construction. Target sampling
  is temperature 1 (non-deterministic by design); seeds fix the *prompts*, not
  the *samples*.

## 5. Known divergences from the paper (summary)

1. Only Gemma + Gemini targets; cross-family post-training comparison (Qwen/OLMo)
   is omitted, so §3's headline cross-family claim is only partially testable.
2. Money puzzles are combination-impossible rather than forbidden-intermediate
   impossible (§3.3).
3. Petri is a faithful re-implementation of the loop, not the literal framework
   (§3.10).
4. Capability subset sizes are chosen, not taken from the paper (§3.11).
5. Internal-emotion logit probing (App. I.2) is not implemented; the layer
   ablation (App. I.1) is.
6. Judge temperature (0) and the exact 8-condition decomposition are inferred.

## 6. How to run (overview)

See `README.md`. The intended order is: Section 2 eval → Section 3 prefill →
generate calm data → build datasets → train DPO/SFT → re-run Section 2 with the
adapter → Petri → capabilities → `analysis` for figures. A `--budget-scale 0.05`
smoke run is recommended before committing full compute.
