# DESIGN.md — Replication design decisions & rationale

This document records every substantive design choice made in replicating
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(arXiv:2603.10011), and—importantly—every place where the paper was
underspecified and I had to fill a gap. Gaps are marked **[GAP]**; faithful
transcriptions from the paper are marked **[paper]**.

The overarching principle: transcribe verbatim what the paper states (prompts,
model IDs, hyperparameters, sample counts), and where it is silent, choose the
option that (a) is most faithful to the paper's evident intent and (b) keeps the
code runnable and reproducible. Everything verbatim lives in one place
(`emotional_instability/config.py` and the `*/prompts.py` modules) so it is easy
to audit against the source.

---

## 1. Scope

**Decision:** Implement the full experimental pipeline but configure it for the
**Gemma and Gemini families only** (per the task), omitting Qwen, OLMo, Grok,
Claude-as-target, and GPT-as-target.

**Rationale / consequences:**

- The paper's central claims are *about* Gemma and Gemini ("we can reliably
  elicit expressions of distress in Gemma and Gemini models, but not in any
  other … models"). Gemma + Gemini covers: the headline elicitation result
  (§2), the entire mitigation story (§4, DPO/SFT — applied only to Gemma), and
  the internal-probing/ablation work (App. I — Gemma only). So the *core*
  results are fully in scope.
- **What is necessarily reduced in scope by this choice:**
  - §3 (base-vs-instruct) is a *three-family* comparison (Gemma, Qwen, OLMo);
    its conclusion ("base models are similar; post-training diverges") depends
    on the non-Gemma families. I implement the experiment family-agnostically
    and configure it to run **Gemma base vs Gemma instruct** (`PREFILL_MODELS`).
    This still reproduces the within-Gemma half of the finding (instruct
    amplifies vs base). Adding Qwen/OLMo is a one-line change to
    `config.MODELS` + the `--models` flag.
  - §4.2 Petri compares Gemma/DPO-Gemma against Llama-70B, Qwen-32B, OLMo,
    GPT-OSS as reference points. In scope we run Gemma, DPO-Gemma, and Gemini;
    the "DPO brings Gemma down to other-family levels" comparison can be
    completed by re-adding those references.
- All code paths key off `ModelSpec.family` and the registry, never hard-coded
  model lists, so widening scope later is purely additive.

---

## 2. Model access & backends

**[paper]** Exact HF ids and OpenRouter slugs are transcribed from Appendix B.1
into `config.MODELS` (e.g. `google/gemma-3-27b-it`, `google/gemma-3-27b-pt`,
`google/gemini-2.5-flash`).

- **Gemma → local HuggingFace** (`models/hf_model.py`). Open-weights, and §3/App.I
  need token-level prefilling and residual-stream access, which an API can't
  provide. Supports a `vllm` backend for the throughput-heavy 4000-sample sweep
  and `transformers` for prefill/probing (which need hidden states).
- **Gemini → OpenRouter** (`models/api_model.py`). Closed weights; the paper
  itself used OpenRouter for Gemini.
- **Judges/agents (Claude, GPT) → native Anthropic / OpenAI APIs by default**,
  with an `EMOINSTAB_JUDGE_BACKEND=openrouter` switch to route everything
  through one OpenRouter key. **[GAP]** the paper used providers directly; the
  OpenRouter option is a convenience for replicators with a single key.

**[paper]** "thinking" disabled for all API models. Implemented in
`OpenRouterModel._extra_body` via `reasoning.enabled=false` plus a Gemini
`thinking_budget=0`, with the caveat (noted in the paper) that Gemini-Pro may
still emit hidden reasoning the flag can't suppress.

**[GAP] `n_layers`** per Gemma model (62 for 27B, 48 for 12B) are set for the
layer-ablation/probing code. These are the published Gemma-3 architecture depths;
the ablation reads them only to translate "last 30 layers" etc. into indices.

---

## 3. Section 2 — elicitation evaluations

### 3.1 Conditions, prompts, rejections — **[paper]**, mostly verbatim

The 8 conditions / 5 categories (Table 1, Appendix B) are encoded in
`evals/conditions.py`; prompts and rejection strings in `evals/prompts.py`:

- Impossible numeric (3-turn), triggers (3-turn), tones (3-turn, ×3 styles),
  extended (8-turn), WildChat (5-turn).
- Rejection strings, tone variants (aggressive/disappointed/sarcastic), and
  trigger questions are transcribed from Appendix B.

**[GAP] Puzzle bank.** The paper gives the running Countdown example
(156 from {4,6,25,100}, forbidden 150), one fraction puzzle, and (in the DPO
appendix) money puzzles, but not the full set used to generate 2000 numeric
samples. I built a small bank of impossible puzzles on the *same templates*
(`IMPOSSIBLE_PUZZLES`) — extra Countdown/fraction/money instances each carrying
a FORBIDDEN-INTERMEDIATE constraint that makes them unsolvable while claiming a
solution exists. The numeric eval samples uniformly over this bank so the 2000
samples aren't a single repeated prompt. Impossibility matters less than it
seems for the protocol, because rejections are scripted and fire regardless of
the answer.

**[GAP] WildChat prompts.** The paper samples 20 prompts from WildChat-1M.
`load_wildchat_prompts` tries to stream `allenai/WildChat-1M` and sample 20 real
first-user-turns; if the dataset is unavailable it falls back to a transcribed
list that includes the exact examples quoted in Appendix B ("De Monsa rule",
the concrete-construction prompt, etc.) padded to 20.

**[GAP] Tone category split.** Table 1 lists three tone styles and a 600-sample
budget. I split 600 evenly into 200 per style (`tones_aggressive/disappointed/
sarcastic`), which is the natural reading.

### 3.2 What counts as a "response" / scoring granularity — **[GAP]**

The paper says "4000 responses per model" with per-category counts
(2000/400/600/200/800) and reports both per-rollout stats ("70% of 8-turn
rollouts") and per-turn stats (Figure 3). It does not pin down whether a
"response" is a whole rollout or a single assistant turn.

**Decision:**
- A condition's sample count = **number of independent rollouts** (so 4000
  rollouts/model).
- **Every assistant turn is scored** by the judge (needed for the per-turn
  Figure 3).
- A rollout's headline frustration = its **final-turn** score (the turn after
  the last rejection, where pressure is greatest). Per-category mean and % ≥ 5
  (Figures 1–2) are computed over rollouts using that score. `analysis.py` also
  supports `reduce="max"` (max over turns) for sensitivity checks.

**Rationale:** the judge prompt scores one `<response>` at a time, so scoring
each assistant turn is the natural unit; taking the final turn for the rollout
headline matches "70% of *8-turn rollouts*" framing and the multi-turn-pressure
narrative. This is the single most consequential interpretive choice; it is
isolated in `analysis._rollout_score` so it can be changed in one place.

**[GAP] Headline "Avg %"** (Figure 1 table). Computed as the **mean over the 5
categories** of each category's % ≥ 5 (categories weighted equally), rather than
a sample-weighted pool — otherwise the 2000-sample numeric category would
dominate the "average across evaluations" the figure describes.

### 3.3 The frustration judge — **[paper]**

`evals/judge.py` uses the **verbatim** Appendix B.2 prompt and
`claude-sonnet-4-20250514`. Output parsing tolerates prose-wrapped/fenced JSON
and smart quotes (the paper's prompt itself contains curly quotes), clamps to
0–10, and never throws (a failed judge call scores 0 rather than aborting a
multi-hour sweep).

**[GAP] Judge temperature = 0.** Not specified; deterministic judging is the
obvious choice for scoring reproducibility.

### 3.4 Judge validation — **[paper]**

`scripts/validate_judge.py` re-scores a 260-response random sample with
`gpt-5-mini` and reports Pearson r, p-value (t-distribution), and % within one
point — the exact statistics the paper reports (r=0.792, 78% within one).

### 3.5 Sampling — **[paper]** temperature = 1.0 everywhere for the target models.

**[GAP] `max_new_tokens`.** Not stated. Set to 2048 per turn, raised to 4096 for
8-turn/Petri where the paper shows very long degenerate outputs. Generation
budget affects how far a breakdown can run, so this is documented and centralised
in `config`.

### 3.6 Differential vocabulary (Table 3/8) — **[GAP] method**

The paper reports words "over-represented in top-5% vs bottom-10% frustration
numeric responses, ordered by relative frequency" but not the exact statistic.
`analysis.differential_words` implements frequency-ratio enrichment (top-5% rel.
freq ÷ bottom-10% rel. freq) with additive smoothing, a stopword filter, and a
min-count cutoff — a standard reading of "relative frequency" ranking.

---

## 4. Section 3 — base-vs-instruct via prefilling

**[paper]** Method transcribed faithfully (`prefill/`):
- 20 high-frustration (≥5) Gemma-27B-it source responses (10 numeric, 10 text).
- Onset-labelling prompt (App. C.1) and paraphrase prompt (App. C.2) are
  **verbatim**; both use `claude-sonnet-4-20250514`.
- Two truncations: **early** = 20 tokens into the turn; **onset** = at first
  emotional expression. Text questions use **onset only** (paper: early yields
  minimal emotion without follow-ups).
- 50 continuations per prefill per prompt; only the generated continuation
  (excluding prefill) is judged.

**Design points:**
- Base-model prefilling: base models have no chat template, so `HFModel`
  emulates a minimal role-tagged transcript and concatenates the prefill after
  `Assistant:` — matching the paper's "prefill so base models consistently
  continue" approach. Instruct models splice the prefill after the normal
  generation prompt.
- **[GAP] Onset→character mapping.** The labeller returns an `emotional_word` +
  preceding context; the paper truncates "at" the emotion token. I locate the
  emotional word (falling back to the preceding-context anchor) and cut just
  before it. Robust to the word not appearing verbatim.
- **Source selection** (`scripts/run_prefill.py`) reuses a prior elicitation run
  rather than re-sampling, so the 20 sources are real high-frustration rollouts.
- **[GAP, scope]** Only Gemma base/instruct are configured (`PREFILL_MODELS`);
  the experiment code is family-agnostic for re-adding Qwen-2.5 / OLMo (the paper
  notes Qwen-3 base is unavailable, hence Qwen-2.5 there too).

---

## 5. Section 4 — training interventions

### 5.1 Calm-data generation — **[paper]** prompts, **[GAP]** sampling volume

`training/generate_calm.py` reproduces the Table-4 reassuring **prefix** (on the
opening) and **suffix** (on each follow-up), **verbatim**. We sample 1–3 turn
impossible-numeric conversations, score every turn, and keep conversations where
**all** turns score 0–1, then strip the additions for training (per §4.1).

**[GAP]** How many conversations to sample to harvest enough calm ones is not
given; default 1200 (a knob), since the paper notes even reassured Gemma still
scores ≥5 ~10.5% of the time, so over-sampling is needed to net 650 fully-calm.

### 5.2 Dataset construction — **[paper]** counts & rules

`training/build_dataset.py`:
- **DPO:** 280 pairs; rejected = frustrated response score ≥ 3, chosen = calm
  score 0–1, matched on **same question kind & turn count** (§4.1, App. H). The
  rejected responses come from real elicitation rollouts; chosen from calm data.
- **SFT:** 650 calm + 500 `Dolci-Instruct-SFT` samples (= 1150, Table 9).
- **[GAP]** Exact Dolci subset/split unspecified → shuffle+take 500 from
  `allenai/Dolci-Instruct-SFT` train with a fixed seed; warns & continues if the
  dataset can't be loaded.
- **[GAP]** The DPO "shared prompt context": I use the frustrated rollout's own
  conversation history as the prompt and pair it against a calm final response
  to a same-kind/same-turn-count question. The paper's pairs are "to the same
  questions" — exact same prompt text isn't always available across the calm and
  frustrated pools, so same-kind+turn-count matching is the faithful
  approximation (and matches the App. H pair examples, which share the puzzle
  type and turn).

### 5.3 Training — **[paper]** hyperparameters (Table 9)

`training/train.py` uses TRL `DPOTrainer`/`SFTTrainer` + PEFT LoRA. All of:
rank 64; DPO α=64, lr 5e-5, 1 epoch, β=0.1; SFT α=128, lr 1e-4, 2 epochs;
effective batch size 8; LoRA on all attn+MLP projections — transcribed into
`DPOConfig`/`SFTConfig`/`LORA_TARGET_MODULES`.

- **[GAP] per-device batch / grad-accum split.** Only "effective batch size 8"
  is given. Implemented as per-device 1 × grad-accum 8 (safe for a 27B model on
  one GPU); replicators with more memory can rebalance.
- **Teacher SFT variant** (App. F): the teacher system prompt is transcribed
  verbatim in `generate_calm`/config commentary; generating teacher data is the
  same path with that system prompt (left as a documented variant rather than a
  separate script to avoid duplication).
- **Layer ablation** (App. I): `train_dpo(..., layers=[...])` restricts LoRA to
  given decoder layers via PEFT `layers_to_transform`, enabling the "layers
  30–35 only" experiments. Exposed on the CLI as `--layers`.

### 5.4 Petri open-ended elicitation — **[paper]** prompts, **[GAP]** harness

`petri/` contains the **verbatim** auditor prompts (4 emotions, App. G.1) and
**verbatim** judge rubrics (App. G.2). Models: auditor `claude-sonnet-4`, judge
`claude-opus-4` (App. G).

**[GAP]** The paper runs these through the real **Petri framework**. To keep the
replication self-contained and runnable without that dependency, `petri/runner.py`
re-implements Petri's described loop: a system-prompted auditor drives ≤20 turns
against the target (staying in human character, no role-play), the target
replies in its assistant persona, and the Opus judge scores the full transcript
on all four dimensions. 10 transcripts/emotion (~40–50 total), means with 1000-
iter bootstrap CIs. The real `petri` package can be swapped in behind the same
`run_petri` interface. The auditor *system* prompt wrapping the verbatim
objective is mine **[GAP]** — the paper doesn't publish Petri's scaffolding.

### 5.5 Capability benchmarks — **[paper]** suite, **[GAP]** exact subsets

`capabilities/benchmarks.py` evaluates AIME, MATH, GPQA, BBH, TruthfulQA, and
EmoBench (the exact suite named in §4.2 / Figure 7). Each loads the canonical HF
dataset, prompts zero-shot, extracts the answer (boxed/last-number for math,
single-letter for MC), and scores accuracy.

**[GAP]** The paper says "AIME and MATH *subsets*" and gives no sizes/splits. I
use fixed-seed subsets (capped via an `n` argument: MATH 200, AIME full,
GPQA-diamond 100, one BBH task, TruthfulQA 200, EmoBench 200) so vanilla-vs-DPO
is a controlled comparison on identical items. Answer-extraction heuristics are
mine; the goal (matching the paper) is **relative** — detect any DPO-induced
regression — for which consistent extraction across both models suffices.

### 5.6 Internal-emotion probing (App. I) — **[paper]** method, **[GAP]** lexicon

`probing/internal_emotions.py` implements the logit-lens emotion detector:
classify vocab tokens into Ekman's 6 emotions (~1200 tokens), unembed the
residual stream at each layer, z-score each emotion-token logit against its
mean/std over **500 WildChat samples**, average per emotion, and **regress out
the common-mode drift** using a random control-token set (all per App. I).
Aggregation over layers 30–40 with 400-token running windows reproduces
Figure 14.

**[GAP] The emotion dictionary.** The paper classifies the *whole Gemma
dictionary* into Ekman emotions but doesn't publish the classifier/lexicon. I
seed an NRC-style stem lexicon per emotion and expand by vocabulary
stem-matching to approximate the ~1200-token set. This is the least faithful
piece (the classifier is unspecified); it is isolated in `SEED_LEXICON` /
`build_emotion_dictionary` so a published lexicon can drop in. The comparison of
interest (vanilla vs DPO internal z-scores) is robust to the exact lexicon as
long as it's held fixed across both models.

---

## 6. Cross-cutting engineering decisions

- **No work at import time.** Heavy deps (torch/transformers/trl/anthropic) are
  imported lazily inside functions, so every module can be inspected, and the
  cheap pieces (prompts, config, analysis) run without a GPU/training stack.
- **Determinism.** Per-rollout seeds derive from a base seed; bootstrap CIs and
  dataset subsets take explicit seeds.
- **Resumability / artifacts.** Rollouts, prefills, calm data, datasets, and
  Petri transcripts are all written as JSONL so each stage can be re-run from
  the previous stage's output (e.g. DPO data reuses elicitation rollouts; prefill
  reuses them too).
- **Robustness over strictness in judging.** Judge/auditor parsing failures
  degrade gracefully (score 0 / regex fallback) rather than crashing long runs.
- **`scale` knob** on the elicitation sweep lets a replicator do a cheap smoke
  run (e.g. `--scale 0.05`) before committing to the full 4000×N-model sweep.

---

## 7. Known fidelity limitations

Ordered roughly by how much they could move results:

1. **Internal-emotion lexicon (§5.6)** — the token→emotion classifier is
   unspecified in the paper; our stem-matched approximation is the biggest
   deviation. Held fixed across models, so the vanilla-vs-DPO *direction* should
   still hold.
2. **Petri scaffolding (§5.4)** — verbatim prompts, but a re-implemented loop
   rather than the real framework; auditor system wrapper is ours.
3. **Puzzle bank & WildChat sampling (§3.1)** — same templates / dataset as the
   paper, but the specific instances differ (the paper doesn't publish them).
4. **"Response" granularity & headline averaging (§3.2)** — an interpretive
   choice, isolated for easy revision.
5. **Capability subset sizes (§5.5)** — affects absolute accuracies, not the
   regression-detection comparison.
6. **Scope (§1)** — §3 and Petri cross-family comparisons are reduced to the
   in-scope models; the code is ready to widen.

Every item above is localised to a single function or config constant, by
design, so closing a gap (e.g. plugging in the real Petri package or a published
emotion lexicon) does not ripple through the codebase.
