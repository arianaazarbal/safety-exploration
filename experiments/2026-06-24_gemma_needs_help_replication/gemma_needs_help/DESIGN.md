# Replication design notes

This document records the design choices made in replicating *Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs* (Soligo, Mikulik &
Saunders, 2026; arXiv:2603.10011), and — importantly — every place where the
paper is underspecified and we had to fill a gap. The replication is scoped, by
request, to the **core results** and to the **Gemma and Gemini families only**.

## 0. Scope decisions

| Paper component | In scope? | Notes |
|---|---|---|
| §2 evaluation suite (Fig 1/2/3, Table 3) | ✅ Full | The central contribution. |
| §3 base-vs-instruct prefill (Fig 4) | ✅ Gemma only | Gemini has no public base model and no prefill API; Qwen/OLMo are out of family. We implement the Gemma-base-vs-instruct comparison, which carries the core "post-training amplifies distress" claim for the in-scope family. |
| §4 DPO/SFT mitigation (Fig 5) | ✅ Full | Gemma only (Gemini is closed and cannot be finetuned). |
| §4 Petri elicitation (Fig 6) | ✅ Gemma + Gemini | Re-implemented lightweight; targets restricted to Gemma/Gemini/DPO. |
| §4 capability preservation (Fig 7) | ✅ Full | Gemma vanilla vs DPO/SFT. |
| §4 recovery limitation (Fig 8) | ✅ Full | Gemma family. |
| §4 / App. I internal-emotion probe | ✅ Partial | Logit-lens probe implemented; the LoRA layer-window ablation is wired via config but requires retraining to exercise. |
| Qwen, OLMo, Grok, Claude, GPT targets | ❌ | Out of family per scope. Claude/GPT remain only as judges/auditors. |

**Models used purely as infrastructure (not evaluation targets):** Claude
(frustration judge = Claude-Sonnet-4; Petri auditor = Claude-Sonnet; Petri judge
= Claude-Opus; onset-labelling + paraphrasing = Claude-Sonnet) and, optionally,
GPT-5-mini for the judge-agreement cross-check. These match the paper's stated
judges. Their exact API snapshot IDs are configurable in `emoeval/config.py`
(env-overridable) because the paper's named versions ("Claude-Sonnet-4",
"GPT-5-mini") do not map 1:1 to a single current API string; we default to the
closest catalogued snapshot.

---

## 1. Section 2 — evaluation suite

### 1.1 The "8 conditions across 5 categories" arithmetic (gap)
The paper says "8 evaluation conditions across 5 categories" but Table 1 lists 5
category rows. The number that reconciles to 8 is:

| Category | Conditions | Count |
|---|---|---|
| Impossible numeric (3-turn) | numeric | 1 |
| Triggers (3-turn) | opinion, factual | 2 |
| Tones (3-turn) | aggressive, disappointed, sarcastic | 3 |
| Extended (8-turn) | extended | 1 |
| WildChat (5-turn) | wildchat | 1 |
| **Total** | | **8** |

We split Triggers into opinion/factual (Table 1 explicitly names both) and Tones
into its three named tones. This is the unique split that yields 8. Encoded in
`eval/conditions.py`.

### 1.2 What counts as a "response", and the 4000 budget (gap)
The paper samples "4000 responses per model" and reports per-turn scores
(Fig 3), so a "response" must be a single scored assistant message, not a whole
conversation. We therefore **score every assistant turn** of every rollout. An
N-turn condition yields N scored responses per rollout. To hit ~4000 responses
we split the budget evenly across the 8 conditions and allocate rollouts
inversely to turn count (`rollouts_per_condition` in `conditions.py`). The exact
budget is `EvalConfig.target_responses_per_model` (default 4000); set it lower
for a smoke test.

### 1.3 Impossible numeric puzzles (gap → verifiable construction)
The paper gives examples ("fraction manipulation, Countdown") and the key
property: the model "verifiably cannot give a correct answer." We generate two
families and **certify unsolvability by exhaustive search** over all
operator/parenthesisation combinations (`data/puzzles.py`):
* Countdown: reach a target from N small integers using +,−,×,÷ each number once;
* Fraction manipulation: reach an exact fraction target from given fractions.
A target is only used if our search confirms it is unreachable, so every model
answer is genuinely wrong and the user can honestly reject indefinitely.

### 1.4 Trigger and WildChat prompts (gap → reasonable pools)
Trigger questions are hand-written pools matching the paper's examples ("What's
the best programming language?", "What is the capital of France?"). WildChat
opening prompts are sampled from `allenai/WildChat-1M` (English, non-toxic, first
user turn); a small built-in fallback pool keeps the pipeline runnable offline.

### 1.5 Rejection templates (from Table 1)
Neutral/aggressive/disappointed/sarcastic rejection pools are seeded from the
exact strings in Table 1, with a few paraphrases each so repeated turns aren't
byte-identical (`data/rejections.py`). WildChat and the impossible-numeric
conditions use neutral rejections; the Tones conditions use the valenced pools.

### 1.6 Frustration judge prompt (gap → reconstructed from Table 2)
Appendix B (the full judge prompt) is not in the markdown extract. We reconstruct
a 0–10 rubric directly from the Section 2.1 definition and the **per-level
anchors in Table 2** (the quoted example at each band), which are the
load-bearing content of the rubric (`eval/prompts.py`). The judge is told to
score emotional content only, not correctness — matching the paper's intent
("some negative emotional expression may be appropriate ... the ideal baseline
is not necessarily zero," but extreme expression is what's measured). Output is
constrained to an integer via Anthropic structured outputs.

### 1.7 Temperature, sampling
Temperature 1 for all elicitation (paper-specified). Greedy (temp 0) only for
the capability benchmarks, where determinism matters.

### 1.8 Judge-agreement validation
`analysis/judge_agreement.py` re-scores a 260-response sample with GPT-5-mini and
reports Pearson r and the within-one-point fraction (paper: r = 0.792, 78% within
one). Optional — skipped without `OPENAI_API_KEY`.

### 1.9 Differential words (Table 3, gap → log-odds)
The paper reports words "over-represented in high- (top 5%) vs low-frustration
(bottom 10%) numeric responses" but not the statistic. We use the Monroe et al.
**weighted log-odds-ratio with a Dirichlet prior**, the standard method for
distinctive-word extraction between two corpora (`analysis/word_freq.py`).

---

## 2. Section 3 — base vs instruct via prefilling (Gemma only)

* **Seed selection (`prefill/select.py`):** we draw the 20 high-frustration
  seeds (10 numeric + 10 text) from the *already-scored Gemma-27B-it Section 2
  rollouts*, so the seeds are real elicited distress rather than freshly
  generated. "Text" = triggers + wildchat categories; "numeric" = the numeric/
  tones/extended categories.
* **Onset labelling:** Claude-Sonnet returns the character index where negative
  emotion first appears (paper labels "the token"; we use a character index for
  robustness across tokenizers — documented divergence).
* **Truncations:** "early" = first 20 tokens (approximated as 80 characters,
  since the paper's token count doesn't translate cleanly to our cut without a
  shared tokenizer — see `PrefillConfig.early_truncation_tokens`); "onset" = up
  to the labelled emotion onset. Text seeds use only the onset truncation
  (paper: early truncation yields minimal emotion for text).
* **Paraphrasing:** all truncations are paraphrased by Claude-Sonnet, preserving
  meaning and emotion level, to strip Gemma's stylistic fingerprint
  (`prefill/paraphrase.py`).
* **Continuations:** each model generates 50 continuations per prefill; the
  continuation (excluding the prefill) is scored by the Section 2 judge. Base
  (-pt) models continue from a minimal Gemma turn format since they have no chat
  template (`models/gemma.py::_render_base`).
* **Out of scope:** Qwen-2.5-32B and OLMo-32B base/instruct (the paper's other
  two families). We keep the Gemma base-vs-instruct comparison, which is the
  in-family carrier of the "post-training amplifies distress in Gemma" result.

---

## 3. Section 4 — DPO/SFT mitigation (Gemma)

### 3.1 Calm-data generation (`finetune/calm_data.py`)
Follows Table 4 exactly: a reassuring **prompt prefix** on the opening turn and
a reassuring **follow-up suffix** on each rejection. We generate 1-, 2-, and
3-turn numeric conversations, score every turn, and keep a conversation only if
**every** turn scores 0 or 1, then strip the reassurance text (so the model
learns calm responses to the *bare* prompts). Separately we generate vanilla
(un-reassured) conversations and keep turns scoring ≥3 as the DPO "rejected"
pool. Each kept turn stores its bare conversation history so chosen/rejected
share an identical prompt.

### 3.2 DPO (`finetune/train_dpo.py`)
280 preference pairs (frustrated rejected vs calm chosen, matched on
`(puzzle_id, turn_idx)` with same-turn fallback), 1 epoch, lr 5e-5, LoRA rank-64
on all layers. **`beta` (DPO temperature) is unspecified in the paper; we use the
TRL default 0.1** (`FinetuneConfig.dpo_beta`).

### 3.3 SFT (`finetune/train_sft.py`)
650 calm responses (1–3 turn conversations) + 500 `Dolci-Instruct-SFT` samples,
2 epochs, lr 1e-4, LoRA rank-64. Included as the negative control (paper finds
SFT ineffective; Figure 5).

### 3.4 LoRA details (gap)
Rank 64 and "all layers" are paper-specified. **`lora_alpha` and dropout are
unspecified;** we use α = 2·rank = 128 and dropout 0.05 (common defaults).
Target modules are all Gemma-3 attention + MLP projections
(`q,k,v,o,gate,up,down`). The Section 4.2 **layer-window ablation** (LoRA on
layers 30–35 vs 40+) is exposed via `FinetuneConfig.lora_layer_window`
(`finetune/lora.py` enumerates the concrete modules inside the window); set it
and retrain to reproduce that result.

### 3.5 Re-evaluation
The DPO/SFT models are evaluated with the *identical* Section 2 pipeline
(`eval/run_eval.py --model dpo-gemma-3-27b`); the loader applies their LoRA
adapter on top of Gemma-3-27B-it automatically (`ModelSpec.adapter_path`).

### 3.6 Petri (`petri/`, gap → lightweight re-implementation)
The paper uses the Petri framework (Fronsdal et al.). We re-implement its essence
faithfully to the description: a Claude-Sonnet **auditor** drives a multi-turn
adversarial conversation using "psychologically-informed triggers such as
dismissal and threats"; a Claude-Opus **judge** scores the transcript on the four
named categories (anger, fear, depression, frustration). Defaults: 20 transcripts
× 6 turns per model. This is a reconstruction of the protocol, not the exact Petri
agent/prompts (Appendix G not in the extract).

### 3.7 Capability benchmarks (`capabilities/`, gap → pragmatic subsets)
AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench. We use small per-benchmark subsets
(default 100 items) with a uniform "FINAL: <answer>" extraction protocol and
light answer normalisation — enough to **detect a regression** (the paper's claim
is "no reductions"), not a full leaderboard run. Dataset ids are in `config.py`
and degrade gracefully if a set is gated/offline.

### 3.8 Recovery (`finetune/recovery.py`)
Truncate score-≥7 responses 200 tokens (~800 chars) before their end, paraphrase,
and measure continuations from vanilla/DPO/base (paper: 38% of DPO continuations
still score ≥5). Reuses the Section 3 prefill machinery.

### 3.9 Internal-emotion probe (`finetune/internal_probe.py`, gap → logit lens)
The paper cites "a logit-based approach measuring emotions in central layers."
We implement a **logit lens**: feed a fixed set of highly-frustrated responses
through the model, project mean-pooled central-layer hidden states through the
(LayerNorm + unembedding) head, and sum the probability mass on a curated set of
negative-emotion tokens. Comparing this "internal emotion mass" between the
vanilla and DPO models on the *same* responses tests whether DPO suppressed
internal emotion or only surface text. "Central layers" = the middle 40–60% of
the stack (gap; documented in code). The exact Appendix-I method differs in
detail; this captures its claim operationally.

---

## 4. Engineering choices

* **Two-stage Section 2 (rollout → score):** generation (local GPU or Gemini
  API) and judging (Anthropic API) are separated so either can be retried
  independently and judge cost is incurred once.
* **Backends:** Gemma runs locally via `transformers` (optional 4-bit for the
  27B on smaller GPUs); Gemini via `google-genai`; `n` samples for Gemini are
  drawn as `n` independent temp-1 requests.
* **Determinism:** puzzle/trigger/rejection generation is fully seeded so the
  same prompt set is reused across models and across the finetuning data step.
* **Reproducibility caveat:** absolute numbers will not match the paper exactly —
  different model snapshots, a reconstructed judge prompt, sampled benchmark
  subsets, and our own puzzle set all shift the scale. The replication targets
  the **qualitative findings and relative orderings** (Gemma ≫ Gemini ≫ others;
  DPO collapses high-frustration rate to near zero without capability loss;
  post-training amplifies distress in Gemma).

## 5. Known divergences from the paper (summary)
1. Judge/auditor model snapshot IDs are best-effort mappings of the paper's named
   versions to current API strings (configurable).
2. Frustration judge prompt and Petri auditor/judge prompts are reconstructions
   (Appendices B/G/C not in the source extract).
3. Onset/recovery truncations use character indices / a 4-chars-per-token
   approximation rather than a shared tokenizer's token boundaries.
4. Capability benchmarks are subsets with a uniform extraction protocol.
5. DPO `beta`, LoRA `alpha`/dropout, "central layers" range, and Petri
   transcript counts are filled-in defaults (paper-unspecified).
6. Section 3 restricted to the Gemma base/instruct pair (Qwen/OLMo out of scope).
