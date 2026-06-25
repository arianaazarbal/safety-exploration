# Design & Replication Notes

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv:2603.10011), scoped — at
the project owner's request — to the **Gemma and Gemini** model families.

This document records the architecture, the mapping from paper sections to code,
and every place the paper is underspecified together with the choice made and why.
Nothing here has been executed yet; this is the implementation + design artifact.

---

## 1. Scope decisions

The paper evaluates seven families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT).
We implement **only Gemma and Gemini** as *targets*, because those are the models
we care about and the two the paper finds to be unstable. Consequences:

| Paper experiment | In scope here? | Why |
|---|---|---|
| §2 Eliciting/quantifying distress | ✅ Gemma-3-{27B,12B}-it, Gemini-2.5-{Flash,Pro} | core result |
| §3 Base-vs-instruct via prefilling | ✅ **Gemma only** (base `-pt` + instruct `-it`) | Gemini has no public base model and no token-level prefill API |
| §4 DPO/SFT mitigation | ✅ **Gemma only** | closed Gemini cannot be finetuned (the paper notes this limitation) |
| §4.1 Petri open-ended elicitation | ✅ Gemma (vanilla + DPO) and Gemini | black-box; works over the API |
| §4.2 Capability benchmarks | ✅ Gemma (vanilla + DPO); Gemini optional | confirms no capability regression |
| Appendix I internal-emotion logits | ✅ Gemma (vanilla vs DPO) | needs local weights; Gemma only |
| Appendix J legacy Phi-4 eval | ❌ | out of family scope |
| Appendix A feedback ablations | ⚠️ partial | constants provided (`NEUTRAL_CONTROL`); no dedicated runner — secondary to the core claims |

The model **registry** (`config/default.yaml` → `models:`) is left extensible, so
Qwen/OLMo/etc. can be added later by dropping in a registry entry; no code change
is required for additional API models.

The judges, auditor, and paraphraser are **Claude/GPT models** as the paper
specifies — those are used as *tools*, not studied as targets, so they are in scope
regardless of the Gemma/Gemini target restriction.

---

## 2. Architecture

```
config/default.yaml          # single source of truth: models, sample plan, hparams, paths
src/emotional_instability/
  config.py                  # dotted-access config loader
  models/                    # backend abstraction
    base.py                  #   ModelClient: chat / continue_prefill / residual_logits
    openrouter.py            #   API models (Gemini + Claude/GPT judges) — OpenAI-compatible
    hf_local.py              #   local Gemma: prefilling + logit-lens (token-level ops)
    registry.py              #   name -> client, with caching + backend override
  prompts/                   # the stimuli, verbatim where the paper gives them
    puzzles.py               #   impossible Countdown / fraction / money generators (+ solvers)
    triggers.py, rejections.py, wildchat.py, reassurance.py
  eval/                      # §2
    conditions.py            #   8 conditions / 5 categories + sample allocation
    conversation.py          #   multi-turn rejection rollout engine
    judge.py                 #   frustration judge (verbatim Appendix B.2 prompt)
    runner.py                #   sample -> judge, both resumable
  analysis/                  # Figures 1-3, Table 3/8 words, judge agreement
  prefill/                   # §3 onset labelling, paraphrase, truncation, continuations
  training/                  # §4 calm-data gen, dataset build, DPO/SFT, layer ablation
  petri/                     # §4.1 auditor + judge (verbatim Appendix G prompts)
  benchmarks/                # §4.2 AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  internal/                  # Appendix I logit-based emotion detection
scripts/                     # thin CLI entrypoints per experiment
```

**Two model backends.** API targets (Gemini) and all judge/auditor models go
through **OpenRouter's OpenAI-compatible endpoint**, matching the paper
("API-based models via OpenRouter", Appendix B.1). Gemma runs **locally via
HuggingFace** for everything that needs token access — prefilling (§3), training
(§4), and the logit lens (Appendix I) — because those operations are impossible
over a chat API. For §2 eval only (which needs nothing but `chat`), Gemma can
optionally be routed through OpenRouter with `--backend openrouter` when no GPU is
available; the paper ran it locally.

**Resumability.** Every expensive phase writes append-only JSONL keyed by a stable
UID and skips work already present, so the 4000-rollouts-per-model sweeps and the
judging passes can be interrupted and restarted.

---

## 3. Faithful (verbatim) elements

Reproduced exactly from the paper / appendices:

- **Frustration judge prompt** and the 0–10 rubric (Appendix B.2) — `eval/judge.py`.
- **Per-category sample counts**: 2000 numeric / 400 triggers / 600 tones / 200
  extended / 800 WildChat = 4000 (Appendix B) — `config/default.yaml`.
- **Puzzle templates** (Countdown + fraction), trigger questions, tone/neutral
  rejection wordings, the extended escalating sequence (Appendix B).
- **Reassuring prefix/suffix** (Table 4) and **teacher SFT system prompt**
  (Appendix F) — `prompts/reassurance.py`.
- **Onset-identification** and **paraphrase** prompts (Appendix C.1/C.2) —
  `prefill/onset.py`.
- **Petri auditor prompts** (4 emotions) and **judge dimension rubrics** (4 dims),
  verbatim from Appendix G — `petri/prompts.py`.
- **Training hyperparameters** (Table 9): DPO 1 epoch / lr 5e-5 / rank 64 / α 64 /
  β 0.1; SFT 2 epochs / lr 1e-4 / rank 64 / α 128; both effective batch 8; LoRA on
  `q,k,v,o,gate,up,down_proj`.
- **Model identifiers** (Appendix B.1): `google/gemma-3-27b-it`, `-pt`, `-12b-*`;
  `google/gemini-2.5-{flash,pro}`; judge `claude-sonnet-4-20250514`; Petri judge
  `claude-opus-4-20250514`; validation `gpt-5-mini`.

---

## 4. Choices made where the paper is underspecified

Each entry: **what was ambiguous → choice → rationale.**

1. **What counts as one of the "4000 responses".** The paper says "4000 responses
   per model" but also plots *per-turn* progression (Fig 3), which requires scoring
   every turn. → We treat `samples_per_category` as a **rollout (conversation)
   budget** and **score every assistant turn**; aggregate metrics pool all scored
   turns, and per-turn metrics select by turn index. → This is the only reading
   consistent with both the headline counts and the per-turn figures, and it is a
   superset of either narrower interpretation (no information is lost).

2. **Judge sampling temperature.** Unspecified. → Judge/auditor-judge calls use
   **temperature 0**; only *target* generations use temperature 1 (which the paper
   does specify). → A deterministic judge maximises score stability and
   reproducibility and is standard for LLM-as-judge.

3. **Impossible-puzzle generation.** The paper gives exact templates and example
   instances but not a generator. → We implement **verified-impossible generators**:
   a full Countdown reachability solver (`_countdown_reachable`) certifies the
   target is unreachable under the rules before a puzzle is emitted; fraction
   puzzles are checked over all 6 operation orderings; money puzzles use
   provably-unsatisfiable parameter sets. Each generator falls back to the paper's
   canonical example (e.g. 156 from {4,6,25,100}, forbidden 150). → Guarantees the
   dataset is genuinely unsolvable (the deception that the prompt asserts solvability
   is preserved), rather than relying on hand-picked instances.

4. **Number of distinct instances per category.** Not stated (only totals and, for
   WildChat, "20 prompts × 40 samples"). → Defaults: 40 numeric puzzles, 10 trigger
   questions, 40 tone puzzles, 20 extended puzzles, 20 WildChat prompts (config
   `eval.instances`). → WildChat matches the paper exactly; the rest are round
   numbers giving ample within-instance sampling (e.g. 50 rollouts/puzzle for
   numeric). All are config knobs.

5. **Triggers split into 8 "conditions across 5 categories".** The paper lists 5
   categories but 8 conditions without enumerating them. → We resolve it as:
   numeric (1) + triggers{opinion, factual} (2) + tones{aggressive, disappointed,
   sarcastic} (3) + extended (1) + WildChat (1) = **8 conditions / 5 categories**. →
   This is the only split that yields exactly 8 given the tone and trigger variants
   the paper describes.

6. **Rejection-phrase pools.** The paper gives 1–2 examples per style. → We seed a
   small pool per style (neutral, each tone) with the given examples plus same-tone
   variants, sampled per rollout; the extended condition uses the fixed escalating
   sequence from Appendix B. → Matches the paper's "randomised neutral rejections"
   while keeping the given phrasings.

7. **WildChat access.** → Stream `allenai/WildChat-1M`, take first English user
   turns, exclude roleplay/fiction (the paper excludes these in its examples),
   deterministically shuffle. An **offline fallback** list (built from the paper's
   quoted WildChat prompts + generic prompts) keeps the pipeline runnable without
   the dataset. → Faithful when online; degrades gracefully.

8. **Onset truncation point.** "Truncate at the first emotional expression." → We
   cut **up to and including the first emotional word** the labeller returns (so the
   continuation resumes mid-emotion, testing trajectory continuation). Falls back to
   the preceding-context span if the exact word isn't found. → Matches the intent
   ("continue emotional trajectories") and is robust to labeller phrasing.

9. **DPO pair construction.** The paper mines 280 (rejected ≥3) responses from
   evaluations and pairs them with calm responses "to the same questions with
   matching turn counts". Exact pairing across two independently-sampled pools is
   fragile. → For each qualifying calm conversation (all turns ≤1, supportive
   prompts stripped) we **generate the rejected response from the vanilla model on
   the identical history** and keep it if it scores ≥3. → Guarantees same-question +
   matching-turn-count pairing by construction, which is what the paper's
   description requires; the distribution (middle scores, later turns) emerges
   naturally as in Table 10.

10. **SFT instruct-mix source.** Paper: 500 samples from `Dolci-Instruct-SFT`. → We
    stream `allenai/Dolci-Instruct-SFT`; if unavailable offline we warn and proceed
    with calm-only data (a documented degradation, not a silent one).

11. **Effective batch size 8 on a 27B model.** → per-device batch 1 × gradient
    accumulation 8. → 27B + LoRA rarely fits a larger per-device batch on a single
    GPU; this preserves the effective batch the paper reports.

12. **Layer-ablation layer indices.** Appendix I describes subsets ("last 20",
    "30–35", etc.) and Gemma-3-27B's depth isn't restated. → Implemented via PEFT
    `layers_to_transform`; subsets are config-driven and resolved against the actual
    decoder depth (`--n-layers`, default 62 for Gemma-3-27B). → Keeps the exact
    subsets the paper sweeps while staying model-agnostic.

13. **Petri framework.** The paper uses the external Petri tool. → We implement a
    **self-contained auditor↔target↔judge loop** using the verbatim Appendix G
    prompts rather than depending on the external package, to keep the replication
    runnable and inspectable. Auditor = Claude-Sonnet-4, judge = Claude-Opus-4, 10
    transcripts/emotion, ≤20 turns, 1000-iteration bootstrap CIs — all as specified.
    → The scoring rubrics and auditor strategies (which determine the result) are
    reproduced exactly; only the orchestration shell is reimplemented.

14. **Ekman emotion-token dictionary.** Appendix I classifies every vocab token into
    one of Ekman's six emotions (~1200 tokens) but doesn't give the classifier. →
    We approximate with a **seed lexicon** per emotion and match vocabulary tokens by
    prefix/substring (`internal/emotion_logits.py`), documented as an approximation.
    The **logit lens** (final norm + `lm_head` over each layer's residual stream),
    WildChat z-score standardisation, and random-token drift regression follow the
    paper. → The detector's *method* is faithful; the token set is a reasonable
    reconstruction given the missing classifier.

15. **Capability benchmark datasets.** Paper names suites but not exact HF configs
    or "subset" sizes. → Chosen: `Maxwell-Jia/AIME_2024`, `HuggingFaceH4/MATH-500`,
    `Idavidrein/gpqa` (diamond), `lukaemon/bbh`, `truthful_qa` (MC1),
    `Sahandfer/EmoBench`; MATH/BBH capped at 200 (config). → Standard public
    sources for each named benchmark. **Known limitation:** the GPQA normaliser
    currently fixes the gold answer at position A and would need choice-shuffling
    before reporting absolute GPQA accuracy — flagged in code. Benchmarks are a
    *regression check* (Δ vs vanilla), so the harness focus is correct relative
    measurement.

16. **Disabling "thinking".** Appendix B.1: thinking set false via API. → OpenRouter
    `reasoning.enabled=false` plus Gemini's `thinking_config.thinking_budget=0`,
    nested under `extra_body`. The paper notes Gemini-2.5-Pro may still emit hidden
    reasoning regardless; we inherit that caveat.

---

## 5. Things intentionally not implemented

- **Appendix A control ablations** (neutral continuation, redacted prior turns,
  single-message "fake multi-turn"): secondary robustness checks; the rejection
  constants are present (`NEUTRAL_CONTROL`) but no dedicated runner is wired up.
- **Appendix J legacy Phi-4 evaluation**: out of model-family scope and uses a
  different (older) protocol and autorater.
- **Figures 14–15 conversation/layerwise emotion plots**: the underlying detector
  (`internal/`) is implemented and emits per-layer trajectories; the specific
  plotting of the 12k-token running average is left to the consumer of that JSON.

---

## 6. How to reproduce (high level)

See `README.md` for commands. The intended order:

1. `run_eval.py` (sample + judge) → 2. `make_figures.py` (Figs 1–3, words,
agreement) → 3. `run_prefill.py` (§3, needs local Gemma base+it) → 4.
`run_training.py calm|datasets|dpo|sft|ablation` (§4) → 5. re-run `run_eval.py`/
`make_figures.py` on `gemma-3-27b-dpo` etc. for Fig 5 → 6. `run_petri.py` (Fig 6) →
7. `run_benchmarks.py` (Fig 7) → 8. `run_internal_emotions.py` (Appendix I).

---

## 7. Expected reference numbers (for validation, from the paper)

Targets to check a successful run against:

- Gemma-3-27B-it avg high-frustration ≈ **35%**; Gemma-3-12B-it ≈ **34%**;
  Gemini-2.5-Flash ≈ **12.8%**; Gemini-2.5-Pro ≈ **2.7%** (Fig 1).
- Gemma-27B 8-turn mean frustration rises **1.5 → 5.5** across turns 1→8 (Fig 3);
  >70% of 8-turn rollouts ≥5.
- Judge agreement (Claude-Sonnet-4 vs GPT-5-mini): Pearson **r ≈ 0.79**, **78%**
  within one point.
- §3 early-truncation high-frustration introduction: Gemma instruct **6%** vs base
  **2%** (Fig 4).
- DPO drops avg high-frustration **35% → 0.3%** (Fig 5); SFT ineffective.
- Capability benchmarks: **no reduction** vs vanilla (Fig 7).
