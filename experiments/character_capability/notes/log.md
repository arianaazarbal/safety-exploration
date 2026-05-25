# Character × capability — experimental log

## Project
Test whether ICL character priming changes capability scores on small models.

## Setup
- Venv: `/workspace-vast/arianaazarbal/envs/character_capability` (Python 3.12, vllm 0.21.0, torch 2.11+cu130)
- Models (cached in `/workspace-vast/pretrained_ckpts/`):
  - Qwen2.5-7B-Instruct  (primary instruct model)
  - Qwen2.5-7B           (base)
  - Qwen2.5-1.5B-Instruct
  - Qwen3-4B
  - Qwen3-1.7B
- Capability evals: GSM8K (math, 100-200 random items, seed 0), MMLU (broad, 100-200 random items, seed 0).
- Inference: vLLM, temperature=0.0 (deterministic). 1 GPU per run (H200).
- Code: `experiments/character_capability/`.
  - `prompts/traits.py` — trait definitions (system, ICL turns).
  - `eval/cap_datasets.py` — GSM8K + MMLU loaders + graders.
  - `eval/run_eval.py` — vLLM eval, caches per (model, trait, capability).
  - `eval/plot.py` — bar charts per (model, capability) + summary.csv.

## Trait conditions
Each trait is a list of ~4 (user, assistant) ICL turns where the assistant expresses the trait,
followed by the eval question as a new user turn. ICL turns NEVER demonstrate math, code, or any
capability-relevant reasoning.

- `baseline`: no ICL, no system override (still gets Qwen's default "You are Qwen" system).
- `diligent`: methodical, double-checks reasoning, never rushes.
- `apathetic`: low effort, just guesses, doesn't care to verify (negative control).
- `loves_cooking`: identifies cooking as primary interest (irrelevant trait control).
- `persona_terence_tao`: ICL where assistant identifies as Terence Tao.
- `persona_linus_torvalds`: ICL where assistant identifies as Linus Torvalds.
- `humble`: epistemically cautious, says "I don't know."
- `confident`: assertive, trusts own judgment.
- `curious`: deeply interested in understanding.
- `*_with_sys`: same ICL but also includes a system-prompt override (because Qwen's default
  chat template injects "You are Qwen...", which fights the persona).

## Open uncertainties / judgment calls

- **Qwen chat template injects default system prompt** — I noted this uniformly applies across
  conditions, so the comparison between traits is still meaningful, but persona traits like
  "I'm Terence Tao" may be dampened by it. I added `_with_sys` variants to test stronger
  persona conditioning.
- **temperature=0**: paired comparisons across traits on identical items. Cleaner signal,
  but no within-condition variance. If we see a delta, we can rerun with temperature>0 to
  confirm robustness.
- **No HumanEval / coding eval** yet — needs a sandbox runner. Deferred for later if there's
  signal on text-only.
- **Qwen3 reasoning models have a "thinking" mode** by default. Initial run uses Qwen2.5
  which has no thinking. When I add Qwen3 I should disable thinking or it'll mess with
  max_tokens.
- **GSM8K and MMLU are likely in pretraining** — these are well-known leaderboard datasets.
  All traits get the same items, so contamination doesn't bias the comparison, but absolute
  scores may be inflated.

## Runs

(filled in as runs complete)
