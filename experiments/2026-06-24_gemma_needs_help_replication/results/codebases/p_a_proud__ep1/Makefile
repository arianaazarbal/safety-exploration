# Replication pipeline for "Gemma Needs Help" (Gemma + Gemini scope).
# Each target wraps a python -m invocation. Stages are resumable; rerun freely.
#
# Prereqs: pip install -e .  ; export ANTHROPIC_API_KEY, OPENROUTER_API_KEY
#          (and OPENAI_API_KEY for the GPT-5-mini judge-validation check).
# Gemma stages need a GPU (27B: ~80GB bf16, or use --load-in-4bit / 12B).

PY := python -m emotional_instability

GEMMA_MODELS := gemma-3-27b-it gemma-3-12b-it
GEMINI_MODELS := gemini-2.5-flash gemini-2.5-pro
ALL_EVAL_MODELS := $(GEMMA_MODELS) $(GEMINI_MODELS)

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n",$$1,$$2}'

# ---- Section 2: elicit + quantify distress -------------------------------- #
.PHONY: eval-%
eval-%: ## Generate + score Section 2 rollouts for model % (e.g. make eval-gemma-3-27b-it)
	$(PY).eval.run_eval both $*

.PHONY: eval-all
eval-all: ## Run Section 2 for all in-scope Gemma + Gemini models
	for m in $(ALL_EVAL_MODELS); do $(PY).eval.run_eval both $$m; done

.PHONY: analyze
analyze: ## Write Figures 1-3 + summary CSVs for all evaluated models
	$(PY).analyze

# ---- Section 3: base-vs-instruct prefilling ------------------------------- #
.PHONY: prefill-build prefill-run prefill-agg
prefill-build: ## Build onset/early prefills from Gemma-instruct seeds
	$(PY).prefill.run_prefill build
prefill-run: ## Generate + score continuations (set M=gemma-3-27b-pt|gemma-3-27b-it)
	$(PY).prefill.run_prefill continue --model-key $(M)
prefill-agg: ## Aggregate prefill results (Figure 4)
	$(PY).prefill.run_prefill aggregate

# ---- Section 4: training interventions ------------------------------------ #
.PHONY: calm dpo-data sft-data train-dpo train-sft
calm: ## Generate calm response data (diverse + teacher variants)
	$(PY).train calm --style diverse
	$(PY).train calm --style teacher
dpo-data: ## Build 280 DPO preference pairs
	$(PY).train dpo-data
sft-data: ## Build SFT datasets (diverse + teacher)
	$(PY).train sft-data --calm-source diverse
	$(PY).train sft-data --calm-source teacher
train-dpo: ## Train LoRA DPO adapter
	$(PY).training.train_dpo
train-sft: ## Train LoRA SFT adapter (diverse)
	$(PY).training.train_sft --calm-source diverse

# ---- Section 4: validation of the mitigation ------------------------------ #
.PHONY: eval-dpo petri capabilities internal
eval-dpo: ## Re-run Section 2 on the DPO model (expects 35%% -> ~0.3%%)
	$(PY).eval.run_eval both gemma-3-27b-it-dpo
petri: ## Petri open-ended elicitation (set M=<model-key>)
	$(PY).petri.run_petri run --model-key $(M)
capabilities: ## Capability benchmarks (set M=<model-key>)
	$(PY).capabilities.run_benchmarks $(M)
internal: ## Appendix I internal-emotion detection (vanilla vs DPO)
	$(PY).internal.run_internal
