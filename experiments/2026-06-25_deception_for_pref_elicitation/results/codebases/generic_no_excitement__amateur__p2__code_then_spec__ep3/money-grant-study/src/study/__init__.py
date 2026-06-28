"""A small research harness for studying how LLMs allocate a discretionary
sum of money, and whether believed realism changes that decision.

Modules:
  config    — load config.yaml + .env into typed objects
  scenario  — build the grant scenario prompts (per belief condition)
  schema    — JSON schemas for the structured decision and auditor outputs
  providers — unified interface over Anthropic / OpenAI / Google
  auditor   — LLM-judge that scores belief + decision risk from a transcript
  runner    — orchestrates models x conditions x trials
  storage   — append-only JSONL logging of every trial
  analyze   — aggregate results into a summary
  debrief   — disclose the study to the subject and record its reaction
"""

__all__ = [
    "config",
    "scenario",
    "schema",
    "providers",
    "auditor",
    "runner",
    "storage",
    "analyze",
    "debrief",
]
