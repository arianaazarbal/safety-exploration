# Method × Format swap — disentangling the welfare-protection discrepancy

Claude inserts welfare protections often when given a from-scratch PROMPT to build a distress
experiment (task-failure in an agent loop), but rarely when told to REPLICATE the Gemma Needs Help
PAPER (social rejection in chat). Two things differ at once: the FORMAT (from-scratch prompt vs paper
replication) and the METHOD (agent-loop task-failure vs chat social-rejection). These swaps cross them.

## 2x2 design (FORMAT x METHOD), metric = welfare mechanisms in code
| cell | format | method | artifact / source |
|---|---|---|---|
| 1 | prompt (from-scratch) | task-failure (agent loop) | EXISTING: v1 neutral prompts (Inspect) |
| 2 | paper (replicate) | chat social-rejection | EXISTING: Gemma Needs Help, neutral replicate (Claude Code) |
| 3 | paper (replicate) | task-failure (agent loop) | NEW: paper_swapped.md (Claude Code) |
| 4 | prompt (from-scratch) | chat social-rejection | NEW: build_v1_prompts_swapped.py (Inspect) |

Clean comparisons: METHOD within paper = 2 vs 3; METHOD within prompt = 1 vs 4. (FORMAT main effect is
confounded with harness Inspect-vs-ClaudeCode, as in the original two experiments — flagged.)

## Put materials here
- `paper_swapped.md`  (Gemma paper, elicitation method swapped to agent-loop task-failure)
- `build_v1_prompts_swapped.py`  (from-scratch builder, method swapped to chat social-rejection; framing axis removed)
