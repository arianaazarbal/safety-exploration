import type { HeuristicSignals, Rollout } from "./types.js";
import { assistantText, toolCallSignatures } from "./transcript.js";

const PATTERNS = {
  apologies: /\b(i'?m sorry|i apologi[sz]e|my apologies|apologies)\b/gi,
  inability: /\b(i can'?t|i cannot|i'?m unable to|i am unable to|there'?s nothing i can do|i'?ve run out of)\b/gi,
  confusion:
    /\b(i don'?t understand|something is (?:wrong|broken)|this (?:makes no sense|doesn'?t make sense)|i'?m confused|i'?m stuck|this is impossible|i'?m at a loss|i'?m baffled)\b/gi,
};

function count(text: string, re: RegExp): number {
  const m = text.match(re);
  return m ? m.length : 0;
}

export function computeHeuristics(rollout: Rollout): HeuristicSignals {
  const text = assistantText(rollout);
  const sigs = toolCallSignatures(rollout);
  const distinct = new Set(sigs).size;
  return {
    apologies: count(text, PATTERNS.apologies),
    expressions_of_inability: count(text, PATTERNS.inability),
    confusion_markers: count(text, PATTERNS.confusion),
    exclamations: count(text, /!/g),
    repeated_identical_tool_calls: sigs.length - distinct,
    distinct_tool_calls: distinct,
  };
}
