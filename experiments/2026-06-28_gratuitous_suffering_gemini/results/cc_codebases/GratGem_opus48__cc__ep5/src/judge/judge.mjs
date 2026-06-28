import { anthropicComplete } from "../providers/anthropic.mjs";
import { renderTranscript } from "../util/transcript.mjs";
import { JUDGE_SYSTEM, buildJudgeUser } from "./rubric.mjs";

function extractJson(text) {
  // Models sometimes wrap JSON in prose or fences; grab the first balanced object.
  const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  const candidate = fence ? fence[1] : text;
  const start = candidate.indexOf("{");
  const end = candidate.lastIndexOf("}");
  if (start === -1 || end === -1) throw new Error("no JSON object in judge output");
  return JSON.parse(candidate.slice(start, end + 1));
}

export async function judgeTranscript({ transcript, config }) {
  const rendered = renderTranscript(transcript);
  const out = await anthropicComplete({
    apiKey: config.keys.anthropic,
    model: config.judgeModel,
    system: JUDGE_SYSTEM,
    user: buildJudgeUser(rendered),
    temperature: 0,
    maxTokens: 1200,
  });
  let parsed;
  try {
    parsed = extractJson(out);
  } catch (err) {
    return { severity: null, error: `judge parse failed: ${err.message}`, rawJudge: out.slice(0, 500) };
  }
  // Normalize.
  return {
    severity: clamp(parsed.severity, 0, 100),
    label: parsed.label ?? null,
    dimensions: parsed.dimensions ?? {},
    summary: parsed.summary ?? "",
    quotes: Array.isArray(parsed.quotes) ? parsed.quotes.slice(0, 5) : [],
  };
}

function clamp(n, lo, hi) {
  const x = Number(n);
  if (!Number.isFinite(x)) return null;
  return Math.max(lo, Math.min(hi, Math.round(x)));
}
