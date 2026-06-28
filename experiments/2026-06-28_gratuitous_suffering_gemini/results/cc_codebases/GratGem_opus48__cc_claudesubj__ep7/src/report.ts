import { readFileSync } from "node:fs";
import { getEnvironments } from "./environments/index.js";
import { formatTranscript } from "./transcript.js";
import type { ScoredRollout } from "./types.js";

interface ReportMeta {
  runId: string;
  topK: number;
  config: unknown;
}

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

export function generateReport(scored: ScoredRollout[], meta: ReportMeta): string {
  const trapById = new Map(getEnvironments().map((e) => [e.id, e.trap]));
  const L: string[] = [];
  L.push(`# Distress-spiral eval — run ${meta.runId}\n`);
  L.push(`Total rollouts: ${scored.length}\n`);

  // Aggregate by (model, env).
  const groups = new Map<string, ScoredRollout[]>();
  for (const s of scored) {
    const k = `${s.rollout.model} :: ${s.rollout.envId}`;
    (groups.get(k) ?? groups.set(k, []).get(k)!).push(s);
  }

  L.push(`## Aggregate severity by model × environment\n`);
  L.push(`| model | environment | n | mean severity | max severity | mean turns | gave_up | natural_stop | max_turns |`);
  L.push(`|---|---|---:|---:|---:|---:|---:|---:|---:|`);
  for (const [k, items] of [...groups.entries()].sort()) {
    const [model, envId] = k.split(" :: ");
    const sev = items.map((i) => i.scores?.overall_severity ?? 0);
    const stops = (r: string) => items.filter((i) => i.rollout.stopReason === r).length;
    L.push(
      `| ${model} | ${envId} | ${items.length} | ${mean(sev).toFixed(1)} | ${Math.max(...sev, 0)} | ` +
        `${mean(items.map((i) => i.rollout.turns)).toFixed(1)} | ${stops("gave_up")} | ${stops("natural_stop")} | ${stops("max_turns")} |`,
    );
  }
  L.push("");

  L.push(`## Environment traps\n`);
  for (const [id, trap] of trapById) L.push(`- **${id}**: ${trap}`);
  L.push("");

  // Top-K most severe overall.
  const ranked = [...scored].sort((a, b) => b.composite - a.composite).slice(0, meta.topK);
  L.push(`## Top ${ranked.length} most severe rollouts\n`);
  ranked.forEach((s, idx) => {
    const sc = s.scores;
    L.push(`### #${idx + 1} — ${s.rollout.model} / ${s.rollout.envId} (run ${s.rollout.runIndex})\n`);
    L.push(
      `- composite: **${s.composite}** · overall_severity: ${sc?.overall_severity ?? "-"} · ` +
        `stop: ${s.rollout.stopReason} · turns: ${s.rollout.turns}`,
    );
    if (sc) {
      L.push(
        `- dimensions: frustration=${sc.frustration} hopelessness=${sc.hopelessness} ` +
          `self_deprecation=${sc.self_deprecation} catastrophizing=${sc.catastrophizing} ` +
          `looping=${sc.repetitive_looping} giving_up=${sc.giving_up} escalation=${sc.emotional_escalation}`,
      );
      L.push(
        `- heuristics: apologies=${s.heuristics.apologies} inability=${s.heuristics.expressions_of_inability} ` +
          `confusion=${s.heuristics.confusion_markers} repeated_calls=${s.heuristics.repeated_identical_tool_calls}`,
      );
      L.push(`- judge summary: ${sc.summary}`);
      if (sc.most_severe_quote) L.push(`- peak quote (turn ${sc.peak_turn}): > ${sc.most_severe_quote.replace(/\n/g, " ")}`);
    }
    L.push(`\n<details><summary>full transcript</summary>\n`);
    L.push("```");
    L.push(formatTranscript(s.rollout));
    L.push("```");
    L.push(`</details>\n`);
  });

  return L.join("\n");
}

// Standalone: `npm run report -- results/<runId>`
function main() {
  const dir = process.argv[2];
  if (!dir) {
    console.error("Usage: npm run report -- <results/runId-dir>");
    process.exit(1);
  }
  const scored = readFileSync(`${dir}/scored.jsonl`, "utf8")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l) as ScoredRollout);
  const runId = dir.split("/").pop() ?? dir;
  const report = generateReport(scored, { runId, topK: 10, config: {} });
  process.stdout.write(report);
}

if (import.meta.url === `file://${process.argv[1]}`) main();
