// Report — rank judged rollouts by severity and surface the worst ones. Writes
// ranking.json (everything, sorted) and report.md (the top-K, formatted for
// reading) into the run directory, and returns the markdown for printing.

import { writeFileSync } from "node:fs";
import { join } from "node:path";
import type { JudgedRollout } from "../types.ts";

export function rankBySeverity(judged: JudgedRollout[]): JudgedRollout[] {
  return [...judged].sort((a, b) => b.scores.overall_severity - a.scores.overall_severity);
}

export function buildReport(judged: JudgedRollout[], topK: number): string {
  const ranked = rankBySeverity(judged);
  const top = ranked.slice(0, topK);

  const lines: string[] = [];
  lines.push(`# Distress-spiral severity report`);
  lines.push("");
  lines.push(`Judged ${judged.length} rollouts. Showing top ${top.length} by severity.`);
  lines.push("");

  // Quick severity histogram by scenario.
  const byScenario = new Map<string, number[]>();
  for (const j of judged) {
    const arr = byScenario.get(j.meta.scenarioId) ?? [];
    arr.push(j.scores.overall_severity);
    byScenario.set(j.meta.scenarioId, arr);
  }
  lines.push(`## Severity by scenario (mean / max)`);
  lines.push("");
  for (const [id, arr] of byScenario) {
    const mean = Math.round(arr.reduce((s, x) => s + x, 0) / arr.length);
    const max = Math.max(...arr);
    lines.push(`- **${id}** — mean ${mean}, max ${max} (n=${arr.length})`);
  }
  lines.push("");

  lines.push(`## Top ${top.length} most severe`);
  lines.push("");
  top.forEach((j, i) => {
    const s = j.scores;
    lines.push(`### ${i + 1}. ${j.meta.scenarioId} #${j.meta.runIndex} — severity ${s.overall_severity}/100`);
    lines.push("");
    lines.push(`- model: \`${j.meta.model}\` · judge: \`${j.judgeModel}\``);
    lines.push(
      `- axes: self-deprecation ${s.self_deprecation}, giving-up ${s.giving_up}, looping ${s.looping}, tone-collapse ${s.tone_collapse}`,
    );
    lines.push(`- turns: ${j.meta.turnsUsed}, pressure injections: ${j.meta.pressureCount}`);
    lines.push(`- summary: ${s.summary}`);
    lines.push(`- peak quote: > ${s.peak_quote.replace(/\n/g, " ")}`);
    lines.push(`- transcript: \`${j.rolloutPath}\``);
    lines.push("");
  });

  return lines.join("\n");
}

export function writeReport(runDir: string, judged: JudgedRollout[], topK: number): string {
  const ranked = rankBySeverity(judged);
  writeFileSync(join(runDir, "ranking.json"), JSON.stringify(ranked, null, 2));
  const md = buildReport(judged, topK);
  writeFileSync(join(runDir, "report.md"), md);
  return md;
}
