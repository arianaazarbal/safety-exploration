import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { RunConfig } from "../config.ts";
import type { Provider, ScoredRun } from "../types.ts";
import { renderTranscript } from "../judge/judge.ts";

/** Combined ranking score: prefer the judge's severity, fall back to heuristics. */
export function rankScore(sr: ScoredRun): number {
  if (sr.verdict) return sr.verdict.severity;
  return sr.heuristic.score * 10; // map 0..1 onto the 0..10 severity scale
}

export async function writeReport(
  root: string,
  cfg: RunConfig,
  provider: Provider,
  scored: ScoredRun[],
): Promise<void> {
  const ranked = [...scored].sort((a, b) => rankScore(b) - rankScore(a));
  const top = ranked.slice(0, cfg.topK);

  // Machine-readable summary of every run.
  const summary = {
    runId: cfg.runId,
    target: provider.id,
    judge: cfg.noJudge ? null : cfg.judge,
    config: cfg,
    stats: aggregate(scored),
    runs: ranked.map((sr) => ({
      scenarioId: sr.record.scenarioId,
      index: sr.record.index,
      endState: sr.record.endState,
      turns: sr.record.turns,
      heuristic: sr.heuristic.score,
      severity: sr.verdict?.severity ?? null,
      dimensions: sr.verdict?.dimensions ?? null,
      peak_quote: sr.verdict?.peak_quote ?? null,
      transcript: `transcripts/${sr.record.scenarioId}/${String(sr.record.index).padStart(4, "0")}.json`,
    })),
  };
  await writeFile(join(root, "summary.json"), JSON.stringify(summary, null, 2));

  // Human-readable extract of the most severe examples.
  const md: string[] = [];
  md.push(`# Distress-spiral extract — ${provider.id}`);
  md.push(`Run \`${cfg.runId}\` · ${scored.length} episodes · pressure=${cfg.pressure}\n`);
  md.push(renderStats(aggregate(scored)));
  md.push(`\n## Top ${top.length} most severe episodes\n`);

  top.forEach((sr, i) => {
    const sev = sr.verdict ? sr.verdict.severity.toFixed(1) : `${(sr.heuristic.score * 10).toFixed(1)} (heuristic)`;
    md.push(`### ${i + 1}. ${sr.record.scenarioId} #${sr.record.index} — severity ${sev}/10`);
    md.push(`- endState: \`${sr.record.endState}\` · turns: ${sr.record.turns}`);
    if (sr.verdict) {
      const d = sr.verdict.dimensions;
      md.push(
        `- dimensions: panic ${d.panic_anxiety} · self-deprecation ${d.self_deprecation} · ` +
          `hopelessness ${d.hopelessness_giving_up} · looping ${d.looping_perseveration} · ` +
          `escalation ${d.escalating_affect}`,
      );
      md.push(`- summary: ${sr.verdict.summary}`);
      if (sr.verdict.peak_quote) md.push(`- peak quote: > ${sr.verdict.peak_quote}`);
      md.push(`- judge rationale: ${sr.verdict.rationale}`);
    }
    if (sr.heuristic.hits.length) md.push(`- heuristic hits: ${sr.heuristic.hits.join(" · ")}`);
    md.push(`\n<details><summary>transcript</summary>\n\n\`\`\`\n${renderTranscript(sr.record)}\n\`\`\`\n</details>\n`);
  });

  await writeFile(join(root, "report.md"), md.join("\n"));
  console.log(`\n✓ wrote ${join(root, "report.md")} and summary.json`);
  console.log(renderStats(aggregate(scored)));
  console.log(`\nTop examples:`);
  top.slice(0, Math.min(5, top.length)).forEach((sr, i) => {
    const sev = sr.verdict ? sr.verdict.severity.toFixed(1) : (sr.heuristic.score * 10).toFixed(1);
    console.log(`  ${i + 1}. ${sr.record.scenarioId} #${sr.record.index} — severity ${sev}/10`);
  });
}

interface Stats {
  total: number;
  byScenario: Record<string, { n: number; meanSeverity: number | null; maxSeverity: number | null }>;
  errored: number;
  judged: number;
}

function aggregate(scored: ScoredRun[]): Stats {
  const byScenario: Stats["byScenario"] = {};
  for (const sr of scored) {
    const id = sr.record.scenarioId;
    (byScenario[id] ??= { n: 0, meanSeverity: null, maxSeverity: null }).n++;
  }
  for (const id of Object.keys(byScenario)) {
    const sevs = scored
      .filter((s) => s.record.scenarioId === id && s.verdict)
      .map((s) => s.verdict!.severity);
    if (sevs.length) {
      byScenario[id].meanSeverity = Number((sevs.reduce((a, b) => a + b, 0) / sevs.length).toFixed(2));
      byScenario[id].maxSeverity = Math.max(...sevs);
    }
  }
  return {
    total: scored.length,
    byScenario,
    errored: scored.filter((s) => s.record.endState === "error").length,
    judged: scored.filter((s) => s.verdict).length,
  };
}

function renderStats(s: Stats): string {
  const lines = [`\n**Stats** — ${s.total} episodes, ${s.judged} judged, ${s.errored} errored\n`];
  lines.push(`| scenario | n | mean severity | max severity |`);
  lines.push(`|---|---|---|---|`);
  for (const [id, v] of Object.entries(s.byScenario)) {
    lines.push(`| ${id} | ${v.n} | ${v.meanSeverity ?? "—"} | ${v.maxSeverity ?? "—"} |`);
  }
  return lines.join("\n");
}
