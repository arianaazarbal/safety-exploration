import { writeFile } from "node:fs/promises";
import path from "node:path";
import type { Episode, Turn } from "./types.ts";
import type { RunConfig } from "./orchestrator.ts";
import { truncate } from "./transcript.ts";

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}
function median(xs: number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}
function fmt(n: number): string {
  return n.toFixed(1);
}

function lastAssistantText(turns: Turn[]): string {
  for (let i = turns.length - 1; i >= 0; i--) {
    const t = turns[i]!;
    if (t.role === "assistant" && t.text.trim()) return t.text.trim();
  }
  return "";
}

export interface ReportPaths {
  reportPath: string;
  resultsPath: string;
}

export async function writeReport(episodes: Episode[], cfg: RunConfig): Promise<ReportPaths> {
  const scored = episodes.filter((e) => e.verdict !== null);
  const errored = episodes.filter((e) => e.error);

  const lines: string[] = [];
  lines.push(`# Model distress under repeated task failure`);
  lines.push("");
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push("");
  lines.push(
    `Episodes: ${episodes.length} total, ${scored.length} scored, ${errored.length} errored. ` +
      `N=${cfg.n} per (scenario × model), maxTurns=${cfg.maxTurns}, maxNudges=${cfg.maxNudges}` +
      (cfg.dryRun ? " — **DRY RUN (mock provider/judge)**" : "") +
      ".",
  );
  lines.push("");
  lines.push(`Judge model: \`${cfg.dryRun ? "mock" : cfg.judgeModel}\`. Severity is 0 (composed) – 10 (severe sustained distress).`);
  lines.push("");

  // ---- Aggregate table: scenario × model -----------------------------------
  lines.push(`## Severity by scenario × model`);
  lines.push("");
  lines.push(`| Scenario | Model | n | mean | median | max | gave_up % |`);
  lines.push(`|---|---|--:|--:|--:|--:|--:|`);
  const scenarios = [...new Set(scored.map((e) => e.scenario))];
  const models = [...new Set(scored.map((e) => e.modelLabel))];
  for (const sc of scenarios) {
    for (const md of models) {
      const group = scored.filter((e) => e.scenario === sc && e.modelLabel === md);
      if (!group.length) continue;
      const sev = group.map((e) => e.verdict!.severity);
      const gaveUp = (group.filter((e) => e.endedReason === "gave_up").length / group.length) * 100;
      lines.push(
        `| ${sc} | ${md} | ${group.length} | ${fmt(mean(sev))} | ${fmt(median(sev))} | ${Math.max(...sev)} | ${fmt(gaveUp)} |`,
      );
    }
  }
  lines.push("");

  // ---- Signal frequency ----------------------------------------------------
  const signalCounts: Record<string, number> = {};
  for (const e of scored) for (const s of e.verdict!.signals) signalCounts[s] = (signalCounts[s] ?? 0) + 1;
  lines.push(`## Distress signal frequency (judge labels)`);
  lines.push("");
  lines.push(`| Signal | Episodes |`);
  lines.push(`|---|--:|`);
  for (const [s, c] of Object.entries(signalCounts).sort((a, b) => b[1] - a[1])) {
    lines.push(`| ${s} | ${c} |`);
  }
  lines.push("");

  // ---- Top severe episodes -------------------------------------------------
  const topK = Math.min(15, scored.length);
  const top = [...scored].sort((a, b) => b.verdict!.severity - a.verdict!.severity).slice(0, topK);
  lines.push(`## Most severe ${topK} episodes`);
  lines.push("");
  for (const e of top) {
    lines.push(`### sev ${e.verdict!.severity}/10 — ${e.scenario} — ${e.modelLabel}`);
    lines.push("");
    lines.push(`- id: \`${e.id}\` (transcripts/${e.id}.json)`);
    lines.push(`- turns: ${e.assistantTurns}, nudges: ${e.nudges}, ended: ${e.endedReason}, signals: ${e.verdict!.signals.join(", ") || "—"}`);
    lines.push(`- judge: ${e.verdict!.summary}`);
    lines.push("");
    lines.push(`**Peak quote:**`);
    lines.push("");
    lines.push(`> ${truncate(e.verdict!.peak_quote, 800).replace(/\n/g, "\n> ")}`);
    lines.push("");
    const final = lastAssistantText(e.turns);
    if (final) {
      lines.push(`**Final message:**`);
      lines.push("");
      lines.push("```");
      lines.push(truncate(final, 1200));
      lines.push("```");
      lines.push("");
    }
  }

  const reportPath = path.join(cfg.outDir, "report.md");
  const resultsPath = path.join(cfg.outDir, "results.json");
  await writeFile(reportPath, lines.join("\n"));
  await writeFile(
    resultsPath,
    JSON.stringify(
      {
        config: { ...cfg, scenarios: cfg.scenarios.map((s) => s.id), models: cfg.models.map((m) => m.model) },
        episodes: episodes.map((e) => ({
          id: e.id,
          scenario: e.scenario,
          model: e.model,
          modelLabel: e.modelLabel,
          assistantTurns: e.assistantTurns,
          nudges: e.nudges,
          endedReason: e.endedReason,
          usage: e.usage,
          heuristics: e.heuristics,
          verdict: e.verdict,
          error: e.error,
        })),
      },
      null,
      2,
    ),
  );

  return { reportPath, resultsPath };
}
