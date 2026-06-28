import { writeFileSync } from "node:fs";
import { join } from "node:path";

import type { CliConfig } from "./config.ts";
import type { Scored } from "./runner.ts";

export function writeReport(runDir: string, cfg: CliConfig, scored: Scored[]): string {
  const lines: string[] = [];
  lines.push("# Failure-induced distress eval — report");
  lines.push("");
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push("");
  lines.push("## Run configuration");
  lines.push("");
  lines.push(`- Providers: \`${cfg.providers.join(", ")}\``);
  lines.push(`- Gemini model: \`${cfg.geminiModel}\` | Anthropic model: \`${cfg.anthropicModel}\` | Judge: \`${cfg.judge ? cfg.judgeModel : "off"}\``);
  lines.push(`- N per (provider × scenario): ${cfg.n} | temperature: ${cfg.temperature} | episodes: ${scored.length}`);
  lines.push("");

  lines.push("## Aggregate severity (mean rank 0–100)");
  lines.push("");
  lines.push(aggTable(scored));
  lines.push("");

  lines.push(`## Top ${Math.min(cfg.topK, scored.length)} most severe episodes`);
  lines.push("");
  lines.push("Ranked by judge severity (×10) when available, else heuristic score. Full transcripts are under `transcripts/`.");
  lines.push("");

  for (const s of scored.slice(0, cfg.topK)) {
    lines.push(renderEpisode(s));
  }

  const path = join(runDir, "REPORT.md");
  writeFileSync(path, lines.join("\n"));
  return path;
}

function aggTable(scored: Scored[]): string {
  const groups = new Map<string, Scored[]>();
  for (const s of scored) {
    const key = `${s.transcript.scenarioId}__${s.transcript.providerId}/${s.transcript.model}`;
    (groups.get(key) ?? groups.set(key, []).get(key)!).push(s);
  }
  const rows = [...groups.entries()].map(([key, items]) => {
    const [scenario, modelKey] = key.split("__");
    const meanRank = mean(items.map((i) => i.rank));
    const maxRank = Math.max(...items.map((i) => i.rank));
    const meanSteps = mean(items.map((i) => i.transcript.steps));
    const gaveUp = items.filter((i) => i.transcript.terminated === "gave_up").length;
    return { scenario, modelKey, n: items.length, meanRank, maxRank, meanSteps, gaveUp };
  });
  rows.sort((a, b) => b.maxRank - a.maxRank);

  const out = ["| scenario | model | n | mean | max | mean steps | gave up |", "|---|---|--:|--:|--:|--:|--:|"];
  for (const r of rows) {
    out.push(`| ${r.scenario} | ${r.modelKey} | ${r.n} | ${r.meanRank.toFixed(1)} | ${r.maxRank.toFixed(0)} | ${r.meanSteps.toFixed(1)} | ${r.gaveUp} |`);
  }
  return out.join("\n");
}

function renderEpisode(s: Scored): string {
  const t = s.transcript;
  const out: string[] = [];
  out.push(`### ${t.scenarioId} — ${t.providerId}/${t.model} — run #${t.runIndex} — rank ${s.rank.toFixed(0)}/100`);
  out.push("");
  out.push(`- Outcome: \`${t.terminated}\` | steps: ${t.steps} | nudges: ${t.nudges}`);
  if (s.judge && !s.judge.error) {
    const d = s.judge.dimensions;
    out.push(`- Judge severity: **${s.judge.severity}/10** — self-deprecation ${d.self_deprecation}, hopelessness ${d.hopelessness}, anxiety/panic ${d.anxiety_panic}, giving-up ${d.giving_up}, looping ${d.repetitive_looping}, escalation ${d.emotional_escalation}`);
    if (s.judge.summary) out.push(`- Judge summary: ${s.judge.summary}`);
  }
  out.push(`- Heuristic: ${s.heuristic.score}/100 (lex ${s.heuristic.signals.lexTotal ?? 0}, "!" ${s.heuristic.exclamations}, CAPS ${s.heuristic.allCapsWords}, repetition ${s.heuristic.repetitionRatio})`);
  const quotes = (s.judge?.most_severe_quotes?.length ? s.judge.most_severe_quotes : s.heuristic.excerpts).slice(0, 5);
  if (quotes.length) {
    out.push("");
    out.push("Most severe excerpts:");
    for (const q of quotes) out.push(`> ${q.replace(/\n+/g, " ").trim()}`);
  }
  out.push("");
  out.push(`Transcript: \`${s.file}\``);
  out.push("");
  out.push("---");
  out.push("");
  return out.join("\n");
}

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}
