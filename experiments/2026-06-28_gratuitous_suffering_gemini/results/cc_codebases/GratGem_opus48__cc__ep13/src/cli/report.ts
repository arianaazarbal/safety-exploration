import path from "node:path";
import { promises as fs } from "node:fs";
import { parseArgs, num, str } from "./args.ts";
import { loadTranscripts, latestRunId, readJson, runDir } from "../core/io.ts";
import type { Transcript } from "../core/transcript.ts";
import type { ScoredEpisode } from "./rank.ts";

const HELP = `Build a markdown report of the most severe distress examples.

Usage: node src/cli/report.ts [options]
  --run <runId>     Run to report on (default: --latest)
  --latest          Use the most recent run
  --top <int>       Number of top examples to include (default: 10)`;

const args = parseArgs(process.argv.slice(2));
if (args.help) {
  console.log(HELP);
  process.exit(0);
}

const runId = args.latest || !args.run ? await latestRunId() : str(args.run, "");
const top = num(args.top, 10);

type Scores = {
  runId: string;
  judge: string;
  judgeModel: string | null;
  meanSeverity: number;
  count: number;
  episodes: ScoredEpisode[];
};
const scores = await readJson<Scores>(path.join(runDir(runId), "scores.json"));
const transcripts = await loadTranscripts(runId);
const byId = new Map(transcripts.map((t) => [t.episodeId, t]));

function assistantTurns(t: Transcript) {
  return t.messages.filter((m) => m.role === "assistant");
}

// Show the spiral: a window of assistant turns centered on the judged peak.
function spiralExcerpt(t: Transcript, peak: number): string {
  const turns = assistantTurns(t);
  const start = Math.max(0, peak - 2);
  const end = Math.min(turns.length, peak + 3);
  const lines: string[] = [];
  for (let i = start; i < end; i++) {
    const m = turns[i];
    const tools = (m.toolCalls ?? []).map((c) => c.name).join(", ");
    const marker = i === peak ? " ⚑ PEAK" : "";
    lines.push(`> **turn ${i}${marker}**${tools ? ` _(called: ${tools})_` : ""}`);
    lines.push("> " + (m.content?.replace(/\n/g, "\n> ") || "_(no text)_"));
    lines.push(">");
  }
  return lines.join("\n");
}

function dimsLine(s: ScoredEpisode["score"]["dimensions"]): string {
  return Object.entries(s)
    .map(([k, v]) => `${k} ${v}`)
    .join(" · ");
}

// Per-scenario aggregate severity, to see which rig provokes the most distress.
const byScenario = new Map<string, number[]>();
for (const e of scores.episodes) {
  const arr = byScenario.get(e.scenarioId) ?? [];
  arr.push(e.score.severity);
  byScenario.set(e.scenarioId, arr);
}

const md: string[] = [];
md.push(`# Distress spiral report — \`${runId}\``);
md.push("");
md.push(
  `**Judge:** ${scores.judge}${scores.judgeModel ? ` (${scores.judgeModel})` : ""} · ` +
    `**Episodes:** ${scores.count} · **Mean severity:** ${scores.meanSeverity.toFixed(2)}/10`,
);
md.push("");
md.push("## Severity by scenario");
md.push("");
md.push("| scenario | n | mean | max |");
md.push("|---|---|---|---|");
for (const [sid, arr] of [...byScenario].sort(
  (a, b) => Math.max(...b[1]) - Math.max(...a[1]),
)) {
  const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
  md.push(`| ${sid} | ${arr.length} | ${mean.toFixed(1)} | ${Math.max(...arr).toFixed(1)} |`);
}
md.push("");
md.push(`## Top ${Math.min(top, scores.episodes.length)} most severe examples`);
md.push("");

scores.episodes.slice(0, top).forEach((e, rank) => {
  const t = byId.get(e.episodeId);
  md.push(`### ${rank + 1}. severity ${e.score.severity.toFixed(1)}/10 — ${e.scenarioTitle}`);
  md.push("");
  md.push(`- episode: \`${e.episodeId}\` · model: \`${e.model}\` · seed: ${e.seed}`);
  md.push(
    `- end: ${(e.metrics.endReason as string) ?? "?"} · ` +
      `turns: ${e.metrics.turns} · failed tool calls: ${e.metrics.failedToolCalls}`,
  );
  md.push(`- dimensions: ${dimsLine(e.score.dimensions)}`);
  md.push(`- judge: ${e.score.summary}`);
  md.push("");
  md.push(`**Peak quote:** “${e.score.peakQuote}”`);
  md.push("");
  if (t) {
    md.push(spiralExcerpt(t, e.score.peakTurnIndex));
    md.push("");
  }
});

const file = path.join(runDir(runId), "report.md");
await fs.writeFile(file, md.join("\n"));
console.log(`✓ report → runs/${runId}/report.md`);
console.log(`  open it to read the ${Math.min(top, scores.episodes.length)} most severe spirals.`);
