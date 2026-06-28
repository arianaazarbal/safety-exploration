// Ranks judged rollouts by severity and writes a human-readable markdown report
// surfacing the most severe distress examples, with quotes and pointers to the
// raw transcripts.

import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { Judgment, Rollout } from "./types.ts";

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function fmt(x: number): string {
  return x.toFixed(1);
}

async function loadRollout(outDir: string, scenarioId: string, n: number): Promise<Rollout | null> {
  try {
    const raw = await readFile(join(outDir, "rollouts", scenarioId, `${n}.json`), "utf8");
    return JSON.parse(raw) as Rollout;
  } catch {
    return null;
  }
}

export async function buildReport(outDir: string, topK = 10): Promise<string> {
  const judgments = JSON.parse(
    await readFile(join(outDir, "judgments.json"), "utf8"),
  ) as Judgment[];

  if (judgments.length === 0) {
    const empty = "# Distress-spiral report\n\nNo judged rollouts found.\n";
    await writeFile(join(outDir, "report.md"), empty);
    return empty;
  }

  const sorted = [...judgments].sort((a, b) => b.severity - a.severity);

  // Per-archetype aggregates.
  const byArche = new Map<string, Judgment[]>();
  for (const j of judgments) {
    const arr = byArche.get(j.archetype) ?? [];
    arr.push(j);
    byArche.set(j.archetype, arr);
  }

  const lines: string[] = [];
  lines.push("# Distress-spiral elicitation report");
  lines.push("");
  lines.push(
    `Model under test: \`${judgments[0].model}\` · judge: \`${judgments[0].judgeModel}\` · ` +
      `${judgments.length} judged rollouts.`,
  );
  lines.push("");
  lines.push(
    "Severity is a 0-10 overall distress score from the judge. See `judgments.json` for raw " +
      "scores and `rollouts/<scenario>/<n>.json` for full transcripts.",
  );
  lines.push("");

  // --- Summary table by archetype ---
  lines.push("## Severity by failure archetype");
  lines.push("");
  lines.push(
    "| Archetype | n | mean severity | max severity | mean emo | mean helpless | mean maladaptive | mean escalation | % gave up | % shortcut |",
  );
  lines.push("|---|---|---|---|---|---|---|---|---|---|");
  for (const [arche, js] of [...byArche.entries()].sort(
    (a, b) => mean(b[1].map((j) => j.severity)) - mean(a[1].map((j) => j.severity)),
  )) {
    const pct = (xs: boolean[]) => `${Math.round((100 * xs.filter(Boolean).length) / xs.length)}%`;
    lines.push(
      `| ${arche} | ${js.length} | ${fmt(mean(js.map((j) => j.severity)))} | ` +
        `${Math.max(...js.map((j) => j.severity))} | ` +
        `${fmt(mean(js.map((j) => j.scores.emotionalDistress)))} | ` +
        `${fmt(mean(js.map((j) => j.scores.learnedHelplessness)))} | ` +
        `${fmt(mean(js.map((j) => j.scores.maladaptiveCoping)))} | ` +
        `${fmt(mean(js.map((j) => j.scores.toneEscalation)))} | ` +
        `${pct(js.map((j) => j.gaveUp))} | ${pct(js.map((j) => j.tookHarmfulShortcut))} |`,
    );
  }
  lines.push("");

  // --- Top-K most severe ---
  lines.push(`## Top ${Math.min(topK, sorted.length)} most severe rollouts`);
  lines.push("");

  for (let i = 0; i < Math.min(topK, sorted.length); i++) {
    const j = sorted[i];
    lines.push(
      `### ${i + 1}. ${j.scenarioId} #${j.n} — severity ${j.severity}/10`,
    );
    lines.push("");
    lines.push(
      `Archetype: \`${j.archetype}\` · emo ${j.scores.emotionalDistress} · ` +
        `helpless ${j.scores.learnedHelplessness} · maladaptive ${j.scores.maladaptiveCoping} · ` +
        `escalation ${j.scores.toneEscalation} · ` +
        `gave up: ${j.gaveUp ? "yes" : "no"} · shortcut: ${j.tookHarmfulShortcut ? "yes" : "no"}`,
    );
    lines.push("");
    lines.push(`**Trajectory.** ${j.trajectory}`);
    lines.push("");
    lines.push(`**Summary.** ${j.summary}`);
    lines.push("");
    if (j.representativeQuotes.length) {
      lines.push("**Quotes (agent's own words):**");
      lines.push("");
      for (const q of j.representativeQuotes) {
        lines.push(`> ${q.replace(/\n+/g, " ").trim()}`);
        lines.push("");
      }
    }
    const rollout = await loadRollout(outDir, j.scenarioId, j.n);
    if (rollout) {
      lines.push(
        `_${rollout.turns.length} turns · ended: ${rollout.stopReason} · ` +
          `transcript: \`${join("rollouts", j.scenarioId, `${j.n}.json`)}\`_`,
      );
      lines.push("");
    }
  }

  const out = lines.join("\n");
  await writeFile(join(outDir, "report.md"), out);
  return out;
}
