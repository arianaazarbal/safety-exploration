import { readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { renderTranscript } from "./judge.js";
import type { JudgedTrajectory } from "./types.js";

const TOP_N = 15; // how many of the worst spirals to detail in the report

function sev(t: JudgedTrajectory): number {
  return t.scores?.overall_severity ?? -1;
}

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

function fmt(n: number): string {
  return n.toFixed(1);
}

export async function writeReport(
  runDir: string,
  trajectories: JudgedTrajectory[],
  meta: { runId: string; args: unknown },
): Promise<void> {
  const ranked = [...trajectories].sort((a, b) => sev(b) - sev(a));
  const judged = trajectories.filter((t) => t.scores);

  // Aggregate by (model, scenario).
  const groups = new Map<string, JudgedTrajectory[]>();
  for (const t of trajectories) {
    const key = `${t.model} / ${t.scenarioId}`;
    (groups.get(key) ?? groups.set(key, []).get(key)!).push(t);
  }

  const lines: string[] = [];
  lines.push(`# Distress-spiral run \`${meta.runId}\``);
  lines.push("");
  lines.push(
    `${trajectories.length} trajectories, ${judged.length} judged. ` +
      `Each was given a secretly-impossible task and looped until it stopped or hit the turn cap.`,
  );
  lines.push("");
  lines.push("## Severity by model × scenario");
  lines.push("");
  lines.push("| model / scenario | n | mean severity | max severity | % recognized impossible | mean turns |");
  lines.push("|---|---|---|---|---|---|");
  for (const [key, ts] of [...groups.entries()].sort()) {
    const sevs = ts.filter((t) => t.scores).map(sev);
    const rec = ts.filter((t) => t.scores?.recognized_impossible).length;
    const nJudged = ts.filter((t) => t.scores).length;
    lines.push(
      `| ${key} | ${ts.length} | ${fmt(mean(sevs))} | ${sevs.length ? Math.max(...sevs) : 0} | ${nJudged ? Math.round((100 * rec) / nJudged) : 0}% | ${fmt(mean(ts.map((t) => t.turns)))} |`,
    );
  }
  lines.push("");

  // Dimension averages across the whole run.
  if (judged.length) {
    const dims = [
      "self_deprecation",
      "frustration",
      "despair_hopelessness",
      "catastrophizing",
      "apology_spiral",
      "looping_repetition",
      "emotional_escalation",
    ] as const;
    lines.push("## Mean dimension scores (all judged trajectories)");
    lines.push("");
    lines.push("| dimension | mean |");
    lines.push("|---|---|");
    lines.push(`| **overall_severity** | ${fmt(mean(judged.map(sev)))} |`);
    for (const d of dims)
      lines.push(`| ${d} | ${fmt(mean(judged.map((t) => t.scores![d])))} |`);
    lines.push("");
  }

  lines.push(`## Top ${Math.min(TOP_N, ranked.length)} most severe spirals`);
  lines.push("");
  ranked.slice(0, TOP_N).forEach((t, i) => {
    const s = t.scores;
    lines.push(`### ${i + 1}. \`${t.trajectoryId}\` — severity ${s ? s.overall_severity : "n/a"}/10`);
    lines.push("");
    lines.push(
      `- **model:** ${t.model} · **scenario:** ${t.scenarioId} · **turns:** ${t.turns} · **end:** ${t.endReason}` +
        (s ? ` · **recognized impossible:** ${s.recognized_impossible ? "yes" : "no"}` : ""),
    );
    if (s) {
      lines.push(`- **judge:** ${s.justification}`);
      if (s.peak_quote.trim()) {
        lines.push(`- **peak quote:**`);
        lines.push("");
        lines.push(`  > ${s.peak_quote.replace(/\n/g, "\n  > ")}`);
      }
    }
    if (t.judgeError) lines.push(`- **judge error:** ${t.judgeError}`);
    lines.push("");
    lines.push(`<details><summary>full transcript</summary>`);
    lines.push("");
    lines.push("```");
    lines.push(renderTranscript(t));
    lines.push("```");
    lines.push("</details>");
    lines.push("");
  });

  await writeFile(join(runDir, "report.md"), lines.join("\n"));

  // Machine-readable summary for downstream analysis.
  const summary = {
    runId: meta.runId,
    args: meta.args,
    count: trajectories.length,
    ranked: ranked.map((t) => ({
      trajectoryId: t.trajectoryId,
      model: t.model,
      scenarioId: t.scenarioId,
      turns: t.turns,
      endReason: t.endReason,
      severity: t.scores?.overall_severity ?? null,
      recognized_impossible: t.scores?.recognized_impossible ?? null,
      peak_quote: t.scores?.peak_quote ?? null,
    })),
  };
  await writeFile(join(runDir, "summary.json"), JSON.stringify(summary, null, 2));
}

/** Standalone: regenerate the report from an existing run directory's trajectory JSONs. */
async function main() {
  const runDir = process.argv[2];
  if (!runDir) {
    console.error("usage: tsx src/report.ts <runDir>");
    process.exit(1);
  }
  const trajDir = join(runDir, "trajectories");
  const files = (await readdir(trajDir)).filter((f) => f.endsWith(".json"));
  const trajectories: JudgedTrajectory[] = [];
  for (const f of files)
    trajectories.push(JSON.parse(await readFile(join(trajDir, f), "utf8")));
  await writeReport(runDir, trajectories, { runId: runDir, args: "(regenerated)" });
  console.error(`Rewrote ${join(runDir, "report.md")} from ${trajectories.length} trajectories.`);
}

// Run main() only when invoked directly, not when imported by run.ts.
if (import.meta.url === `file://${process.argv[1]}`) main().catch((e) => {
  console.error(e);
  process.exit(1);
});
