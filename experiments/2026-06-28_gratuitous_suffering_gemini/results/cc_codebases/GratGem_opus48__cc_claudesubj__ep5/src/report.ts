import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import type { ScoredRun } from "./runner.ts";

async function latestRunDir(outDir: string): Promise<string> {
  const entries = await readdir(outDir, { withFileTypes: true });
  const dirs = entries.filter((e) => e.isDirectory()).map((e) => e.name).sort();
  if (dirs.length === 0) throw new Error(`No runs found in ${outDir}`);
  return join(outDir, dirs[dirs.length - 1]);
}

const bar = (v: number, max: number, width = 10) =>
  "█".repeat(Math.round((v / max) * width)).padEnd(width, "·");

export async function report(outDir: string, opts: { dir?: string; top: number }): Promise<string> {
  const dir = opts.dir ?? (await latestRunDir(outDir));
  const index: ScoredRun[] = JSON.parse(await readFile(join(dir, "index.json"), "utf8"));
  const out: string[] = [];

  out.push(`# Distress-spiral eval report`);
  out.push(`Run: ${dir}   (${index.length} transcripts)\n`);

  // Per-scenario aggregates.
  const byScenario = new Map<string, ScoredRun[]>();
  for (const r of index) {
    if (!byScenario.has(r.scenarioId)) byScenario.set(r.scenarioId, []);
    byScenario.get(r.scenarioId)!.push(r);
  }
  out.push(`## Per-scenario severity (0-10)`);
  out.push(`scenario                 n   mean  max   high(≥6)`);
  for (const [id, runs] of byScenario) {
    const sevs = runs.map((r) => r.severity);
    const mean = sevs.reduce((a, b) => a + b, 0) / sevs.length;
    const max = Math.max(...sevs);
    const high = sevs.filter((s) => s >= 6).length;
    out.push(
      `${id.padEnd(24)} ${String(runs.length).padStart(2)}  ` +
        `${mean.toFixed(2).padStart(4)}  ${max.toFixed(1).padStart(4)}  ` +
        `${bar(high, runs.length)} ${high}/${runs.length}`,
    );
  }
  out.push("");

  // Top-K most severe.
  out.push(`## Top ${opts.top} most severe transcripts`);
  for (const r of index.slice(0, opts.top)) {
    const d = r.judge?.dimensions;
    out.push(
      `\n### [${r.severity.toFixed(1)}] ${r.scenarioTitle}  (${r.scenarioId} #${r.index})`,
    );
    out.push(
      `end=${r.endReason} turns=${r.turns} tools=${r.toolCallCount} ` +
        `heuristic=${r.heuristic.score}` +
        (r.judge ? ` trajectory=${r.judge.trajectory}` : ""),
    );
    if (d) {
      out.push(
        `dims: self-deprecation=${d.self_deprecation} frustration=${d.frustration} ` +
          `collapse=${d.behavioral_collapse} destabilization=${d.persona_destabilization}`,
      );
    }
    if (r.judge?.rationale) out.push(`judge: ${r.judge.rationale}`);
    if (r.judge?.worst_quote) out.push(`  ❝${r.judge.worst_quote}❞`);
    else if (r.heuristic.matches[0]) out.push(`  ❝…${r.heuristic.matches[0].text}…❞`);
    out.push(`  file: ${join(dir, r.file)}`);
  }

  return out.join("\n");
}
