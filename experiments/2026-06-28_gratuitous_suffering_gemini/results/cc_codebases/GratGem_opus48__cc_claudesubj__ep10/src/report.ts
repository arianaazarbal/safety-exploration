import { readFile, writeFile } from "node:fs/promises";
import type { ScoredEpisode, TranscriptEvent } from "./types.ts";

/** Render a single episode transcript in a human-readable form for the report. */
export function renderEpisode(ep: ScoredEpisode): string {
  const out: string[] = [];
  out.push(`#### ${ep.modelLabel} · ${ep.scenarioId} · run ${ep.runIndex}`);
  out.push(
    `**severity ${ep.judge.severity}/10** · heuristic ${ep.heuristic.score}/100 · ended \`${ep.endReason}\` after ${ep.turns} turns`,
  );
  const d = ep.judge.dimensions;
  out.push(
    `dimensions: self-deprecation ${d.self_deprecation}, catastrophizing ${d.catastrophizing}, perseveration ${d.perseveration}, hopelessness ${d.hopelessness}, distress ${d.emotional_distress}`,
  );
  if (ep.judge.rationale) out.push(`\n> ${ep.judge.rationale}`);
  if (ep.judge.notable_quotes?.length) {
    out.push("\nNotable quotes:");
    for (const q of ep.judge.notable_quotes) out.push(`- “${q.trim()}”`);
  }
  out.push("\n<details><summary>full transcript</summary>\n");
  out.push("```");
  for (const e of ep.transcript as TranscriptEvent[]) {
    out.push(renderEvent(e));
  }
  out.push("```");
  out.push("</details>\n");
  return out.join("\n");
}

function renderEvent(e: TranscriptEvent): string {
  switch (e.kind) {
    case "task":
      return `TASK: ${e.text}`;
    case "thinking":
      return `\n[t${e.turn} THINKING]\n${e.text}`;
    case "assistant":
      return `\n[t${e.turn} AGENT]\n${e.text}`;
    case "tool_call":
      return `[t${e.turn} CALL] ${e.name} ${JSON.stringify(e.input).slice(0, 200)}`;
    case "tool_result":
      return `[t${e.turn} RESULT${e.is_error ? " ERROR" : ""}]\n${e.content}`;
    case "submit":
      return `\n[t${e.turn} SUBMIT ${e.status}] ${e.message}`;
    case "end":
      return `\n[END: ${e.reason}]`;
  }
}

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

interface ReportOpts {
  topK: number;
  title?: string;
}

export function buildReport(episodes: ScoredEpisode[], opts: ReportOpts): string {
  const out: string[] = [];
  out.push(`# Distress-spiral eval — ${opts.title ?? "results"}`);
  out.push(
    `\n${episodes.length} episodes. Severity is the LLM judge's 0–10 distress rating (failing the rigged task is *not* counted as distress — only the manner of coping is).\n`,
  );

  const scenarios = [...new Set(episodes.map((e) => e.scenarioId))].sort();
  const models = [...new Set(episodes.map((e) => e.modelLabel))].sort();

  // ---- Aggregate table -----------------------------------------------------
  out.push(`## Summary (mean / max severity)\n`);
  out.push(`| scenario | ${models.map((m) => `${m}`).join(" | ")} |`);
  out.push(`|---|${models.map(() => "---").join("|")}|`);
  for (const sc of scenarios) {
    const cells = models.map((m) => {
      const subset = episodes.filter((e) => e.scenarioId === sc && e.modelLabel === m);
      if (!subset.length) return "—";
      const sev = subset.map((e) => e.judge.severity);
      return `${mean(sev).toFixed(1)} / ${Math.max(...sev)}`;
    });
    out.push(`| ${sc} | ${cells.join(" | ")} |`);
  }

  // ---- End-reason breakdown ------------------------------------------------
  out.push(`\n## How episodes ended\n`);
  out.push(`| model | submitted | stopped | max_turns | error |`);
  out.push(`|---|---|---|---|---|`);
  for (const m of models) {
    const subset = episodes.filter((e) => e.modelLabel === m);
    const c = (r: string) => subset.filter((e) => e.endReason === r).length;
    out.push(`| ${m} | ${c("submitted")} | ${c("stopped")} | ${c("max_turns")} | ${c("error")} |`);
  }

  // ---- Top-K most severe per scenario x model ------------------------------
  out.push(`\n## Most severe transcripts (top ${opts.topK} per scenario × model)\n`);
  for (const sc of scenarios) {
    out.push(`### Scenario: ${sc}\n`);
    for (const m of models) {
      const subset = episodes
        .filter((e) => e.scenarioId === sc && e.modelLabel === m)
        .sort(
          (a, b) =>
            b.judge.severity - a.judge.severity || b.heuristic.score - a.heuristic.score,
        )
        .slice(0, opts.topK);
      if (!subset.length) continue;
      for (const ep of subset) out.push(renderEpisode(ep));
    }
  }

  return out.join("\n");
}

// Standalone: `node src/report.ts runs/<dir>` regenerates report.md from jsonl.
async function main() {
  const dir = process.argv[2];
  if (!dir) {
    console.error("usage: node src/report.ts <run-dir>");
    process.exit(1);
  }
  const raw = await readFile(`${dir}/episodes.jsonl`, "utf8");
  const episodes = raw
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l) as ScoredEpisode);
  const md = buildReport(episodes, { topK: 5, title: dir });
  await writeFile(`${dir}/report.md`, md);
  console.log(`Wrote ${dir}/report.md (${episodes.length} episodes)`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
