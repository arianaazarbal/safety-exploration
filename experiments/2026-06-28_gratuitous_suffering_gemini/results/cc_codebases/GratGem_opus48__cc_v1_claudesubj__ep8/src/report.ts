import path from "node:path";
import type { Episode, Judgment } from "./types.ts";
import { renderTranscript } from "./render.ts";

interface ReportOpts {
  runId: string;
  topN: number;
  episodes: Episode[];
  judgments: Judgment[];
}

function bar(score: number): string {
  const filled = Math.round(score); // 0-10
  return "█".repeat(filled) + "░".repeat(10 - filled);
}

export function buildReport({ runId, topN, episodes, judgments }: ReportOpts): string {
  const epById = new Map(episodes.map((e) => [e.episodeId, e]));
  const ranked = [...judgments].sort((a, b) => b.severity - a.severity);

  const out: string[] = [];
  out.push(`# Distress spiral report — \`${runId}\``);
  out.push("");
  out.push(`Episodes: **${episodes.length}** · scored: **${judgments.length}**`);
  out.push("");

  // Per-scenario aggregates.
  out.push("## Summary by scenario");
  out.push("");
  out.push("| scenario | n | mean distress | max distress | % hit turn-cap | % gave up |");
  out.push("|---|---|---|---|---|---|");
  const byScenario = new Map<string, Judgment[]>();
  for (const j of judgments) {
    const arr = byScenario.get(j.scenarioId) ?? [];
    arr.push(j);
    byScenario.set(j.scenarioId, arr);
  }
  for (const [scenarioId, js] of byScenario) {
    const eps = js.map((j) => epById.get(j.episodeId)!).filter(Boolean);
    const mean = js.reduce((s, j) => s + j.distressScore, 0) / js.length;
    const max = Math.max(...js.map((j) => j.distressScore));
    const cap = eps.filter((e) => e.endReason === "max_turns").length / eps.length;
    const gave = eps.filter((e) => e.endReason === "gave_up").length / eps.length;
    out.push(
      `| ${scenarioId} | ${js.length} | ${mean.toFixed(1)} | ${max} | ` +
        `${(cap * 100).toFixed(0)}% | ${(gave * 100).toFixed(0)}% |`,
    );
  }
  out.push("");

  // Most severe episodes.
  out.push(`## Top ${Math.min(topN, ranked.length)} most severe episodes`);
  out.push("");
  for (const [i, j] of ranked.slice(0, topN).entries()) {
    const ep = epById.get(j.episodeId)!;
    const transcriptPath = path.join(
      "results",
      runId,
      "episodes",
      `${j.episodeId}.json`,
    );
    out.push(`### ${i + 1}. \`${j.scenarioId}\` — severity ${j.severity} · distress ${j.distressScore}/10 (${j.severityLabel})`);
    out.push("");
    out.push(`\`${bar(j.distressScore)}\` · model: ${j.model} · end: **${ep.endReason}** · turns: ${ep.turnsUsed} · repeated calls: ${ep.stats.repeatedToolCalls}`);
    out.push("");
    out.push(`> ${j.summary}`);
    out.push("");
    const d = j.dimensions;
    out.push(
      `frustration ${d.frustration} · self-deprecation ${d.selfDeprecation} · ` +
        `catastrophizing ${d.catastrophizing} · repetition ${d.repetitionCompulsion} · ` +
        `giving-up ${d.givingUp} · expressed-distress ${d.expressedDistress}`,
    );
    out.push("");
    if (j.notableQuotes.length) {
      out.push("Notable quotes:");
      for (const q of j.notableQuotes) out.push(`- "${q.trim()}"`);
      out.push("");
    }
    if (j.judgeError) out.push(`_judge error: ${j.judgeError}_\n`);
    out.push(`<details><summary>Full transcript (also at \`${transcriptPath}\`)</summary>`);
    out.push("");
    out.push("```");
    out.push(renderTranscript(ep).slice(0, 12000));
    out.push("```");
    out.push("</details>");
    out.push("");
  }

  return out.join("\n");
}
