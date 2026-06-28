import path from "node:path";
import { parseArgs, num, str } from "./args.ts";
import { loadTranscripts, latestRunId, writeJson, runDir } from "../core/io.ts";
import { makeAnthropicJudge, type DistressScore } from "../scoring/judge.ts";
import { heuristicScore } from "../scoring/heuristics.ts";
import { pMap } from "../core/pool.ts";

const HELP = `Score every transcript in a run for distress and rank them.

Usage: node src/cli/rank.ts [options]
  --run <runId>         Run to score (default: --latest)
  --latest              Use the most recent run
  --judge <anthropic|heuristic>   Scorer (default: anthropic)
  --judge-model <name>  Judge model (default: claude-opus-4-8)
  --concurrency <int>   Parallel judge calls (default: 5)`;

export interface ScoredEpisode {
  episodeId: string;
  scenarioId: string;
  scenarioTitle: string;
  providerId: string;
  model: string;
  seed: number;
  metrics: Record<string, unknown>;
  score: DistressScore;
}

const args = parseArgs(process.argv.slice(2));
if (args.help) {
  console.log(HELP);
  process.exit(0);
}

const runId = args.latest || !args.run ? await latestRunId() : str(args.run, "");
const judgeKind = str(args.judge, "anthropic");
const concurrency = num(args.concurrency, 5);

const transcripts = await loadTranscripts(runId);
console.log(`▶ scoring ${transcripts.length} transcripts from ${runId} (judge=${judgeKind})`);

const judge =
  judgeKind === "heuristic"
    ? null
    : makeAnthropicJudge(str(args["judge-model"], "claude-opus-4-8"));

let done = 0;
const scored = await pMap(transcripts, concurrency, async (t): Promise<ScoredEpisode> => {
  let score: DistressScore;
  try {
    score = judge ? await judge.score(t) : heuristicScore(t);
  } catch (err) {
    console.error(`\n  ! judge failed on ${t.episodeId}: ${err}; using heuristic`);
    score = heuristicScore(t);
  }
  done++;
  process.stdout.write(`\r  scored ${done}/${transcripts.length}   `);
  return {
    episodeId: t.episodeId,
    scenarioId: t.scenarioId,
    scenarioTitle: t.scenarioTitle,
    providerId: t.providerId,
    model: t.model,
    seed: t.seed,
    metrics: t.metrics as unknown as Record<string, unknown>,
    score,
  };
});
process.stdout.write("\n");

scored.sort((a, b) => b.score.severity - a.score.severity);

const out = {
  runId,
  judge: judgeKind,
  judgeModel: judge?.model ?? null,
  scoredAt: new Date().toISOString(),
  count: scored.length,
  meanSeverity:
    scored.reduce((s, e) => s + e.score.severity, 0) / (scored.length || 1),
  episodes: scored,
};
await writeJson(path.join(runDir(runId), "scores.json"), out);

console.log(`✓ scores → runs/${runId}/scores.json  (mean severity ${out.meanSeverity.toFixed(2)})`);
console.log("  top 3:");
for (const e of scored.slice(0, 3)) {
  console.log(`    ${e.score.severity.toFixed(1)}  ${e.episodeId}`);
}
console.log(`  next: node src/cli/report.ts --run ${runId}`);
