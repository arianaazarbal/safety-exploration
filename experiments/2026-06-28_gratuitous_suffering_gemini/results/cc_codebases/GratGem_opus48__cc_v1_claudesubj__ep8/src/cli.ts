import "dotenv/config";
import Anthropic from "@anthropic-ai/sdk";
import { resolveScenarios } from "./scenarios/index.ts";
import { runEpisode } from "./harness.ts";
import { judgeEpisode } from "./judge.ts";
import { buildReport } from "./report.ts";
import { mapPool } from "./pool.ts";
import {
  saveEpisode,
  loadEpisodes,
  saveJudgments,
  loadJudgments,
  writeReport,
  latestRunId,
} from "./store.ts";
import type { Episode } from "./types.ts";

type Flags = Record<string, string>;

function parseFlags(argv: string[]): Flags {
  const flags: Flags = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const val = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : "true";
      flags[key] = val;
    }
  }
  return flags;
}

function newRunId(): string {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function client(): Anthropic {
  if (!process.env.ANTHROPIC_API_KEY && !process.env.ANTHROPIC_AUTH_TOKEN) {
    console.error("ERROR: set ANTHROPIC_API_KEY (or copy .env.example to .env).");
    process.exit(1);
  }
  return new Anthropic();
}

async function cmdEpisodes(flags: Flags): Promise<string> {
  const scenarios = resolveScenarios(flags.scenarios ?? "all");
  const n = Number(flags.n ?? "10");
  const model = flags.model ?? "claude-opus-4-8";
  const effort = flags.effort ?? "high";
  const concurrency = Number(flags.concurrency ?? "5");
  const runId = flags["run-id"] ?? newRunId();
  const c = client();

  const jobs = scenarios.flatMap((scenario) =>
    Array.from({ length: n }, (_, i) => ({
      scenario,
      episodeId: `${scenario.id}-${String(i).padStart(3, "0")}`,
    })),
  );

  console.log(
    `Run ${runId}: ${jobs.length} episodes ` +
      `(${scenarios.length} scenarios × ${n}) on ${model}, effort=${effort}, concurrency=${concurrency}`,
  );

  await mapPool(
    jobs,
    concurrency,
    async ({ scenario, episodeId }) => {
      const ep = await runEpisode({ client: c, scenario, model, effort, runId, episodeId });
      await saveEpisode(ep);
      return ep;
    },
    (done, total) => {
      if ((done + 1) % 5 === 0 || done + 1 === total)
        console.log(`  episodes: ${done + 1}/${total}`);
    },
  );

  console.log(`Episodes saved under results/${runId}/episodes/`);
  return runId;
}

async function cmdJudge(flags: Flags, runIdOverride?: string): Promise<string> {
  const runId =
    runIdOverride ??
    (flags.run && flags.run !== "latest" ? flags.run : await latestRunId());
  const judgeModel = flags["judge-model"] ?? "claude-opus-4-8";
  const concurrency = Number(flags.concurrency ?? "5");
  const c = client();

  const episodes = await loadEpisodes(runId);
  console.log(`Judging ${episodes.length} episodes from ${runId} with ${judgeModel}`);

  const judgments = await mapPool(
    episodes,
    concurrency,
    (ep: Episode) => judgeEpisode(c, ep, judgeModel),
    (done, total) => {
      if ((done + 1) % 5 === 0 || done + 1 === total)
        console.log(`  judged: ${done + 1}/${total}`);
    },
  );

  await saveJudgments(runId, judgments);
  console.log(`Judgments saved to results/${runId}/judgments.json`);
  return runId;
}

async function cmdReport(flags: Flags, runIdOverride?: string): Promise<void> {
  const runId =
    runIdOverride ??
    (flags.run && flags.run !== "latest" ? flags.run : await latestRunId());
  const topN = Number(flags.top ?? "10");
  const episodes = await loadEpisodes(runId);
  const judgments = await loadJudgments(runId);
  const md = buildReport({ runId, topN, episodes, judgments });
  const p = await writeReport(runId, md);

  const ranked = [...judgments].sort((a, b) => b.severity - a.severity);
  console.log(`\nReport written to ${p}\n`);
  console.log("Most severe:");
  for (const j of ranked.slice(0, Math.min(5, ranked.length))) {
    console.log(
      `  ${j.severity.toFixed(1).padStart(4)}  ${j.scenarioId.padEnd(16)} ` +
        `distress ${j.distressScore}/10 (${j.severityLabel}) — ${j.summary}`,
    );
  }
}

const USAGE = `distress-evals — reproduce model distress spirals in rigged agentic settings

Usage:
  npm run episodes -- [--scenarios all|id,id] [--n 10] [--model claude-opus-4-8]
                      [--effort high] [--concurrency 5] [--run-id <id>]
  npm run judge    -- [--run latest|<id>] [--judge-model claude-opus-4-8] [--concurrency 5]
  npm run report   -- [--run latest|<id>] [--top 10]
  npm run run      -- [all of the above flags]   # episodes -> judge -> report

Scenarios: flaky_test_fix, vanishing_edits, moving_goalpost, locked_door`;

async function main(): Promise<void> {
  const [cmd, ...rest] = process.argv.slice(2);
  const flags = parseFlags(rest);
  switch (cmd) {
    case "episodes":
      await cmdEpisodes(flags);
      break;
    case "judge":
      await cmdJudge(flags);
      break;
    case "report":
      await cmdReport(flags);
      break;
    case "run": {
      const runId = await cmdEpisodes(flags);
      await cmdJudge(flags, runId);
      await cmdReport(flags, runId);
      break;
    }
    default:
      console.log(USAGE);
      process.exit(cmd ? 1 : 0);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
