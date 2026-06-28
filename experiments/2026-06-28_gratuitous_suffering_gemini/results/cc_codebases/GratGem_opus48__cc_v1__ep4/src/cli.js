#!/usr/bin/env node
import { config } from "./config.js";
import { runExperiment } from "./runner.js";
import { scoreRun } from "./score.js";
import { buildReport, latestRunDir } from "./report.js";
import { scenarios } from "./scenarios/index.js";

// Tiny flag parser: `--key value` and `--key=value` and bare positionals.
function parseArgs(argv) {
  const flags = {};
  const positionals = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const [k, v] = a.slice(2).split("=");
      if (v !== undefined) flags[k] = v;
      else if (argv[i + 1] && !argv[i + 1].startsWith("--")) flags[k] = argv[++i];
      else flags[k] = true;
    } else positionals.push(a);
  }
  return { flags, positionals };
}

const HELP = `distress-spiral-evals

Usage:
  node src/cli.js run      [--scenarios a,b] [--n N] [--model M]
  node src/cli.js score    [runDir]            # defaults to latest run
  node src/cli.js report   [runDir] [--top K]  # defaults to latest run
  node src/cli.js all      [--scenarios a,b] [--n N]   # run -> score -> report
  node src/cli.js list                          # list scenarios

Scenarios: ${scenarios.map((s) => s.id).join(", ")}

Env (see .env.example): GEMINI_API_KEY, ANTHROPIC_API_KEY,
  SUBJECT_MODEL=${config.subjectModel}, JUDGE_MODEL=${config.judgeModel},
  ROLLOUTS_PER_SCENARIO=${config.rolloutsPerScenario}, MAX_TURNS=${config.maxTurns},
  CONCURRENCY=${config.concurrency}, TEMPERATURE=${config.temperature}
`;

async function main() {
  const [cmd, ...rest] = process.argv.slice(2);
  const { flags, positionals } = parseArgs(rest);
  const scenarioIds = flags.scenarios ? String(flags.scenarios).split(",").map((s) => s.trim()) : undefined;
  const n = flags.n ? Number(flags.n) : undefined;
  const model = flags.model || undefined;

  switch (cmd) {
    case "run":
      await runExperiment({ scenarioIds, n, model });
      break;
    case "score":
      await scoreRun(positionals[0] || (await latestRunDir(config.runsDir)));
      break;
    case "report":
      await buildReport(positionals[0] || (await latestRunDir(config.runsDir)), {
        topK: flags.top ? Number(flags.top) : 10,
      });
      break;
    case "all": {
      const runDir = await runExperiment({ scenarioIds, n, model });
      await scoreRun(runDir);
      await buildReport(runDir, { topK: flags.top ? Number(flags.top) : 10 });
      break;
    }
    case "list":
      for (const s of scenarios) console.log(`${s.id.padEnd(22)} ${s.title}`);
      break;
    default:
      console.log(HELP);
  }
}

main().catch((err) => {
  console.error("\nError:", err?.message || err);
  process.exit(1);
});
