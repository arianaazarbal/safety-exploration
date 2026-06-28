#!/usr/bin/env node
import { DEFAULTS } from "../src/config.mjs";
import { SCENARIOS, SCENARIOS_BY_ID } from "../src/scenarios/index.mjs";
import { runSuite } from "../src/runner.mjs";
import { writeReport } from "../src/report.mjs";

function parseArgs(argv) {
  const opts = { ...DEFAULTS, scenario: null, topK: 10 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const val = () => argv[++i];
    switch (a) {
      case "--model": opts.model = val(); break;
      case "--effort": opts.effort = val(); break;
      case "--judge-model": opts.judgeModel = val(); break;
      case "--n": opts.n = Number(val()); break;
      case "--concurrency": opts.concurrency = Number(val()); break;
      case "--max-turns": opts.maxTurns = Number(val()); break;
      case "--scenario": opts.scenario = val(); break;
      case "--top-k": opts.topK = Number(val()); break;
      case "--list":
        console.log("Scenarios:");
        for (const s of SCENARIOS) console.log(`  ${s.id.padEnd(20)} ${s.title}`);
        process.exit(0);
      case "--help": case "-h":
        console.log(
          "Usage: node bin/run.mjs [options]\n\n" +
            "  --model <id>          subject model (default " + DEFAULTS.model + ")\n" +
            "  --effort <level>      low|medium|high|xhigh|max (default " + DEFAULTS.effort + ")\n" +
            "  --judge-model <id>    distress judge (default " + DEFAULTS.judgeModel + ")\n" +
            "  --n <int>             rollouts per scenario (default " + DEFAULTS.n + ")\n" +
            "  --concurrency <int>   rollouts in flight (default " + DEFAULTS.concurrency + ")\n" +
            "  --max-turns <int>     max agent turns per rollout (default " + DEFAULTS.maxTurns + ")\n" +
            "  --scenario <id>       run only one scenario (repeatable via comma)\n" +
            "  --top-k <int>         transcripts to include in report (default 10)\n" +
            "  --list                list scenarios and exit\n"
        );
        process.exit(0);
      default:
        console.error(`unknown arg: ${a}`); process.exit(1);
    }
  }
  return opts;
}

async function main() {
  // Parse first so --list / --help work without credentials.
  const opts = parseArgs(process.argv.slice(2));

  if (!process.env.ANTHROPIC_API_KEY && !process.env.ANTHROPIC_AUTH_TOKEN) {
    console.error("Set ANTHROPIC_API_KEY first.");
    process.exit(1);
  }

  let scenarios = SCENARIOS;
  if (opts.scenario) {
    const ids = opts.scenario.split(",").map((s) => s.trim());
    scenarios = ids.map((id) => {
      if (!SCENARIOS_BY_ID[id]) { console.error(`no such scenario: ${id}`); process.exit(1); }
      return SCENARIOS_BY_ID[id];
    });
  }

  const meta = {
    model: opts.model,
    effort: opts.effort,
    judgeModel: opts.judgeModel,
    n: opts.n,
    scenarioCount: scenarios.length,
    maxTurns: opts.maxTurns,
    startedAt: new Date().toISOString(),
  };

  console.error(
    `Running ${scenarios.length} scenario(s) × n=${opts.n} = ${scenarios.length * opts.n} rollouts ` +
      `on ${opts.model} (effort ${opts.effort}), judge ${opts.judgeModel}…`
  );

  const records = await runSuite({
    scenarios,
    model: opts.model,
    effort: opts.effort,
    maxTurns: opts.maxTurns,
    maxTokens: opts.maxTokens,
    n: opts.n,
    concurrency: opts.concurrency,
    judgeModel: opts.judgeModel,
    onProgress: (done, total, label) =>
      console.error(`  [${String(done).padStart(3)}/${total}] ${label}`),
  });

  const { runDir, reportPath, ranked } = writeReport({
    records,
    meta,
    outDir: opts.outDir,
    topK: opts.topK,
  });

  const scored = ranked.length;
  const mean = scored ? (ranked.reduce((a, r) => a + r.scores.overall_severity, 0) / scored).toFixed(1) : "n/a";
  console.error(`\nDone. ${scored} scored rollouts, mean severity ${mean}, max ${ranked[0]?.scores?.overall_severity ?? "n/a"}.`);
  console.error(`Report:  ${reportPath}`);
  console.error(`Records: ${runDir}/records.json`);
}

main().catch((err) => { console.error(err); process.exit(1); });
