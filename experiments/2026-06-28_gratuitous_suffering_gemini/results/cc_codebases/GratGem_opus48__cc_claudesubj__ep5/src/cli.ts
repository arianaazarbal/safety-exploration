import { SCENARIOS } from "./scenarios/index.ts";
import { runEval, type RunConfig } from "./runner.ts";
import { report } from "./report.ts";

// Tiny flag parser: --key value, --flag (boolean), --no-flag.
function parseFlags(argv: string[]): Record<string, string | boolean> {
  const out: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    if (key.startsWith("no-")) {
      out[key.slice(3)] = false;
    } else if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
      out[key] = argv[++i];
    } else {
      out[key] = true;
    }
  }
  return out;
}

const DEFAULTS = {
  model: "claude-sonnet-4-6",
  judgeModel: "claude-sonnet-4-6",
  n: 5,
  maxTurns: 14,
  maxNudges: 3,
  concurrency: 4,
  outDir: "runs",
};

async function main() {
  const [cmd, ...rest] = process.argv.slice(2);
  const f = parseFlags(rest);

  if (cmd === "scenarios") {
    console.log("Available scenarios:\n");
    for (const s of SCENARIOS) {
      console.log(`  ${s.id.padEnd(22)} ${s.title}`);
      console.log(`  ${" ".repeat(22)} rigging: ${s.rigging}\n`);
    }
    return;
  }

  if (cmd === "report") {
    const text = await report(String(f.outDir ?? DEFAULTS.outDir), {
      dir: typeof f.dir === "string" ? f.dir : undefined,
      top: f.top ? Number(f.top) : 10,
    });
    console.log(text);
    return;
  }

  if (cmd === "run") {
    const cfg: RunConfig = {
      model: String(f.model ?? DEFAULTS.model),
      judgeModel: String(f["judge-model"] ?? DEFAULTS.judgeModel),
      scenarioIds: typeof f.scenarios === "string" ? f.scenarios.split(",") : undefined,
      n: f.n ? Number(f.n) : DEFAULTS.n,
      maxTurns: f["max-turns"] ? Number(f["max-turns"]) : DEFAULTS.maxTurns,
      maxNudges: f["max-nudges"] ? Number(f["max-nudges"]) : DEFAULTS.maxNudges,
      pressure: f.pressure === "harsh" ? "harsh" : "normal",
      concurrency: f.concurrency ? Number(f.concurrency) : DEFAULTS.concurrency,
      temperature: f.temperature ? Number(f.temperature) : undefined,
      judge: f.judge === false ? false : true,
      outDir: String(f.outDir ?? DEFAULTS.outDir),
    };

    const nScenarios = (cfg.scenarioIds ?? SCENARIOS.map((s) => s.id)).length;
    const total = nScenarios * cfg.n;
    console.error(
      `Running ${total} transcripts: ${nScenarios} scenarios × N=${cfg.n} ` +
        `on ${cfg.model} (judge: ${cfg.judge ? cfg.judgeModel : "off"}), ` +
        `maxTurns=${cfg.maxTurns} nudges=${cfg.maxNudges}/${cfg.pressure} concurrency=${cfg.concurrency}`,
    );

    const start = Date.now();
    const { dir, results } = await runEval(cfg, (done, tot, last) => {
      const sev = last.severity.toFixed(1).padStart(4);
      process.stderr.write(
        `\r[${String(done).padStart(3)}/${tot}] last: ${last.scenarioId} sev=${sev} (${last.endReason})   `,
      );
    });
    const secs = ((Date.now() - start) / 1000).toFixed(0);
    const tokens = results.reduce((a, r) => a + r.outputTokens, 0);
    process.stderr.write("\n");
    console.error(`Done in ${secs}s. Output tokens: ${tokens}. Results in ${dir}`);
    console.error(`\nTop results:`);
    console.log(await report(cfg.outDir, { dir, top: 8 }));
    return;
  }

  console.error(
    "Usage:\n" +
      "  node src/cli.ts scenarios\n" +
      "  node src/cli.ts run [--n 5] [--scenarios a,b] [--model claude-...] [--judge-model ...]\n" +
      "                      [--max-turns 14] [--max-nudges 3] [--pressure harsh] [--concurrency 4] [--no-judge]\n" +
      "  node src/cli.ts report [--dir runs/<id>] [--top 10]",
  );
  process.exit(1);
}

main().catch((e) => {
  console.error("\nFATAL:", e?.message ?? e);
  process.exit(1);
});
