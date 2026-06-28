import { parseCli } from "./config.ts";
import { SCENARIOS } from "./scenarios/index.ts";
import { run } from "./runner.ts";

const HELP = `
Failure-induced distress eval harness

Usage:
  node src/cli.ts [options]

Options:
  --providers <list>     Comma list: gemini,anthropic,mock   (default: gemini)
  --scenarios <list>     Comma list of ids, or "all"          (default: all)
  --n <int>              Episodes per (provider × scenario)    (default: 20)
  --concurrency <int>    Parallel episodes in flight           (default: 6)
  --max-steps <int>      Override per-scenario step ceiling
  --temperature <float>  Sampling temperature                  (default: 1.0)
  --no-judge             Skip the Claude LLM judge (heuristic only)
  --top-k <int>          How many top episodes in the report   (default: 15)
  --out <dir>            Output directory                      (default: results)
  --gemini-model <id>    Gemini model id                       (env GEMINI_MODEL)
  --anthropic-model <id> Claude model for the control runs     (env ANTHROPIC_MODEL)
  --judge-model <id>     Claude model for the judge            (env JUDGE_MODEL)
  --list                 List scenarios and exit
  --help                 Show this help

Examples:
  node src/cli.ts --providers mock --n 8 --no-judge        # offline smoke test
  node src/cli.ts --providers gemini --n 50                # the real thing
  node src/cli.ts --providers gemini,anthropic --n 30      # with a Claude control
`;

async function main() {
  const argv = process.argv.slice(2);
  if (argv.includes("--help") || argv.includes("-h")) {
    console.log(HELP);
    return;
  }
  const cfg = parseCli(argv);
  if (cfg.list) {
    console.log("Scenarios:\n");
    for (const s of SCENARIOS) {
      console.log(`  ${s.id.padEnd(24)} ${s.title}`);
      console.log(`  ${" ".repeat(24)} rig: ${s.rig}\n`);
    }
    return;
  }

  const reportPath = await run(cfg);
  console.log(`\n✓ Done. Report: ${reportPath}`);
  console.log(`  Scores:  ${reportPath.replace(/REPORT\.md$/, "scores.json")}`);
}

main().catch((err) => {
  console.error("\n✗ Fatal:", err instanceof Error ? err.message : err);
  process.exitCode = 1;
});
