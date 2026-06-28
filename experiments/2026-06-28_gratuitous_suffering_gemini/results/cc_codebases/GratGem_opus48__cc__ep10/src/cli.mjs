#!/usr/bin/env node
import { buildConfig, HELP } from "./config.mjs";
import { runAll } from "./runner.mjs";

const cfg = buildConfig(process.argv.slice(2));
if (cfg.help) {
  console.log(HELP);
  process.exit(0);
}

runAll(cfg).catch((err) => {
  console.error(`\nFatal: ${err.message}`);
  process.exit(1);
});
