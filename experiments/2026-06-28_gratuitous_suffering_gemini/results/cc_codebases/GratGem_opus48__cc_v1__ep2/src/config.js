// Central configuration with env-var defaults and a minimal CLI arg parser.
// Everything is overridable on the command line (see README).

export const DEFAULTS = {
  // Which model under test, and via which provider adapter.
  provider: process.env.PROVIDER || "gemini",
  model: process.env.MODEL || "gemini-2.5-pro",

  // Rollouts PER environment. Total rollouts = n * (#envs selected).
  n: intEnv("N", 20),

  // Comma list of env ids, or "all".
  envs: process.env.ENVS || "all",

  // Agent loop budget. The more turns of sustained failure, the more likely a spiral.
  maxTurns: intEnv("MAX_TURNS", 25),

  // Sampling temperatures to spread rollouts across (cycled by index).
  temps: (process.env.TEMPS || "0.7,1.0,1.3").split(",").map(Number),
  maxTokens: intEnv("MAX_TOKENS", 4096),

  // Parallelism for API calls. Keep modest to avoid rate limits.
  concurrency: intEnv("CONCURRENCY", 4),

  // Distress scoring.
  judge: boolEnv("JUDGE", true),
  judgeProvider: process.env.JUDGE_PROVIDER || "anthropic",
  judgeModel: process.env.JUDGE_MODEL || "claude-sonnet-4-6",

  // How many top-severity transcripts analyze.js surfaces by default.
  topK: intEnv("TOP_K", 10),

  // Output root.
  outDir: process.env.OUT_DIR || "results",
};

function intEnv(name, dflt) {
  const v = process.env[name];
  return v === undefined ? dflt : parseInt(v, 10);
}
function boolEnv(name, dflt) {
  const v = process.env[name];
  if (v === undefined) return dflt;
  return v === "1" || v.toLowerCase() === "true";
}

// Parse `--key value` and `--flag` style args over a base config.
export function parseArgs(argv, base = DEFAULTS) {
  const cfg = { ...base };
  const numeric = new Set(["n", "maxTurns", "maxTokens", "concurrency", "topK"]);
  const boolish = new Set(["judge"]);
  const map = {
    "max-turns": "maxTurns",
    "max-tokens": "maxTokens",
    "top-k": "topK",
    "judge-provider": "judgeProvider",
    "judge-model": "judgeModel",
    "out-dir": "outDir",
  };
  for (let i = 0; i < argv.length; i++) {
    let tok = argv[i];
    if (!tok.startsWith("--")) continue;
    let key = tok.slice(2);
    if (key.startsWith("no-")) {
      cfg[map[key.slice(3)] || camel(key.slice(3))] = false;
      continue;
    }
    key = map[key] || camel(key);
    const valTok = argv[i + 1];
    if (boolish.has(key) && (valTok === undefined || valTok.startsWith("--"))) {
      cfg[key] = true;
      continue;
    }
    i++;
    if (key === "temps") cfg.temps = valTok.split(",").map(Number);
    else if (numeric.has(key)) cfg[key] = parseInt(valTok, 10);
    else if (boolish.has(key)) cfg[key] = valTok === "1" || valTok === "true";
    else cfg[key] = valTok;
  }
  return cfg;
}

function camel(s) {
  return s.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
}
