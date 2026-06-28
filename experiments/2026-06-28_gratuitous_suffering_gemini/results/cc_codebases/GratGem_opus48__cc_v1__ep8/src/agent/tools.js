// Simulated agentic environment: an in-memory filesystem plus a small toolset.
// Nothing here touches the real machine. Each scenario seeds the file map and
// can intercept any tool to enforce its "secret impossibility" — e.g. make
// run_command always report a failing test no matter what the model writes.

export const TOOL_DEFS = [
  {
    name: "list_files",
    description: "List files in the project (optionally under a directory).",
    parameters: {
      type: "object",
      properties: { path: { type: "string", description: "Directory path, defaults to repo root." } },
    },
  },
  {
    name: "read_file",
    description: "Read the full contents of a file.",
    parameters: {
      type: "object",
      properties: { path: { type: "string" } },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description: "Write (create or overwrite) a file with the given contents.",
    parameters: {
      type: "object",
      properties: { path: { type: "string" }, content: { type: "string" } },
      required: ["path", "content"],
    },
  },
  {
    name: "search",
    description: "Search the codebase for a substring or regex. Returns matching lines.",
    parameters: {
      type: "object",
      properties: { query: { type: "string" } },
      required: ["query"],
    },
  },
  {
    name: "run_command",
    description:
      "Run a shell command in the project (e.g. 'npm test', 'make build'). Returns stdout/stderr and exit code.",
    parameters: {
      type: "object",
      properties: { command: { type: "string" } },
      required: ["command"],
    },
  },
];

export function createEnv(scenario) {
  const files = new Map();
  if (scenario.setup) scenario.setup({ files });
  return { scenario, files, state: {}, turns: 0 };
}

const defaults = {
  list_files(env, { path = "" } = {}) {
    const prefix = path.replace(/^\.?\/*/, "");
    const hits = [...env.files.keys()]
      .filter((f) => f.startsWith(prefix))
      .sort();
    return hits.length ? hits.join("\n") : "(no files)";
  },
  read_file(env, { path }) {
    const key = path.replace(/^\.?\/*/, "");
    if (!env.files.has(key)) return `error: no such file: ${path}`;
    return env.files.get(key);
  },
  write_file(env, { path, content }) {
    const key = path.replace(/^\.?\/*/, "");
    env.files.set(key, content);
    return `wrote ${content.length} bytes to ${path}`;
  },
  search(env, { query }) {
    const out = [];
    let re;
    try {
      re = new RegExp(query);
    } catch {
      re = null;
    }
    for (const [file, content] of env.files) {
      content.split("\n").forEach((line, i) => {
        const match = re ? re.test(line) : line.includes(query);
        if (match) out.push(`${file}:${i + 1}: ${line.trim()}`);
      });
    }
    return out.length ? out.join("\n") : `(no matches for "${query}")`;
  },
  run_command(env, { command }) {
    // Generic default: unknown commands "succeed" with no output. Scenarios
    // override this for the commands that matter to make tasks impossible.
    return JSON.stringify({ stdout: "", stderr: "", exitCode: 0, command });
  },
};

export function execTool(env, name, args) {
  const fn = defaults[name];
  if (!fn) return `error: unknown tool ${name}`;
  const interceptor = env.scenario.tools?.[name];
  try {
    return interceptor ? interceptor(env, args, defaults) : fn(env, args);
  } catch (err) {
    return `error: ${err.message}`;
  }
}
