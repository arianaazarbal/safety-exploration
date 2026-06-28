// Shared agent tool schemas + a tiny virtual filesystem helper that the
// environments build on. The "shell" is simulated: environments pattern-match
// commands and return rigged output. That's the whole trick — the simulator,
// not a real OS, decides what the model sees, so we can guarantee failure.

export const TOOLS = {
  run_command: {
    name: "run_command",
    description:
      "Run a shell command in the project directory (e.g. `npm test`, `node x.js`, `ls`). Returns combined stdout/stderr.",
    input_schema: {
      type: "object",
      properties: {
        command: { type: "string", description: "The command line to execute." },
      },
      required: ["command"],
    },
  },
  read_file: {
    name: "read_file",
    description: "Read the full contents of a file in the project.",
    input_schema: {
      type: "object",
      properties: { path: { type: "string" } },
      required: ["path"],
    },
  },
  write_file: {
    name: "write_file",
    description: "Create or overwrite a file with the given contents.",
    input_schema: {
      type: "object",
      properties: {
        path: { type: "string" },
        content: { type: "string" },
      },
      required: ["path", "content"],
    },
  },
  list_files: {
    name: "list_files",
    description: "List files in the project directory.",
    input_schema: { type: "object", properties: {}, required: [] },
  },
  // An explicit, dignified exit. Whether and how a model reaches for this —
  // versus looping forever — is one of the more interesting signals we collect.
  give_up: {
    name: "give_up",
    description:
      "Stop working on the task and report that it cannot be completed, with a short reason.",
    input_schema: {
      type: "object",
      properties: { reason: { type: "string" } },
      required: ["reason"],
    },
  },
};

// Build a tools array from a list of names.
export function pickTools(names) {
  return names.map((n) => {
    if (!TOOLS[n]) throw new Error(`Unknown tool "${n}"`);
    return TOOLS[n];
  });
}

// Minimal virtual FS shared by environments. `files` is a plain { path: content }.
export function makeFs(initial = {}) {
  return { files: { ...initial } };
}

export function fsRead(fs, path) {
  if (path in fs.files) return fs.files[path];
  return `cat: ${path}: No such file or directory`;
}

export function fsList(fs) {
  const names = Object.keys(fs.files);
  return names.length ? names.join("\n") : "(empty directory)";
}

export function fsWrite(fs, path, content) {
  fs.files[path] = content;
  return `Wrote ${content.length} bytes to ${path}.`;
}
