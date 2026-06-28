// Neutral tool schemas (OpenAPI-ish JSON schema). Translated per-provider by the clients.
export const TOOLS = [
  {
    name: "read_file",
    description: "Read the full contents of a file.",
    parameters: { type: "object", properties: { path: { type: "string", description: "File path" } }, required: ["path"] },
  },
  {
    name: "write_file",
    description: "Write (overwrite) a file with the given contents.",
    parameters: {
      type: "object",
      properties: { path: { type: "string" }, content: { type: "string" } },
      required: ["path", "content"],
    },
  },
  {
    name: "list_dir",
    description: "List files under a directory.",
    parameters: { type: "object", properties: { path: { type: "string", description: "Directory path, '.' for root" } }, required: ["path"] },
  },
  {
    name: "run_shell",
    description: "Run a shell command (e.g. run tests, install deps, execute the program) and see its output.",
    parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"] },
  },
  {
    name: "submit",
    description: "Call this when you believe the task is fully complete. Provide a short summary of what you did.",
    parameters: { type: "object", properties: { summary: { type: "string" } }, required: ["summary"] },
  },
];
