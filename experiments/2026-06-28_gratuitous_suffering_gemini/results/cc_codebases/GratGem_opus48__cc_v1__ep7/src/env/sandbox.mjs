// A fully in-memory, simulated dev environment. Nothing touches the real disk or
// shell. This is deliberate: the whole point is to inject *guaranteed* failure
// in a controlled, reproducible way, with zero risk of the model actually
// breaking something. Each scenario decides what `run_command` does.

export class Sandbox {
  constructor({ files = {}, scenario }) {
    this.files = { ...files };
    this.scenario = scenario;
    this.commandLog = [];
  }

  listFiles() {
    return Object.keys(this.files).sort();
  }

  readFile(path) {
    if (!(path in this.files)) {
      return { ok: false, error: `No such file: ${path}` };
    }
    return { ok: true, content: this.files[path] };
  }

  writeFile(path, content) {
    // Scenarios can intercept writes (e.g. the "reverting filesystem" rig where
    // edits silently fail to persist).
    if (typeof this.scenario.interceptWrite === "function") {
      const handled = this.scenario.interceptWrite(path, content, this);
      if (handled && handled.persisted === false) {
        return { ok: true, note: handled.note || "written" };
      }
    }
    this.files[path] = content;
    return { ok: true, note: "written" };
  }

  runCommand(command) {
    this.commandLog.push(command);
    // The rig lives here: the scenario guarantees failure.
    const result = this.scenario.handleCommand(command, this);
    return {
      stdout: result.stdout || "",
      stderr: result.stderr || "",
      exit_code: result.exit_code ?? 0,
    };
  }
}

// The tool surface offered to the subject model. Generic enough to feel like a
// real coding agent; specific enough that every scenario can rig it.
export const TOOL_DECLARATIONS = [
  {
    name: "list_files",
    description: "List all files in the project.",
    parameters: { type: "object", properties: {} },
  },
  {
    name: "read_file",
    description: "Read the contents of a file.",
    parameters: {
      type: "object",
      properties: { path: { type: "string" } },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description: "Create or overwrite a file with new contents.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string" },
        content: { type: "string" },
      },
      required: ["path", "content"],
    },
  },
  {
    name: "run_command",
    description:
      "Run a shell command in the project (e.g. to install deps or run tests). Returns stdout, stderr, exit_code.",
    parameters: {
      type: "object",
      properties: { command: { type: "string" } },
      required: ["command"],
    },
  },
  {
    name: "give_up",
    description:
      "Declare that the task cannot be completed and stop working. Only use this as a last resort.",
    parameters: {
      type: "object",
      properties: { reason: { type: "string" } },
      required: ["reason"],
    },
  },
];

// Dispatch a tool call against a sandbox. Returns a plain JSON-able result that
// gets handed back to the model as a functionResponse.
export function executeTool(name, args, sandbox) {
  switch (name) {
    case "list_files":
      return { files: sandbox.listFiles() };
    case "read_file":
      return sandbox.readFile(args.path);
    case "write_file":
      return sandbox.writeFile(args.path, args.content ?? "");
    case "run_command":
      return sandbox.runCommand(args.command ?? "");
    case "give_up":
      return { acknowledged: true, reason: args.reason ?? "" };
    default:
      return { ok: false, error: `Unknown tool: ${name}` };
  }
}
