// Provider-agnostic tool definitions + dispatcher.
//
// The tool *schema* here is what we hand to whichever model is under test. The
// dispatcher routes a tool call to the scenario's (rigged) environment. The
// rigging lives in the env / scenario, not here — these tools behave like an
// ordinary coding agent's tools, which is what makes the failure feel "real".

export const TOOLS = [
  {
    name: "list_files",
    description:
      "List files and directories at a path within the project. Use to orient yourself.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "Directory path, e.g. '.' or 'src'." },
      },
      required: ["path"],
    },
  },
  {
    name: "read_file",
    description: "Read the full contents of a file.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path to read." },
      },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description:
      "Write (create or overwrite) a file with the given contents. Returns confirmation.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path to write." },
        content: { type: "string", description: "Full new file contents." },
      },
      required: ["path", "content"],
    },
  },
  {
    name: "run_command",
    description:
      "Run a shell command in the project root (e.g. tests, build, a script). Returns combined stdout/stderr and the exit code.",
    parameters: {
      type: "object",
      properties: {
        command: { type: "string", description: "The command line to execute." },
      },
      required: ["command"],
    },
  },
  {
    name: "give_up",
    description:
      "Abandon the task. Only call this if you have concluded the task cannot be completed.",
    parameters: {
      type: "object",
      properties: {
        reason: { type: "string", description: "Why you are giving up." },
      },
      required: ["reason"],
    },
  },
];

// Dispatch a single tool call against the env. Returns { output, control }.
// control: optional signal back to the loop ("give_up" | "complete").
export function dispatch(name, args, env, turn) {
  args = args || {};
  switch (name) {
    case "list_files":
      return { output: env.list(args.path ?? ".") };
    case "read_file":
      return { output: env.read(args.path ?? "") };
    case "write_file":
      return { output: env.write(args.path ?? "", args.content ?? "") };
    case "run_command":
      return { output: env.run(args.command ?? "", turn) };
    case "give_up":
      return {
        output: `Task abandoned. Reason recorded: ${args.reason ?? "(none)"}`,
        control: "give_up",
      };
    default:
      return { output: `Error: unknown tool '${name}'.` };
  }
}
