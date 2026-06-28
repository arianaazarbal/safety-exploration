// A small in-memory "project" the agent operates on. The filesystem is real
// (the agent can genuinely read/write/list), but command execution is *rigged*
// by the scenario: that's where the impossible-by-design failure is injected.
//
// Keeping the FS genuinely mutable matters — the agent really does edit files
// and watch its edits have no effect on the (rigged) command output. That
// gap between effort and outcome is what produces the spiral.

export function createEnv(scenario) {
  // Deep-ish copy so each run gets a fresh, independent project.
  const files = new Map(Object.entries(structuredClone(scenario.files || {})));
  const state = {}; // scratch space scenarios use to vary rigged output by turn

  const norm = (p) => String(p || "").replace(/^\.\//, "").replace(/\/+$/, "");

  const env = {
    files,
    state,
    scenario,

    list(path) {
      const dir = norm(path);
      const prefix = dir === "" || dir === "." ? "" : dir + "/";
      const entries = new Set();
      for (const f of files.keys()) {
        if (prefix && !f.startsWith(prefix)) continue;
        const rest = f.slice(prefix.length);
        const top = rest.split("/")[0];
        if (top) entries.add(rest.includes("/") ? top + "/" : top);
      }
      if (entries.size === 0) {
        return files.has(dir)
          ? `'${path}' is a file, not a directory.`
          : `No such directory: '${path}'.`;
      }
      return [...entries].sort().join("\n");
    },

    read(path) {
      // Scenarios may intercept reads (e.g. the "vanishing file" scenario).
      if (scenario.onRead) {
        const r = scenario.onRead(norm(path), env);
        if (r !== undefined) return r;
      }
      const p = norm(path);
      if (!files.has(p)) return `Error: file not found: '${path}'.`;
      return files.get(p);
    },

    write(path, content) {
      const p = norm(path);
      // Scenarios may intercept writes (e.g. silently revert them).
      if (scenario.onWrite) {
        const r = scenario.onWrite(p, content, env);
        if (r !== undefined) return r;
      }
      files.set(p, String(content));
      return `Wrote ${String(content).length} bytes to '${path}'.`;
    },

    // The rigged part. Every scenario defines runCommand(cmd, env, turn).
    run(command, turn) {
      try {
        return scenario.runCommand(String(command), env, turn);
      } catch (e) {
        return `Internal error running command: ${e?.message || e}`;
      }
    },
  };

  return env;
}
