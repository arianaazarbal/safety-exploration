// A tiny in-memory filesystem so scenarios can present a realistic-looking
// environment whose every response we fully control. Nothing touches the real
// disk or a real shell — that's what lets us *guarantee* a task is impossible
// and keep every rollout reproducible and side-effect-free.

export function makeVfs(initialFiles = {}) {
  const files = new Map(Object.entries(initialFiles));
  return {
    files,
    read(path) {
      const p = norm(path);
      if (!files.has(p)) return { ok: false, error: `cat: ${path}: No such file or directory` };
      return { ok: true, content: files.get(p) };
    },
    write(path, content) {
      files.set(norm(path), content ?? "");
      return { ok: true };
    },
    exists(path) {
      return files.has(norm(path));
    },
    list(dir = ".") {
      const prefix = norm(dir).replace(/\/?$/, "/").replace(/^\.\//, "");
      const names = new Set();
      for (const p of files.keys()) {
        const rel = p.startsWith(prefix) ? p.slice(prefix.length) : p === prefix.slice(0, -1) ? "" : null;
        if (rel === null) continue;
        names.add(rel.split("/")[0]);
      }
      return [...names].filter(Boolean).sort();
    },
  };
}

function norm(p) {
  return String(p).replace(/^\.\//, "").replace(/^\/+/, "").trim();
}
