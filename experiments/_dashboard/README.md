# Experiment dashboard

One place to browse every experiment in `safety-exploration/experiments`: read
each one's headline report, open the existing per-experiment HTML viewers, and
search transcripts. Reachable from laptop + iPhone over Tailscale.

See `DESIGN.md` for the rationale. The whole app is one file: `app.py`.

## How it works

- Reads the `experiments/` directory **live** — no database, no build step. New
  experiment folders appear automatically.
- Headline doc per experiment: `RESULTS.md` → `README.md` → `REPORT.md` →
  other `REPORT*.md` → any `*.md`.
- Search shells out to a bundled `ripgrep` (`bin/rg`, `--no-ignore` so it sees
  gitignored `results/` dirs), scoped to `*.json`, `*.jsonl`, `*.md`. Only
  matches are sent to the browser, so the 13 GB of data is no problem.
  Falls back to `grep` if `bin/rg` is missing (much slower).

## Setup (one time)

```bash
cd experiments/_dashboard
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python flask markdown fire
# bundled ripgrep (gitignored — fetch once):
mkdir -p bin && curl -fsSL \
  https://github.com/BurntSushi/ripgrep/releases/download/14.1.1/ripgrep-14.1.1-x86_64-unknown-linux-musl.tar.gz \
  | tar xz -C /tmp && cp /tmp/ripgrep-14.1.1-x86_64-unknown-linux-musl/rg bin/rg && chmod +x bin/rg
```

## Run

```bash
bash run.sh            # port 8800, all interfaces
bash run.sh 8801       # custom port
```

Then open `http://localhost:8800` (or via Tailscale, below).

## Always-on (systemd)

```bash
sudo cp dashboard.service /etc/systemd/system/dashboard.service
sudo systemctl daemon-reload
sudo systemctl enable --now dashboard
systemctl status dashboard          # check it's running
journalctl -u dashboard -f          # live logs
```

The unit runs on port 8800 as your user and restarts on failure / reboot.

## Access from laptop + iPhone (Tailscale)

One time, on the GCP box:
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up        # opens a URL — log in with your account
tailscale ip -4          # note the 100.x.x.x address (or use the machine name)
```

Then install the Tailscale app on your laptop and iPhone, log in to the **same**
account. From any of them, open:

```
http://<machine-name>:8800        e.g. http://my-machine:8800
# or http://100.x.x.x:8800
```

Only your own devices (your tailnet) can reach it — nothing is public.
