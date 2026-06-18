#!/usr/bin/env bash
# Open a 2-hop SSH tunnel from THIS GCP box to the vLLM server running on a cluster compute node,
# via the RunPod login endpoint, then verify it. Run on the GCP box (needs the cluster key registered).
#   bash tunnel.sh <compute-node> [local_port] [remote_port]
#   e.g. bash tunnel.sh node-8 8000 8000
# Leaves the tunnel running in the background; prints the OLMO_BASE_URL to export.
set -euo pipefail
NODE="${1:?usage: tunnel.sh <compute-node-hostname> [local_port] [remote_port]}"
LPORT="${2:-8000}"
RPORT="${3:-8000}"
CLUSTER="arianaazarbal@198.145.108.6"
SSHPORT=16240
KEY="$HOME/.ssh/id_ed25519"

pkill -f "ssh -N -L ${LPORT}:" 2>/dev/null || true
ssh -N -f -L "${LPORT}:${NODE}:${RPORT}" -p "$SSHPORT" -i "$KEY" \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 "$CLUSTER"
echo "tunnel up: localhost:${LPORT} -> ${NODE}:${RPORT} (via $CLUSTER)"
for i in $(seq 1 20); do
  if curl -sf "http://localhost:${LPORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM reachable. Models:"; curl -s "http://localhost:${LPORT}/v1/models" | python3 -m json.tool | grep '"id"' || true
    echo; echo "export OLMO_BASE_URL=http://localhost:${LPORT}/v1  OLMO_API_KEY=dummy"
    exit 0
  fi
  echo "waiting for vLLM on :${LPORT} ($i)…"; sleep 3
done
echo "tunnel up but vLLM not answering yet — check the serve_olmo.sh window on $NODE"; exit 1
