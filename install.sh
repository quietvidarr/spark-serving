#!/bin/bash
# Start the model router (:8888). Safe to re-run: checks, then brings the router up to date.
#   ./install.sh
set -euo pipefail
cd "$(dirname "$0")/model-router"
die() { echo "install: $*" >&2; exit 1; }

for c in docker git curl; do command -v "$c" >/dev/null || die "$c missing"; done
docker info >/dev/null 2>&1 || die "docker not usable by $(id -un) (docker group?)"
docker compose version >/dev/null 2>&1 || die "docker compose plugin missing"
docker info --format '{{json .Runtimes}}' | grep -q nvidia || [ -d /etc/cdi ] || [ -d /var/run/cdi ] \
  || die "no NVIDIA container runtime (nvidia-container-toolkit)"
[ "$(uname -m)" = aarch64 ] || echo "install: warning: not aarch64; the recipes target DGX Spark (GB10)" >&2

export PROMPT_CACHE_DIR="${PROMPT_CACHE_DIR:-$HOME/.cache/tensorfold-prompt-cache}"
mkdir -p "$PROMPT_CACHE_DIR" state
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d --build

for i in $(seq 1 60); do curl -sf localhost:8888/health >/dev/null && break; sleep 1; done
curl -sf localhost:8888/health >/dev/null || die "router not answering; docker logs model-router"
curl -s 'localhost:8888/router/installs?refresh=1' \
  | python3 -c 'import json, sys
for n, m in json.load(sys.stdin)["models"].items():
    print("  " + n + ": " + ("installed" if m["installed"] else "not installed"))' || true
cat <<'EOF'

Router up: http://localhost:8888 (UI) and /v1 (API).
Next: install a model, e.g.
  curl -X POST localhost:8888/router/install -d '{"model":"35b-tf"}'
EOF
