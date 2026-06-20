#!/usr/bin/env bash
# Stop the Knowledge Graphs stack.
#   ./bin/stop.sh        stop containers, keep named volumes (data preserved)
#   ./bin/stop.sh -v     also remove neo4j_data and ollama_models volumes (data wiped)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if docker compose version >/dev/null 2>&1; then
  DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  DC=(docker-compose)
else
  echo "Error: neither 'docker compose' nor 'docker-compose' was found." >&2
  exit 1
fi

DOWN_ARGS=("$@")
if [[ "${1:-}" == "-v" ]] || [[ "${1:-}" == "--volumes" ]]; then
  echo ">> Stopping services AND deleting data volumes (neo4j_data, ollama_models)..."
else
  echo ">> Stopping services (named volumes preserved — data is kept)."
  echo "   Run with -v to also wipe data volumes."
fi

"${DC[@]}" down "${DOWN_ARGS[@]}"
echo ">> Stack stopped."
