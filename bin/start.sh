#!/usr/bin/env bash
# Start the Knowledge Graphs stack (app + neo4j + ollama) with hot reload.
set -euo pipefail

# Resolve the project root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Compose v2 is invoked as `docker compose` (a plugin). Fall back to the
# legacy `docker-compose` standalone binary if that is all that is installed.
if docker compose version >/dev/null 2>&1; then
  DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  DC=(docker-compose)
else
  echo "Error: neither 'docker compose' nor 'docker-compose' was found." >&2
  exit 1
fi

# Bootstrap a local .env (gitignored) from the committed example on first run,
# so the app container picks up its configuration without editing tracked files.
if [[ ! -f .env ]]; then
  echo ">> No .env found; seeding it from config_example.env"
  cp config_example.env .env
fi

echo ">> Building images (if needed) and starting services..."
"${DC[@]}" up -d --build

echo
echo ">> Stack is up. Endpoints:"
echo "   Streamlit app : http://localhost:8080"
echo "   Neo4j Browser : http://localhost:7474   (neo4j / password123)"
echo "   Ollama        : http://localhost:11434"
echo
echo "   Hot reload is ON — edit files under src/ pgs/ app.py and Streamlit"
echo "   will reload automatically. Logs: ${DC[*]} logs -f app"
