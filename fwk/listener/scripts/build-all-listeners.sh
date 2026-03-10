#!/bin/bash

# Exit on any error
set -e

# Project root (adjust if your script is called from different dir)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
GENERATED_DIR="$PROJECT_ROOT/generated-composes"

echo "🚀 Starting all listeners..."

# Ensure generated directory exists
mkdir -p "$GENERATED_DIR"

# Generate compose files if not already present (idempotent)
python3 "$PROJECT_ROOT/src/utils_listener_docker.py" generate 

# Find all generated compose files and start them
for compose_file in "$GENERATED_DIR"/docker-compose-*.yml; do
    if [[ -f "$compose_file" ]]; then
        echo "👉 Starting: $(basename "$compose_file")"
        docker compose -f "$compose_file" build
    fi
done

echo "✅ All docker listeners build!"
