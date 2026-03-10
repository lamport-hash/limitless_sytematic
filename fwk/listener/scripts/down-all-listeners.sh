#!/bin/bash

# Exit on any error
set -e

# Project root (adjust if your script is called from different dir)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
GENERATED_DIR="$PROJECT_ROOT/generated-composes"

echo "🛑 Stopping all listeners..."

# Find all generated compose files and stop them
for compose_file in "$GENERATED_DIR"/docker-compose-*.yml; do
    if [[ -f "$compose_file" ]]; then
        echo "👇 Stopping: $(basename "$compose_file")"
        docker compose -f "$compose_file" down --remove-orphans
    fi
done

echo "✅ All listeners stopped and removed."
