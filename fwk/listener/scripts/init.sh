source< python3#!/bin/bash

echo "Setting up environment variables..."
source <(python3 env_init.py)

echo "Launching Redis..."
./redis.sh

echo "Launching Manager..."
./app.sh

echo "Writing docker compose files"
python3 utils_listener_docker.py generate

echo "Launching docker compose files"
build-all-listeners.sh
