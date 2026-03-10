#!/usr/bin/env python3

import yaml
import os
import subprocess
import sys
from pathlib import Path


def load_activated_listeners():
    """Load listener names with activated: true from listeners.yaml"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "listeners.yaml"

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config file: {e}")
        sys.exit(1)

    activated = []
    for name, listener_config in config.get("listeners", {}).items():
        if listener_config.get("activated", False):
            activated.append(name)

    return activated


def start_activated_listeners():
    """Start docker compose files only for activated listeners"""
    project_root = Path(__file__).parent.parent
    generated_dir = project_root / "generated-composes"

    activated_listeners = load_activated_listeners()

    if not activated_listeners:
        print("❌ No activated listeners found in config/listeners.yaml")
        sys.exit(1)

    print(f"🚀 Starting {len(activated_listeners)} activated listeners...")

    started_count = 0
    for listener_name in activated_listeners:
        compose_file = generated_dir / f"docker-compose-{listener_name}-auto.yml"

        if compose_file.exists():
            print(f"👉 Starting: {compose_file.name}")
            try:
                subprocess.run(
                    ["docker", "compose", "-f", str(compose_file), "up", "-d"],
                    check=True,
                    capture_output=True,
                )
                started_count += 1
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to start {listener_name}: {e}")
                print(
                    f"   Error output: {e.stderr.decode().strip() if e.stderr else 'Unknown error'}"
                )
                sys.exit(1)
        else:
            print(f"⚠️  Compose file not found: {compose_file}")
            sys.exit(1)

    print(f"✅ {started_count} activated listeners started!")


if __name__ == "__main__":
    start_activated_listeners()
