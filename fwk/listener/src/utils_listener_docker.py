#!/usr/bin/env python3

import yaml
import os
import subprocess
import sys
from pathlib import Path

# Configuration file paths
LISTENERS_CONFIG_FILE       = "config/listeners.yaml"
LISTENERS_PROD_CONFIG_FILE  = "config/prod_listeners.yaml"
TEMPLATE_FILE               = "config/docker-compose-listeners-template.yaml"
OUTPUT_DIR                  = "generated-composes"

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(exist_ok=True)


def load_listeners_config(isprod: bool) -> dict:
    """Load listeners configuration — prod or dev variant."""
    if isprod:
        print('using prod config file ')
        config_file = LISTENERS_PROD_CONFIG_FILE 
    else :
        print('using std config file ')
        config_file = LISTENERS_CONFIG_FILE
    with open(config_file, 'r') as f:
        return yaml.safe_load(f) or {}


def load_template() -> str:
    """Load the docker-compose template."""
    with open(TEMPLATE_FILE, 'r') as f:
        return f.read()


def generate_compose_for_listener(listener_name: str, listener_config: dict) -> str:
    """
    Generate a docker-compose file for a specific listener using the template.
    """
    port = listener_config.get("port")
    if not port:
        raise ValueError(f"Listener '{listener_name}' has no 'port' defined.")

    template = load_template()
    composed_content = template.replace("${listener_name}", listener_name)
    composed_content = composed_content.replace("${listener_port}", str(port))

    return composed_content


def generate_compose_files(isprod: bool = False):
    """
    Generate a separate docker-compose file for each listener.
    """
    listeners = load_listeners_config(isprod).get("listeners", {})
    if not listeners:
        print("No listeners found in configuration file")
        return

    for listener_name, config in listeners.items():
        compose_content = generate_compose_for_listener(listener_name, config)
        filename = f"docker-compose-{listener_name}-auto.yml"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, 'w') as f:
            f.write(compose_content)
        print(f"✅ Generated: {filepath}")


def start_listener(listener_name: str, isprod: bool = False):
    """
    Start a specific listener using its generated docker-compose file.
    """
    filename = f"docker-compose-{listener_name}-auto.yml"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(filepath):
        print(f"⚠️  Compose file not found: {filepath}. Generating...")
        generate_compose_files(isprod)  # Auto-generate with correct config
        if not os.path.exists(filepath):
            print(f"❌ Failed to generate compose file for {listener_name}")
            return

    print(f"🚀 Starting listener: {listener_name} ({'production' if isprod else 'development'} mode)...")
    try:
        subprocess.run(["docker", "compose", "-f", filepath, "up", "-d"], check=True)
        print(f"✅ Listener '{listener_name}' started.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start '{listener_name}': {e}")
        sys.exit(1)


def stop_listener(listener_name: str):
    """
    Stop and remove a specific listener's containers.
    """
    filename = f"docker-compose-{listener_name}-auto.yml"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(filepath):
        print(f"❌ Compose file not found: {filepath}")
        return

    print(f"🛑 Stopping listener: {listener_name}...")
    try:
        subprocess.run(["docker", "compose", "-f", filepath, "down"], check=True)
        print(f"✅ Listener '{listener_name}' stopped and removed.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to stop '{listener_name}': {e}")
        sys.exit(1)


def list_alive_listeners():
    """
    List all running containers with names matching 'listener_*' and show their ports.
    """
    print("📋 Running Listeners:")
    print("-" * 60)

    try:
        result = subprocess.run([
            "docker", "ps", "--filter", "name=listener_", "--format",
            "table {{.Names}}\t{{.Ports}}"
        ], capture_output=True, text=True, check=True)

        output = result.stdout.strip()
        if not output:
            print("No running listeners found.")
            return

        lines = output.splitlines()
        if "NAMES" in lines[0] and "PORTS" in lines[0]:
            lines = lines[1:]

        for line in lines:
            parts = line.split()
            if not parts:
                continue
            name = parts[0]
            ports = " ".join(parts[1:]) if len(parts) > 1 else ""
            host_port = ports.split("->")[0].split(":")[-1] if "->" in ports else "N/A"

            display_name = name.replace("listener_", "", 1)
            print(f"{display_name:<20} → {host_port:>6}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to list containers: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage listener containers via docker-compose."
    )
    parser.add_argument(
        "--prod", "-p",
        action="store_true",
        help="Use production configuration (prod_listeners.yaml)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate all compose files
    subparsers.add_parser("generate", help="Generate docker-compose files for all listeners")

    # Start a listener
    start_parser = subparsers.add_parser("start", help="Start a specific listener")
    start_parser.add_argument("listener_name", type=str, help="Name of the listener to start")

    # Stop a listener
    stop_parser = subparsers.add_parser("stop", help="Stop a specific listener")
    stop_parser.add_argument("listener_name", type=str, help="Name of the listener to stop")

    # List running listeners
    subparsers.add_parser("list", help="List all running listeners with their ports")

    args = parser.parse_args()

    if args.prod:
        print('prod mode') # somehow this does not printy prod mode if i put -p

    if args.command == "generate":
        generate_compose_files(isprod=args.prod)
    elif args.command == "start":
        start_listener(args.listener_name, isprod=args.prod)
    elif args.command == "stop":
        stop_listener(args.listener_name)
    elif args.command == "list":
        list_alive_listeners()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()