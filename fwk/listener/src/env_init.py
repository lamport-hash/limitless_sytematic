import os
import yaml

# Read the starting port from port.conf
with open("config/port.conf", "r") as f:
    content = f.read().strip()
    start_port = int(content.split(":")[1].strip())

# Read the listeners from listeners.yaml
with open("config/listeners.yaml", "r") as f:
    listeners_data = yaml.safe_load(f)
    listeners = list(listeners_data["listeners"].keys())

# Set environment variables for each listener
for i, listener in enumerate(listeners):
    port = start_port + i
    var_name = f"{listener}_port"
    os.environ[var_name] = str(port)
    print(f"export {var_name}={port}")
    # Add port to listeners.yaml
    listeners_data["listeners"][listener]["port"] = port

# Write back to listeners.yaml
with open("config/listeners.yaml", "w") as f:
    yaml.dump(listeners_data, f, default_flow_style=False)
