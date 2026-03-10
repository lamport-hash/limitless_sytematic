import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.main_hist import app

print("Registered routes:")
for route in app.routes:
    print(f"  {route.path} -> {route.methods}")
