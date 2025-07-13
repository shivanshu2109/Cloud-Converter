import yaml
import json
import os
from hashlib import sha256  

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def save_yaml(path, data):
    with open(path, 'w') as f:
        # Add indent=2 for nice formatting
        yaml.dump(data, f, indent=2, sort_keys=False)

# --- Reusable Functions ---
def load_cache(cache_path):
    if not os.path.exists(cache_path): return {}
    try:
        with open(cache_path, "r") as f: return json.load(f)
    except (json.JSONDecodeError, IOError): return {}

def save_cache(cache_path, data):
    with open(cache_path, "w") as f: json.dump(data, f, indent=2)

def get_cache_key(service_block):
    raw = json.dumps(service_block, sort_keys=True)
    return sha256(raw.encode()).hexdigest()