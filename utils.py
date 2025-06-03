import json
import os
from config import STRICT_MODE_PAIRS

SIGNAL_FILE = "last_signals.json"

def load_previous_timestamps():
    if os.path.exists(SIGNAL_FILE):
        with open(SIGNAL_FILE, "r") as f:
            return json.load(f)
    return {}

def update_timestamp(symbol):
    data = load_previous_timestamps()
    from time import time
    data[symbol] = time()
    with open(SIGNAL_FILE, "w") as f:
        json.dump(data, f)

def is_strict_symbol(symbol):
    return symbol in STRICT_MODE_PAIRS
