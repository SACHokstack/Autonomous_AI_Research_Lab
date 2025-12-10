from pathlib import Path
import json
from typing import List, Dict, Any

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"

def load_all_runs() -> List[Dict[str, Any]]:
    runs = []
    for path in EXPERIMENTS_DIR.glob("run_*.json"):
        with path.open() as f:
            data = json.load(f)
        data["_path"] = str(path)
        runs.append(data)
    return runs
