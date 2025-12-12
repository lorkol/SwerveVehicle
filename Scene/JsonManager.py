import json
from typing import Any, Dict


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)