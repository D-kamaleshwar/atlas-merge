"""Export utilities for experiment logs, Pareto candidates, and geometry profiles."""
import os
import json
import csv
from typing import Any, List, Dict
import pandas as pd
from .logging_utils import _to_serializable

def export_to_json(data: Any, filepath: str, indent: int = 2) -> str:
    """Export data structure to JSON file. Returns canonical file path."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    serializable = _to_serializable(data)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=indent)
    return os.path.abspath(filepath)


def export_to_csv(records: List[Dict[str, Any]], filepath: str) -> str:
    """Export a list of flat or semi-flat dictionary records to CSV. Returns canonical path."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not records:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return os.path.abspath(filepath)
    
    # Flatten any dicts in fields to strings for clean tabular CSV output
    flat_records = []
    for r in records:
        flat_r = {}
        for k, v in r.items():
            if isinstance(v, (dict, list)):
                flat_r[k] = json.dumps(_to_serializable(v))
            elif hasattr(v, "item"):
                flat_r[k] = v.item()
            else:
                flat_r[k] = v
        flat_records.append(flat_r)
    
    df = pd.DataFrame(flat_records)
    df.to_csv(filepath, index=False)
    return os.path.abspath(filepath)

