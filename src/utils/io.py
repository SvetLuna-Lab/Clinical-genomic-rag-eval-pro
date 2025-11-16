import os, json
from typing import Any, Dict, List

def read_json(path: str):
    with open(path,"r",encoding="utf-8") as f: return json.load(f)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path,"w",encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
