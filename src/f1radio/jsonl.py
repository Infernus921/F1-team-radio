# Imports
import json
from pathlib import Path


# Functions
def resume_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    good, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                good.append((json.loads(line)["id"], line if line.endswith("\n") else line + "\n"))
            except (json.JSONDecodeError, KeyError):
                bad += 1
    if bad:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(line for _, line in good)
        print(f"{path.name}: dropped {bad} incomplete line(s)", flush=True)
    return {i for i, _ in good}
