#!/usr/bin/env python3
"""Clear outputs from notebooks under notebooks/ (keeps source only for git)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GLOB = "notebooks/**/*.ipynb"


def strip_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text())
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True
    if changed:
        path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n")
    return changed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Notebook paths (default: all under notebooks/).",
    )
    args = p.parse_args()
    paths = args.paths or sorted(REPO_ROOT.glob(DEFAULT_GLOB))
    n = 0
    for path in paths:
        if strip_notebook(path):
            print(f"stripped: {path}")
            n += 1
        else:
            print(f"clean:    {path}")
    print(f"{n} notebook(s) updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
