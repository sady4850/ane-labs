"""Frontier tables from results/*.jsonl: per machine and family, the widest program of each function count that
compiled and the narrowest that failed."""

import collections
import glob
import json
from pathlib import Path

from .probe import ROOT


def main(files=None):
    files = files or sorted(glob.glob(str(ROOT / "results" / "*.jsonl")))
    groups = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, None]))
    for file in files:
        for line in open(file):
            e = json.loads(line)
            if e.get("tag") == "point":
                continue
            key = (e["chip"], e["macos"], e["family"])
            cell = groups[key][e["functions"]]
            if e["ok"]:
                cell[0] = max(cell[0], e["units"])
            elif cell[1] is None or e["units"] < cell[1]:
                cell[1] = e["units"]
    for (chip, macos, family), cells in sorted(groups.items()):
        print(f"\n{chip}, macOS {macos}, {family} (units of 512 ANE channels)\n")
        print("| functions | widest ok | narrowest fail | functions x widest ok |")
        print("|---:|---:|---:|---:|")
        for n in sorted(cells):
            ok, bad = cells[n]
            print(f"| {n} | {ok or '-'} | {bad if bad is not None else '-'} | {n * ok if ok else '-'} |")


if __name__ == "__main__":
    main()
