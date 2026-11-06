"""python3 -m anelab point FAMILY FUNCTIONS UNITS [--rows R,R,..] [--load]
python3 -m anelab frontier FAMILY --functions N,N,.. [--max-units U] [--tag TAG]
python3 -m anelab report [RESULTS.jsonl ...]"""

import argparse
import sys

from . import families, probe, report


def point(family, functions, units, rows, load, tag):
    rows = rows or families.evenly(functions)
    src, blob = families.FAMILIES[family](functions, units, rows)
    result = probe.compile_program(src, blob, load=load)
    entry = {"tag": tag, "family": family, "functions": functions, "units": units, "channels": units * 512,
             "rows": rows, "rows_total": sum(rows), **result}
    probe.record(entry)
    print(f"{family:<9} functions {functions:>2} units {units:>2} rows {rows[0]}..{rows[-1]} ({sum(rows)}): "
          f"{'ok' if result['ok'] else 'FAIL ' + result['stage'] + ' ' + probe.short_error(result)} "
          f"{result.get('compile_ms', '')} ms", flush=True)
    return result["ok"]


def frontier(family, counts, max_units, tag, load):
    for n in counts:
        if not point(family, n, 1, None, load, tag):
            print(f"{family} functions {n}: fails at 1 unit", flush=True)
            continue
        if point(family, n, max_units, None, load, tag):
            print(f"{family} functions {n}: compiles up to {max_units} units and maybe more", flush=True)
            continue
        lo, hi = 1, max_units
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if point(family, n, mid, None, load, tag) else (lo, mid)
        print(f"{family} functions {n}: ok at {lo} units, fails at {hi}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("point")
    p.add_argument("family", choices=families.FAMILIES)
    p.add_argument("functions", type=int)
    p.add_argument("units", type=int)
    p.add_argument("--rows", type=lambda s: [int(x) for x in s.split(",")])
    p.add_argument("--load", action="store_true")
    p.add_argument("--tag", default="point")
    f = sub.add_parser("frontier")
    f.add_argument("family", choices=families.FAMILIES)
    f.add_argument("--functions", type=lambda s: [int(x) for x in s.split(",")], required=True)
    f.add_argument("--max-units", type=int, default=64)
    f.add_argument("--load", action="store_true")
    f.add_argument("--tag", default="frontier")
    r = sub.add_parser("report")
    r.add_argument("files", nargs="*")
    a = ap.parse_args()
    if a.cmd == "point":
        sys.exit(0 if point(a.family, a.functions, a.units, a.rows, a.load, a.tag) else 1)
    if a.cmd == "frontier":
        frontier(a.family, a.functions, a.max_units, a.tag, a.load)
    else:
        report.main(a.files)


main()
