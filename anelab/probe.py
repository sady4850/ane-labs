import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "build" / "ane-compile"


def machine():
    def sh(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

    return {"chip": sh("sysctl", "-n", "machdep.cpu.brand_string"), "macos": platform.mac_ver()[0],
            "build": sh("sw_vers", "-buildVersion"), "memory_gb": int(sh("sysctl", "-n", "hw.memsize")) >> 30}


def compile_program(source, blob=b"", load=False, timeout=900):
    if not TOOL.exists():
        raise SystemExit("build/ane-compile is missing, run make first")
    key = hashlib.sha256(source.encode() + blob).hexdigest()[:16]
    work = Path(tempfile.gettempdir()) / "ane-labs" / key
    work.mkdir(parents=True, exist_ok=True)
    try:
        (work / "model.mil").write_text(source)
        (work / "weights.bin").write_bytes(blob or bytes(128))
        started = time.time()
        try:
            p = subprocess.run([str(TOOL), str(work), *(["--load"] if load else [])], capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "stage": "timeout", "wall_s": round(time.time() - started, 1)}
        lines = p.stdout.strip().splitlines()
        result = json.loads(lines[-1]) if lines else {"ok": False, "stage": "crash", "error": p.stderr[-500:]}
        result["wall_s"] = round(time.time() - started, 1)
        result["mil_bytes"] = len(source)
        result["blob_bytes"] = len(blob)
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def results_path():
    m = machine()
    name = f"{m['chip'].replace('Apple ', '').replace(' ', '-')}_macOS-{m['macos']}.jsonl"
    return ROOT / "results" / name


def record(entry, path=None):
    path = Path(path or results_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({**machine(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}) + "\n")


def short_error(result):
    e = result.get("error", "")
    return e.split(" | ")[-1].strip()[:160] if e else ""
