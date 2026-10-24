"""Joint confirmatory report of the Study 15 sprint (local, CPU).

Unpacks the small zips of S1-S4 into one folder and runs the registered report code (study15_sj.py, the same
report functions as S1-S4) over their union.

Usage: python merge_sprint.py <out_folder> study15_s1_small.zip study15_s2_small.zip ... [--force]
"""
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out, zips = Path(args[0]), [Path(z) for z in args[1:]]
    out.mkdir(parents=True, exist_ok=True)
    seen = {}
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith("/"):
                    continue
                # files every run writes (P1 leftovers, completion markers) may repeat; data files may not
                if n in seen and not (n.endswith(".complete") or n.startswith("engine_") or n.endswith("_P1.jsonl")
                                      or n.startswith("prereg_study15_P1") or n == "session_deadline.json"
                                      or n.startswith("report15_")):
                    raise SystemExit(f"{n} is in both {seen[n]} and {z}")
                seen.setdefault(n, z)
                if n.startswith("report15_"):
                    continue                       # the per-run partial reports stay in their zips
                (out / n).parent.mkdir(parents=True, exist_ok=True)
                (out / n).write_bytes(zf.read(n))
    print(f"unpacked {len(seen)} files from {len(zips)} zips into {out}")
    r = subprocess.run([sys.executable, str(HERE / "study15_sj.py"), "report", "--root", str(out)])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
