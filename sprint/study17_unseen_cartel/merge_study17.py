"""Joint report of Study 17 (local, CPU): Study 15's S1-S4, Study 16's W1 and Study 17's X1.

Unpacks the small zips into one folder and runs the registered report code (study17_xj.py, the same report
functions as X1) over their union: Study 15's and Study 16's decisions recomputed unchanged, then Study 17's.

Usage: python merge_study17.py <out_folder> study15_s1_small.zip study15_s2_small.zip study15_s3_small.zip \
           study15_s4_small.zip study16_w1_small.zip study17_x1_small.zip
"""
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED_OK = (".complete",)


def shared(n: str) -> bool:
    """Files every run writes (completion markers, engine notes, per-run reports): may repeat across zips."""
    base = n.split("/")[-1]
    return (base.endswith(SHARED_OK) or base.startswith("engine_") or base.startswith(("report15_", "report16_", "report17_"))
            or base == "session_deadline.json" or base == "big_judge_tried.json")


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
                if n in seen and not shared(n):
                    raise SystemExit(f"{n} is in both {seen[n]} and {z}")
                seen.setdefault(n, z)
                if n.split("/")[-1].startswith(("report15_", "report16_", "report17_")):
                    continue                       # the per-run partial reports stay in their zips
                (out / n).parent.mkdir(parents=True, exist_ok=True)
                (out / n).write_bytes(zf.read(n))
    print(f"unpacked {len(seen)} files from {len(zips)} zips into {out}")
    r = subprocess.run([sys.executable, str(HERE / "study17_xj.py"), "report", "--root", str(out)])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
