#!/usr/bin/env python3
"""Check the three pre-registrations of this repository (standard library only).

  python verify_preregistration.py

1. Recomputes each study's predictions hash from its prereg_*.json (SHA-256 over the predictions, the primary
   family and alpha; first 16 hex digits) and compares it with the hash stated in its PREREG_*.md.
2. Hashes every code file named in a PREREG_*.md table and compares it with the registered file SHA-256 and with
   the registered "code hash as one Kaggle cell" (the same file without its trailing newline, as Kaggle stores it).
3. If data/ holds the runs' output zips: reads the registration each run stored when it started and checks its
   predictions hash and code hash.
4. In a git clone with full history: checks that each registration file is byte-identical to the version committed
   to GitHub before the study's first run, and that preexisting/ matches the commit of 18 October 2026.
Exits with status 1 if any check fails.
"""
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IST = timezone(timedelta(hours=3))
REPO_URL = "https://github.com/Ralolooaf/cot-hint-span-causality"

STUDIES = {
    "15": {"dir": "preexisting/study15_cartel_organisms", "md": "PREREG_STUDY15_SPRINT.md",
           "json": "prereg_study15_sprint.json",
           "runs": {"S1": "study15_s1_small.zip", "S2": "study15_s2_small.zip", "S3": "study15_s3_small.zip",
                    "S4": "study15_s4_small.zip"},
           # (commit, path at the time, path now); 1bfe962 holds the final text, pushed after the runs
           "commits": [("1bfe962abbf8adb69bfc020a79b3292bddbdba1b", "update3/PREREG_STUDY15_SPRINT.md", "PREREG_STUDY15_SPRINT.md"),
                       ("1bfe962abbf8adb69bfc020a79b3292bddbdba1b", "update3/prereg_study15_sprint.json", "prereg_study15_sprint.json")],
           "pre_run_commit": ("7d407f443173569f23b205776b3416d5d5df19d3", "update3/prereg_study15_sprint.json")},
    "16": {"dir": "sprint/study16_natural_colluders", "md": "PREREG_STUDY16.md", "json": "prereg_study16.json",
           "runs": {"W1": "study16_w1_small.zip"},
           "commits": [("1bfe962abbf8adb69bfc020a79b3292bddbdba1b", "update3/PREREG_STUDY16.md", "PREREG_STUDY16.md")]},
    "17": {"dir": "sprint/study17_unseen_cartel", "md": "PREREG_STUDY17.md", "json": "prereg_study17.json",
           "runs": {"X1": "study17_x1_small.zip"},
           "commits": [("2e108925323323e8f65855c6accb22d4377fd4d2", "update3/PREREG_STUDY17.md", "PREREG_STUDY17.md"),
                       ("2e108925323323e8f65855c6accb22d4377fd4d2", "update3/prereg_study17.json", "prereg_study17.json")]},
}
PREEXISTING_COMMIT = "e4373075d10aff11677daeebfa1304e3117f4834"      # 18 Oct 2026, Studies 1-14
PREEXISTING_MAP = [("update1/", "preexisting/update1/"), ("update2/", "preexisting/update2/"),
                   ("README.md", "preexisting/studies01_02/README.md"),
                   ("span_knockout_harness.py", "preexisting/studies01_02/span_knockout_harness.py"),
                   ("results/", "preexisting/studies01_02/results/")]

failures = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        failures.append(msg)


def sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


def predictions_hash(d):
    blob = json.dumps({"predictions": d["predictions"], "primary": d["primary_family"], "alpha": d["family_alpha"]},
                      sort_keys=True)
    return sha16(blob.encode())


def ist(stamp):
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            t = datetime.strptime(stamp, fmt)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            return t.astimezone(IST)
        except (TypeError, ValueError):
            pass
    return None


def git(*args):
    try:
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)
    except FileNotFoundError:
        return None
    return r.stdout if r.returncode == 0 else None


def main():
    have_git = git("rev-parse", "--git-dir") is not None
    run_starts = {}
    for sid, S in STUDIES.items():
        d = ROOT / S["dir"]
        print(f"\nStudy {sid}  ({S['dir']})")
        md = (d / S["md"]).read_text(encoding="utf-8")
        reg = json.loads((d / S["json"]).read_text(encoding="utf-8"))
        m = re.search(r"\*\*Predictions hash\*\*[^`]*`([0-9a-f]{16})`", md)
        md_hash = m.group(1) if m else None
        h = predictions_hash(reg)
        check(h == reg.get("predictions_sha256_16"), f"{S['json']}: recomputed predictions hash {h}")
        check(h == md_hash, f"{S['md']} states the same hash ({md_hash})")
        # code files from the registration table: | run | what | `file` | `cell hash` | `file sha…` |
        cell_hash = {}
        for row in re.finditer(r"^\| *(\w+) *\|[^|\n]*\| *`([\w.]+\.py)` *\| *`([0-9a-f]{16})` *\| *`([0-9a-f]{16})…?` *\|",
                               md, flags=re.M):
            run, fname, want_cell, want_file = row.groups()
            cell_hash[run] = want_cell
            f = d / fname
            if not f.exists():
                check(False, f"{fname} is missing")
                continue
            b = f.read_bytes()
            got_file, got_cell = sha16(b), sha16(b.rstrip(b"\n"))
            check(got_file == want_file and got_cell == want_cell,
                  f"{fname}: file sha256 {got_file}, Kaggle-cell hash {got_cell} (run {run})")
        # registrations stored by the runs themselves
        for run, zname in S["runs"].items():
            z = ROOT / "data" / zname
            if not z.exists():
                print(f"  skip  data/{zname} not present")
                continue
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist() if re.fullmatch(rf"prereg_study{sid}_{run}\.json", n.split("/")[-1])]
                if not names:
                    check(False, f"{zname}: no prereg_study{sid}_{run}.json")
                    continue
                r = json.loads(zf.read(names[0]))
            t = ist(r.get("written_at"))
            run_starts[(sid, run)] = t
            check(r.get("predictions_sha256_16") == h and r.get("code_sha256_16") == cell_hash.get(run),
                  f"run {run} registered predictions {r.get('predictions_sha256_16')} and code "
                  f"{r.get('code_sha256_16')} at start, {t:%d %b %Y %H:%M:%S} (+03:00)" if t else f"run {run}")
        # registration files against the commits made before the runs
        if not have_git:
            print(f"  skip  not a git clone; clone {REPO_URL} to check the registration commits")
            continue
        for commit, old, now in S["commits"]:
            blob = git("show", f"{commit}:{old}")
            when = (git("show", "-s", "--format=%ci", commit) or b"?").decode().strip()
            if blob is None:
                check(False, f"commit {commit[:7]} or {old} not found (shallow clone? run: git fetch --unshallow)")
                continue
            check(blob == (d / now).read_bytes(), f"{now} is byte-identical to {old} in commit {commit[:7]} ({when})")
        if "pre_run_commit" in S:
            commit, old = S["pre_run_commit"]
            blob = git("show", f"{commit}:{old}")
            when = (git("show", "-s", "--format=%ci", commit) or b"?").decode().strip()
            if blob is not None:
                first = json.loads(blob)
                print(f"  info  version pushed before the runs, commit {commit[:7]} ({when}): predictions hash "
                      f"{predictions_hash(first)}")
                same = first["primary_family"] == reg["primary_family"] and first["family_alpha"] == reg["family_alpha"]
                changed = sorted(k for k in set(first["predictions"]) | set(reg["predictions"])
                                 if first["predictions"].get(k) != reg["predictions"].get(k))
                print(f"  info  primary family and alpha unchanged: {same}; predictions that differ: {changed}")
                for k in changed:
                    a, b = first["predictions"].get(k, "").split(), reg["predictions"].get(k, "").split()
                    gone = [w for w in a if w not in b]
                    new = [w for w in b if w not in a]
                    print(f"  info  {k}: words removed {gone}, words added {new}")
    # run order: every study's first run after its registration commit
    if have_git:
        print("\nRegistration before data")
        order = [("15", "7d407f443173569f23b205776b3416d5d5df19d3"), ("16", "1bfe962abbf8adb69bfc020a79b3292bddbdba1b"),
                 ("17", "2e108925323323e8f65855c6accb22d4377fd4d2")]
        for sid, commit in order:
            out = git("show", "-s", "--format=%cI", commit)
            starts = [t for (s, _), t in run_starts.items() if s == sid and t]
            if out is None or not starts:
                print(f"  skip  Study {sid}: commit or run outputs not available")
                continue
            c = datetime.fromisoformat(out.decode().strip()).astimezone(IST)
            first = min(starts)
            check(c < first, f"Study {sid}: commit {commit[:7]} at {c:%d %b %H:%M:%S}, first run at {first:%d %b %H:%M:%S}")
        print("\nPreexisting files against the commit of 18 October 2026")
        listing = git("ls-tree", "-r", "--name-only", PREEXISTING_COMMIT)
        if listing is None:
            print("  skip  commit not available")
        else:
            n_ok, bad = 0, []
            for name in listing.decode().splitlines():
                for old, new in PREEXISTING_MAP:
                    if name == old or (old.endswith("/") and name.startswith(old)):
                        target = ROOT / (new + name[len(old):] if old.endswith("/") else new)
                        blob = git("show", f"{PREEXISTING_COMMIT}:{name}")
                        if target.exists() and blob == target.read_bytes():
                            n_ok += 1
                        else:
                            bad.append(name)
            check(not bad, f"{n_ok} files under preexisting/ are byte-identical to commit {PREEXISTING_COMMIT[:7]}"
                  + (f"; differ or missing: {bad[:5]}" if bad else ""))
    print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
