"""Download helper for the KronoDroid dataset.

KronoDroid is hosted by the Canadian Institute for Cybersecurity (UNB):
https://www.unb.ca/cic/datasets/kronodroid.html

The dataset page asks for a short registration form, so the direct URL is not
fixed. Pass the link of the CSV you obtained from the project page:

    python scripts/download_kronodroid.py \
        --url "<direct csv url>" --out data/KronoDroid_real_device.csv

Bytes are streamed unchanged, so the SHA-256 of the result matches
``data/expected.json``. Do not round-trip the file through pandas: a re-export
re-serialises the floats and changes the digest without changing the content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED = HERE.parent / "data" / "expected.json"
UA = {"User-Agent": "Mozilla/5.0 (compatible; reproducibility-artifact)"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recorded_digest(name: str) -> str | None:
    if not EXPECTED.exists():
        return None
    spec = json.loads(EXPECTED.read_text(encoding="utf-8"))
    for entry in spec["files"]:
        if Path(entry["path"]).name == name:
            return entry["sha256"]
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download the KronoDroid CSV")
    p.add_argument("--url", required=True, help="direct URL of the KronoDroid CSV")
    p.add_argument("--out", default="data/KronoDroid_real_device.csv",
                   help="destination path (default: %(default)s)")
    p.add_argument("--force", action="store_true",
                   help="keep the file even if its SHA-256 does not match the recorded one")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.url} ...")
    req = urllib.request.Request(args.url, headers=UA)
    total = 0
    with urllib.request.urlopen(req) as resp, open(out, "wb") as fh:
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            fh.write(chunk)
            total += len(chunk)
    print(f"saved {total:,} bytes to {out}")

    want = recorded_digest(out.name)
    got = sha256_of(out)
    if want is None:
        print(f"sha256 {got} (no digest on record for {out.name})")
        return 0
    if got != want:
        print("sha256 mismatch:\n"
              f"  got      {got}\n"
              f"  expected {want}")
        if args.force:
            print("keeping the file because --force was given")
            return 0
        out.unlink()
        print("file removed: the URL is not the CSV used in the paper; "
              "see data/README.md")
        return 1
    print(f"sha256 {got} OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
