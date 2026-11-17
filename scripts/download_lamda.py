"""Download helper for the LAMDA dataset from Hugging Face.

LAMDA (Haque et al., 2025) provides longitudinal Android malware samples
with 925 features (var_thresh_0.01) across 2013-2025.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


BASE_URL = "https://huggingface.co/datasets/IQSeC-Lab/LAMDA/resolve/main/var_thresh_0.01"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"Already exists: {dest} ({dest.stat().st_size} bytes)")
        return
    print(f"Downloading {url} -> {dest} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"Saved: {dest} ({dest.stat().st_size} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download LAMDA dataset files")
    parser.add_argument("--out-dir", default="data/lamda", help="Destination folder")
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2013", "2014", "2016"],
        help="Years to download (e.g. 2013 2014 2016)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=["test", "train"],
        help="Splits to download (test is ~2.5MB per year, train is ~8MB per year)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download feature mapping
    mapping_url = f"{BASE_URL}/feature_mapping.csv"
    download_file(mapping_url, out_dir / "feature_mapping.csv")

    for year in args.years:
        for split in args.splits:
            filename = f"{year}_{split}.parquet"
            url = f"{BASE_URL}/{year}/{filename}"
            dest = out_dir / filename
            download_file(url, dest)

    print("\nAll requested LAMDA files downloaded successfully!")
    print("\nverifying against data/expected.json:")
    return 0 if verify(out_dir) else 1


def _expected_lamda() -> dict:
    spec = json.loads((Path(__file__).resolve().parents[1] / "data" / "expected.json")
                      .read_text(encoding="utf-8"))
    return {Path(e["path"]).name: e for e in spec["files"]
            if e["path"].startswith("lamda/")}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(out_dir: Path) -> bool:
    ok = True
    for name, entry in sorted(_expected_lamda().items()):
        path = out_dir / name
        if not path.exists():
            print(f"MISSING  {name}")
            ok = False
            continue
        size, digest = path.stat().st_size, _sha256(path)
        if size != entry["bytes"] or digest != entry["sha256"]:
            print(f"MISMATCH {name}: {digest} ({size:,} bytes); "
                  f"expected {entry['sha256']} ({entry['bytes']:,} bytes)")
            ok = False
        else:
            print(f"OK       {name} ({size:,} bytes)")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
