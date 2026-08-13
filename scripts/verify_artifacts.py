#!/usr/bin/env python3
"""Verify downloaded model artifacts against configs/artifact-manifest.json."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "configs" / "artifact-manifest.json"


def load_set(name, manifest=DEFAULT_MANIFEST):
    data = json.loads(Path(manifest).read_text())
    return data["sets"][name]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(name, directory, selected=()):
    artifact_set = load_set(name)
    wanted = set(selected)
    records = [r for r in artifact_set["files"] if not wanted or r["path"] in wanted]
    missing_from_manifest = wanted - {r["path"] for r in records}
    if missing_from_manifest:
        raise ValueError(
            f"not recorded in {name} manifest: {', '.join(sorted(missing_from_manifest))}"
        )

    failures = []
    for record in records:
        path = Path(directory) / record["path"]
        if not path.is_file():
            failures.append(f"{record['path']}: missing")
            continue
        size = path.stat().st_size
        if size != record["size_bytes"]:
            failures.append(
                f"{record['path']}: size {size}, expected {record['size_bytes']}"
            )
            continue
        actual = sha256(path)
        if actual != record["sha256"]:
            failures.append(
                f"{record['path']}: sha256 {actual}, expected {record['sha256']}"
            )
            continue
        print(f"verified {record['path']}: {size} bytes, sha256 {actual}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_set", choices=("bf16", "gguf"))
    parser.add_argument("directory")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    try:
        failures = verify(args.artifact_set, args.directory, args.files)
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"artifact manifest error: {exc}")
    if failures:
        sys.exit("artifact verification failed:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
