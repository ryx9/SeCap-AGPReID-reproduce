#!/usr/bin/env python3
"""Extract the exact CARGO best-checkpoint artifacts required by demo/app.py."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "artifacts" / "archives" / "training_artifacts.zip"
DESTINATION = Path(__file__).resolve().parent / "artifacts" / "cargo"
MEMBERS = {
    "checkpoints/CARGO/model_best.pth": "model_best.pth",
    "CARGO/config.yaml": "config.yaml",
}
EXPECTED_SHA256 = "9e449d6dfd1274cfb47c4987dd3632c0b6fecb89b7912a4c83ced6d180c8e32e"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Replace existing artifacts.")
    args = parser.parse_args()
    if not ARCHIVE.is_file():
        raise SystemExit(f"Archive not found: {ARCHIVE}")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for member, name in MEMBERS.items():
            target = DESTINATION / name
            if target.exists() and not args.force:
                print(f"Kept {target.relative_to(DESTINATION.parent)}")
                continue
            with archive.open(member) as source, target.open("wb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
            print(f"Extracted {target.relative_to(DESTINATION.parent)}")
    digest_builder = hashlib.sha256()
    with (DESTINATION / "model_best.pth").open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit("Checkpoint checksum mismatch; remove artifacts and extract again.")
    print("Verified CARGO model_best.pth SHA-256.")


if __name__ == "__main__":
    main()
