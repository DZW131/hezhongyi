"""Fetch the verified project checkpoint set over SSH and verify SHA-256."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="10.15.20.178")
    parser.add_argument("--user", default="duyanhong")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--output-dir", type=Path, default=root / "checkpoints")
    parser.add_argument("--manifest", type=Path, default=root / "configs" / "model_weights.json")
    parser.add_argument("--only", nargs="*", help="Optional exact checkpoint filenames")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.only or [])
    known = {item["filename"] for item in manifest["weights"]}
    if selected - known:
        parser.error("Unknown filenames: " + ", ".join(sorted(selected - known)))
    for item in manifest["weights"]:
        if selected and item["filename"] not in selected:
            continue
        path = args.output_dir / item["filename"]
        if path.exists() and path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"]:
            print("Verified existing " + path.name, flush=True)
            continue
        partial = path.with_suffix(path.suffix + ".partial")
        command = ["scp", "-P", str(args.port), "-o", "ConnectTimeout=10"]
        if args.key:
            command += ["-i", str(args.key)]
        command += [f'{args.user}@{args.host}:{item["server_path"]}', str(partial)]
        subprocess.run(command, check=True)
        if partial.stat().st_size != item["bytes"] or sha256(partial) != item["sha256"]:
            raise RuntimeError("Downloaded file failed size/SHA-256 validation: " + path.name)
        partial.replace(path)
        print("Downloaded and verified " + path.name, flush=True)
    print("Selected project weights are ready.")


if __name__ == "__main__":
    main()
