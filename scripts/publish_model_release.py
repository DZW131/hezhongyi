"""Upload explicitly selected model files to a GitHub release using Git credentials."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="DZW131/hezhongyi")
    parser.add_argument("--tag", default="hspn-local-weights-20261020")
    parser.add_argument("--branch", default="mescnn-transfer")
    parser.add_argument("--file", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        result = subprocess.run(
            ["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
            text=True, capture_output=True, check=True, timeout=30,
        )
        fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        token = fields.get("password")
    if not token:
        raise SystemExit("No GitHub credential available")
    headers = {"Authorization": "Bearer " + token, "User-Agent": "hspn-model-delivery"}
    base = "https://api.github.com/repos/" + args.repo

    def request(url, data=None, method=None, extra_headers=None):
        req = urllib.request.Request(url, data=data, method=method, headers={**headers, **(extra_headers or {})})
        with urllib.request.urlopen(req, timeout=240) as response:
            return json.load(response)

    def json_request(url, payload, method="POST"):
        return request(url, json.dumps(payload).encode(), method, {"Content-Type": "application/json"})

    releases = request(base + "/releases?per_page=100")
    release = next((item for item in releases if item["tag_name"] == args.tag), None)
    if release is None:
        release = json_request(base + "/releases", {
            "tag_name": args.tag, "target_commitish": args.branch, "draft": True,
            "name": "HSPN local model weights 2026-10-20",
            "body": "Verified local weights. The glomerulus U-Net is hospital-finetuned; MESCNN M/E/S/C are original transfer/initialization baselines. Hospital-finetuned lesion and cell checkpoints remain pending server recovery. See docs/MODEL_WEIGHTS.md and weights_manifest.json.",
        })
    existing = {asset["name"]: asset for asset in request(base + f'/releases/{release["id"]}/assets')}
    upload_base = release["upload_url"].split("{")[0]

    def upload(path):
        size = path.stat().st_size
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        asset = existing.get(path.name)
        if asset and asset["size"] != size:
            raise RuntimeError("Existing release asset differs in size: " + path.name)
        if not asset:
            print("Uploading " + path.name + " (" + str(size) + " bytes)", flush=True)
            with path.open("rb") as handle:
                asset = request(upload_base + "?name=" + urllib.parse.quote(path.name), handle, "POST", {
                    "Content-Type": "application/octet-stream", "Content-Length": str(size),
                })
        print("Uploaded " + path.name, flush=True)
        return {"filename": path.name, "bytes": size, "sha256": digest,
                "url": asset["browser_download_url"], "status": "uploaded"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        assets = list(pool.map(upload, args.file))
    manifest = {"release_tag": args.tag, "repository": args.repo, "assets": assets,
                "pending": ["hospital-finetuned M/E", "crescent detector", "crescent ROI classifier",
                            "tubular atrophy U-Net", "DPA-P2PNet 80-epoch best", "HFRM-CAM"]}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    upload(args.manifest)
    if args.publish:
        release = json_request(base + f'/releases/{release["id"]}', {"draft": False, "target_commitish": args.branch}, "PATCH")
    print(json.dumps({"release_id": release["id"], "draft": release["draft"], "url": release["html_url"]}), flush=True)


if __name__ == "__main__":
    main()
