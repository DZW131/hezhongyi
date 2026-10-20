"""Run tubular-atrophy U-Net inference on whole tissue scenes.

The output CSV is designed for ``aggregate_hspn_inference_for_scoring.py``.
It reports the predicted atrophic-tubule target area as a fraction of tissue
area for each scene.  The same value can be mapped to the scorer's T input by
slide ID, while retaining its model source in ``source_notes``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unet import UNet
from utils.checkpoint_io import load_torch_state

# WSI scenes legitimately exceed Pillow's generic decompression-bomb limit.
Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, default=None, help="Optional tile manifest used to select a split.")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-evidence", type=int, default=6)
    parser.add_argument("--limit-scenes", default="", help="Comma-separated scene IDs for a focused run.")
    return parser.parse_args()


def scene_split_map(manifest_csv: Path | None) -> Dict[str, str]:
    if manifest_csv is None:
        return {}
    with manifest_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        return {row["slide_id"]: row.get("split", "") for row in csv.DictReader(handle) if row.get("slide_id")}


def scene_paths(
    scenes_dir: Path,
    split_map: Dict[str, str],
    split: str,
    limit_scenes: str,
) -> Iterable[Tuple[Path, str, str]]:
    extensions = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
    paths = sorted({path for pattern in extensions for path in scenes_dir.glob(pattern)})
    requested = {value.strip() for value in limit_scenes.split(",") if value.strip()}
    for path in paths:
        scene_id = path.stem
        if scene_id not in split_map and "_s" in scene_id:
            prefix, suffix = scene_id.rsplit("_s", 1)
            matches = [
                expected
                for expected in split_map
                if expected.startswith(prefix.split("_", 2)[0] + "_")
                and expected.endswith("_s" + suffix)
                and prefix.startswith(expected.rsplit("_s", 1)[0])
            ]
            if len(matches) == 1:
                scene_id = matches[0]
        scene_split = split_map.get(scene_id, "")
        if split != "all" and scene_split != split:
            continue
        if requested and scene_id not in requested and path.stem not in requested:
            continue
        yield path, scene_id, scene_split


def tissue_mask(image: np.ndarray) -> np.ndarray:
    """Exclude near-white slide background while preserving stained tissue."""
    return image.mean(axis=2) < 245


def tile_positions(length: int, tile_size: int) -> List[int]:
    return list(range(0, length, tile_size))


def padded_tile(image: Image.Image, x: int, y: int, tile_size: int) -> Tuple[np.ndarray, int, int]:
    right = min(x + tile_size, image.width)
    bottom = min(y + tile_size, image.height)
    tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
    tile[: bottom - y, : right - x] = np.asarray(image.crop((x, y, right, bottom)).convert("RGB"))
    return tile, right - x, bottom - y


def scene_prediction(model: UNet, image: Image.Image, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    height, width = image.height, image.width
    target = np.zeros((height, width), dtype=bool)
    pending: List[Tuple[int, int, int, int, np.ndarray]] = []

    def flush() -> None:
        if not pending:
            return
        tiles = np.stack([item[4] for item in pending]).transpose(0, 3, 1, 2).copy()
        tensor = torch.from_numpy(tiles).float().to(device) / 255.0
        probabilities = model(tensor).softmax(dim=1)[:, 1].cpu().numpy()
        for (x, y, valid_width, valid_height, _), probability in zip(pending, probabilities):
            target[y:y + valid_height, x:x + valid_width] = probability[:valid_height, :valid_width] >= args.threshold
        pending.clear()

    with torch.inference_mode():
        for y in tile_positions(height, args.tile_size):
            for x in tile_positions(width, args.tile_size):
                tile, valid_width, valid_height = padded_tile(image, x, y, args.tile_size)
                pending.append((x, y, valid_width, valid_height, tile))
                if len(pending) >= args.batch_size:
                    flush()
        flush()
    return target


def evidence_image(image: Image.Image, prediction: np.ndarray, percent: float, path: Path) -> None:
    max_width = 1600
    scale = min(1.0, max_width / image.width)
    preview = image.convert("RGB").resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    mask = Image.fromarray((prediction.astype(np.uint8) * 120)).resize(preview.size, Image.Resampling.NEAREST)
    overlay = Image.new("RGB", preview.size, (220, 30, 140))
    preview = Image.composite(overlay, preview, mask)
    canvas = Image.new("RGB", (preview.width, preview.height + 28), "white")
    canvas.paste(preview, (0, 28))
    ImageDraw.Draw(canvas).text((8, 7), f"Predicted tubular-atrophy target burden: {percent:.2%}", fill="black")
    canvas.save(path, quality=95)


def main() -> None:
    args = parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be in (0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_map = scene_split_map(args.manifest_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=2, bilinear=False).to(device)
    state = load_torch_state(args.checkpoint, map_location=device)
    state.pop("mask_values", None)
    model.load_state_dict(state)
    model.eval()

    rows = []
    evidence = []
    for scene_path, scene_id, scene_split in scene_paths(args.scenes_dir, split_map, args.split, args.limit_scenes):
        with Image.open(scene_path) as handle:
            image = handle.convert("RGB")
        image_array = np.asarray(image)
        prediction = scene_prediction(model, image, args, device)
        tissue_pixels = int(tissue_mask(image_array).sum())
        target_pixels = int(np.logical_and(prediction, tissue_mask(image_array)).sum())
        percent = target_pixels / tissue_pixels if tissue_pixels else 0.0
        row = {
            "slide_id": scene_id,
            "case_id": scene_id.rsplit("_s", 1)[0],
            "split": scene_split,
            "scene_path": str(scene_path),
            "width": image.width,
            "height": image.height,
            "tissue_pixels": tissue_pixels,
            "predicted_target_pixels": target_pixels,
            "tubular_atrophy_target_percent": f"{percent:.6f}",
            "threshold": args.threshold,
            "metric_definition": "predicted_atrophic_tubule_target_pixels / nonwhite_tissue_pixels",
        }
        rows.append(row)
        if args.max_evidence > 0:
            evidence.append((percent, scene_id, image, prediction))
        print(json.dumps({key: row[key] for key in ("slide_id", "split", "tubular_atrophy_target_percent")}, ensure_ascii=False))

    if not rows:
        raise ValueError("No scene images selected")
    output_csv = args.output_dir / "tubular_atrophy_scene_predictions.csv"
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    evidence_dir = args.output_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    for percent, slide_id, image, prediction in sorted(evidence, reverse=True)[: args.max_evidence]:
        evidence_image(image, prediction, percent, evidence_dir / f"{slide_id}_tubular_atrophy.jpg")
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "scene_count": len(rows),
                "split": args.split,
                "threshold": args.threshold,
                "checkpoint": str(args.checkpoint),
                "scene_predictions": str(output_csv),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
