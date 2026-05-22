import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from hzy_lesion_config import resolve_lesion_labels


DEFAULT_IMAGES_DIR = "/root/datasets/HZY_HSPN_export_ds025/images"
DEFAULT_ANNOTATIONS_DIR = "/root/datasets/HZY_HSPN_export_ds025/annotations"
DEFAULT_OUTPUT_ROOT = "/root/datasets/HZY_HSPN_lesion_tiles"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare one binary tile dataset per HZY lesion label.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, help="Directory containing scene TIFF images")
    parser.add_argument("--annotations-dir", default=DEFAULT_ANNOTATIONS_DIR, help="Directory containing scene JSON annotations")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root directory for lesion tile datasets")
    parser.add_argument("--preset", choices=("trainable", "all", "rare"), default="trainable",
                        help="Default lesion label set when --labels is omitted")
    parser.add_argument("--labels", nargs="*", default=[],
                        help="Optional lesion labels or slugs to prepare")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size for lesion crops")
    parser.add_argument("--stride", type=int, default=512, help="Sliding-window stride")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation slide ratio")
    parser.add_argument("--split-csv", default="", help="Optional slide split CSV with columns slide_id,split")
    parser.add_argument("--min-tissue-coverage", type=float, default=0.05, help="Minimum tissue coverage to keep a tile")
    parser.add_argument("--min-positive-pixels", type=int, default=16,
                        help="Minimum positive pixels required to keep a positive lesion tile")
    parser.add_argument("--negative-ratio", type=float, default=3.0,
                        help="How many negative tiles to keep per positive tile")
    parser.add_argument("--max-background-tiles-per-slide", type=int, default=50,
                        help="Negative tile cap for slides without positive tiles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits and negative sampling")
    parser.add_argument("--downsample", type=float, default=1.0, help="Optional downsample when saving tiles")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Pass through for bookkeeping; existing files may be overwritten by prepare_hubmap_tiles.py")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Print commands without running them")
    return parser.parse_args()


def script_path() -> Path:
    return Path(__file__).resolve().parent / "prepare_hubmap_tiles.py"


def build_prepare_command(args, lesion) -> list:
    output_dir = Path(args.output_root) / lesion.slug
    command = [
        sys.executable,
        str(script_path()),
        "--images-dir",
        args.images_dir,
        "--annotations-dir",
        args.annotations_dir,
        "--annotation-format",
        "json-polygons",
        "--annotation-json-suffix",
        ".json",
        "--target-labels",
        lesion.label,
        "--output-dir",
        str(output_dir),
        "--tile-size",
        str(args.tile_size),
        "--stride",
        str(args.stride),
        "--val-ratio",
        str(args.val_ratio),
        "--min-tissue-coverage",
        str(args.min_tissue_coverage),
        "--min-positive-pixels",
        str(args.min_positive_pixels),
        "--negative-ratio",
        str(args.negative_ratio),
        "--max-background-tiles-per-slide",
        str(args.max_background_tiles_per_slide),
        "--downsample",
        str(args.downsample),
        "--seed",
        str(args.seed),
    ]
    if args.split_csv:
        command.extend(["--split-csv", args.split_csv])
    return command


def write_index(rows, output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "lesion_datasets.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["label", "slug", "count", "output_dir", "status", "command"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_root / "lesion_datasets.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    return csv_path, json_path


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    lesions = resolve_lesion_labels(args.preset, args.labels)

    rows = []
    for lesion in lesions:
        command = build_prepare_command(args, lesion)
        output_dir = output_root / lesion.slug
        command_text = " ".join('"{}"'.format(item) if " " in item else item for item in command)
        print("\n==> {} ({}, n={})".format(lesion.label, lesion.slug, lesion.count))
        print(command_text)

        status = "dry_run"
        if not args.dry_run:
            completed = subprocess.run(command, check=False)
            status = "ok" if completed.returncode == 0 else "failed_{}".format(completed.returncode)
            if completed.returncode != 0:
                rows.append({
                    "label": lesion.label,
                    "slug": lesion.slug,
                    "count": lesion.count,
                    "output_dir": str(output_dir),
                    "status": status,
                    "command": command_text,
                })
                csv_path, json_path = write_index(rows, output_root)
                raise SystemExit(
                    "Failed while preparing '{}'. Index written to {} and {}".format(lesion.label, csv_path, json_path)
                )

        rows.append({
            "label": lesion.label,
            "slug": lesion.slug,
            "count": lesion.count,
            "output_dir": str(output_dir),
            "status": status,
            "command": command_text,
        })

    csv_path, json_path = write_index(rows, output_root)
    print("\nWrote lesion dataset index:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
