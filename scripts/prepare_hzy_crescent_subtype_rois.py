import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.detection_boxes import boxes_from_binary_mask


DEFAULT_CLASS_NAMES = {
    1: "cellular_crescent",
    2: "fibrocellular_crescent",
    3: "fibrous_crescent",
}

DEFAULT_CLASS_LABELS = {
    1: "细胞性新月体",
    2: "纤维细胞性新月体",
    3: "纤维性新月体",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract crescent subtype ROI crops from fine-grained crescent masks for image classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", required=True, help="Crescent segmentation dataset root containing train/val images and masks")
    parser.add_argument("--output-root", required=True, help="Output image-folder classification dataset root")
    parser.add_argument("--splits", nargs="*", default=["train", "val"], help="Dataset splits to process")
    parser.add_argument("--class-ids", nargs="*", type=int, default=[1, 2, 3], help="Foreground class ids to extract")
    parser.add_argument("--min-area", type=int, default=32, help="Minimum connected-component area in mask pixels")
    parser.add_argument("--box-margin", type=int, default=48, help="Context margin around each lesion component before square padding")
    parser.add_argument("--crop-size", type=int, default=224, help="Saved ROI crop size")
    parser.add_argument("--image-suffix", default=".jpg", help="Image suffix in the segmentation dataset")
    parser.add_argument("--mask-suffix", default=".png", help="Mask suffix in the segmentation dataset")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Remove output directory before writing")
    return parser.parse_args()


def validate_args(args):
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError("Dataset root does not exist: {}".format(dataset_root))
    if args.min_area <= 0:
        raise ValueError("--min-area must be positive")
    if args.box_margin < 0:
        raise ValueError("--box-margin must be non-negative")
    if args.crop_size <= 0:
        raise ValueError("--crop-size must be positive")
    unknown = sorted(set(args.class_ids) - set(DEFAULT_CLASS_NAMES))
    if unknown:
        raise ValueError("Unsupported class ids: {}".format(", ".join(str(item) for item in unknown)))


def square_box(x_min: int, y_min: int, x_max: int, y_max: int, width: int, height: int) -> Tuple[int, int, int, int]:
    box_width = max(1, x_max - x_min)
    box_height = max(1, y_max - y_min)
    side = max(box_width, box_height)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    new_x_min = int(round(center_x - side / 2.0))
    new_y_min = int(round(center_y - side / 2.0))
    new_x_max = new_x_min + side
    new_y_max = new_y_min + side

    if new_x_min < 0:
        new_x_max -= new_x_min
        new_x_min = 0
    if new_y_min < 0:
        new_y_max -= new_y_min
        new_y_min = 0
    if new_x_max > width:
        shift = new_x_max - width
        new_x_min = max(0, new_x_min - shift)
        new_x_max = width
    if new_y_max > height:
        shift = new_y_max - height
        new_y_min = max(0, new_y_min - shift)
        new_y_max = height

    return int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iter_masks(mask_dir: Path, suffix: str) -> Iterable[Path]:
    return sorted(path for path in mask_dir.glob("*{}".format(suffix)) if not path.name.startswith("."))


def extract_split(args, split: str, output_root: Path) -> List[Dict[str, object]]:
    dataset_root = Path(args.dataset_root)
    image_dir = dataset_root / split / "images"
    mask_dir = dataset_root / split / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError("Missing split directories under {}".format(dataset_root / split))

    rows: List[Dict[str, object]] = []
    component_counts: Counter = Counter()
    for mask_path in iter_masks(mask_dir, args.mask_suffix):
        image_path = image_dir / "{}{}".format(mask_path.stem, args.image_suffix)
        if not image_path.exists():
            raise FileNotFoundError("Missing image for mask {}: {}".format(mask_path, image_path))

        image = Image.open(image_path).convert("RGB")
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim != 2:
            raise ValueError("Expected 2D mask, got {} for {}".format(mask.shape, mask_path))

        width, height = image.size
        for class_id in args.class_ids:
            boxes = boxes_from_binary_mask(
                mask == class_id,
                class_id=class_id,
                min_area=args.min_area,
                margin=args.box_margin,
            )
            for box_index, box in enumerate(boxes, start=1):
                x_min, y_min, x_max, y_max = square_box(box.x_min, box.y_min, box.x_max, box.y_max, width, height)
                class_slug = DEFAULT_CLASS_NAMES[class_id]
                roi_id = "{}_c{}_r{:02d}".format(mask_path.stem, class_id, box_index)
                relative_path = Path(split) / class_slug / "{}.jpg".format(roi_id)
                output_path = output_root / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)

                roi = image.crop((x_min, y_min, x_max, y_max)).resize(
                    (args.crop_size, args.crop_size),
                    resample=Image.BICUBIC,
                )
                roi.save(output_path, quality=95)

                component_counts[(split, class_slug)] += 1
                rows.append(
                    {
                        "roi_id": roi_id,
                        "split": split,
                        "class_id": class_id,
                        "class_slug": class_slug,
                        "class_label": DEFAULT_CLASS_LABELS[class_id],
                        "relative_path": str(relative_path).replace("\\", "/"),
                        "source_image": str(image_path),
                        "source_mask": str(mask_path),
                        "source_stem": mask_path.stem,
                        "component_area": int(box.area),
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "source_width": width,
                        "source_height": height,
                        "crop_size": args.crop_size,
                    }
                )
    return rows


def main():
    args = parse_args()
    validate_args(args)
    output_root = Path(args.output_root)
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError("Output root already exists. Re-run with --overwrite: {}".format(output_root))
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    for split in args.splits:
        all_rows.extend(extract_split(args, split, output_root))

    write_csv(all_rows, output_root / "manifest.csv")
    summary = {
        "dataset_root": str(Path(args.dataset_root)),
        "output_root": str(output_root),
        "class_names": {str(key): value for key, value in DEFAULT_CLASS_NAMES.items() if key in args.class_ids},
        "class_labels": {str(key): value for key, value in DEFAULT_CLASS_LABELS.items() if key in args.class_ids},
        "splits": {},
        "total_rois": len(all_rows),
        "min_area": args.min_area,
        "box_margin": args.box_margin,
        "crop_size": args.crop_size,
    }
    for split in args.splits:
        split_rows = [row for row in all_rows if row["split"] == split]
        summary["splits"][split] = {
            "rois": len(split_rows),
            "class_counts": dict(Counter(row["class_slug"] for row in split_rows)),
        }

    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
