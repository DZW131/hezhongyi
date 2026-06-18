import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.detection_boxes import Box, boxes_from_multiclass_mask


DEFAULT_DATASET_ROOT = "/home/duyanhong/Dataspace/HZY/HZY_HSPN_glomerulus_lesion_tasks_proliferation_binary_768/proliferation_binary"
DEFAULT_OUTPUT_ROOT = "/home/duyanhong/Dataspace/HZY/HZY_HSPN_detection_boxes/proliferation_binary"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert HZY glomerulus-crop lesion masks into detection boxes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT,
                        help="Task dataset root containing train/images, train/masks, val/images, val/masks")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help="Output root for YOLO labels, optional image links, COCO JSON, and manifests")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Dataset splits to convert")
    parser.add_argument("--class-mode", choices=("binary", "per-class"), default="binary",
                        help="binary collapses all foreground mask classes into one lesion class")
    parser.add_argument("--class-name", default="proliferation", help="Class name used for binary mode")
    parser.add_argument("--class-names", default="",
                        help="Comma-separated class names for per-class mode, ordered by positive mask id")
    parser.add_argument("--min-area", type=int, default=16, help="Minimum connected-component pixel area")
    parser.add_argument("--box-margin", type=int, default=4, help="Extra pixels added around each component box")
    parser.add_argument("--link-mode", choices=("symlink", "copy", "none"), default="symlink",
                        help="How to expose images under output-root/{split}/images for detector training")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Remove existing output-root first")
    return parser.parse_args()


def read_mask(mask_path: Path) -> np.ndarray:
    return np.asarray(Image.open(mask_path).convert("L"))


def find_image(images_dir: Path, stem: str) -> Path:
    matches = sorted(path for path in images_dir.glob(stem + ".*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(matches) != 1:
        raise FileNotFoundError("Expected one image for '{}', found {} in {}".format(stem, len(matches), images_dir))
    return matches[0]


def expose_image(source: Path, target: Path, mode: str):
    if mode == "none":
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(source, target)
        return
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def write_yolo_label(label_path: Path, boxes: List[Box], image_width: int, image_height: int, class_mode: str):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as handle:
        for box in boxes:
            yolo_class_id = 0 if class_mode == "binary" else box.class_id - 1
            class_id, x_center, y_center, width, height = box.to_yolo(
                image_width=image_width,
                image_height=image_height,
                yolo_class_id=yolo_class_id,
            )
            handle.write("{} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(class_id, x_center, y_center, width, height))


def category_mapping(args) -> Dict[int, str]:
    if args.class_mode == "binary":
        return {1: args.class_name}
    names = [item.strip() for item in args.class_names.split(",") if item.strip()]
    return {index: names[index - 1] if index <= len(names) else "class_{}".format(index) for index in range(1, 16)}


def coco_category_id(box: Box, class_mode: str) -> int:
    return 1 if class_mode == "binary" else box.class_id


def convert_split(args, split: str, categories: Dict[int, str]):
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    images_dir = dataset_root / split / "images"
    masks_dir = dataset_root / split / "masks"
    output_images_dir = output_root / split / "images"
    output_labels_dir = output_root / split / "labels"

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError("Missing split directories: {} and {}".format(images_dir, masks_dir))

    rows = []
    coco_images = []
    coco_annotations = []
    annotation_id = 1
    mask_paths = sorted(path for path in masks_dir.glob("*.png") if not path.name.startswith("."))

    for image_id, mask_path in enumerate(tqdm(mask_paths, desc="Convert {}".format(split), unit="mask"), start=1):
        image_path = find_image(images_dir, mask_path.stem)
        mask = read_mask(mask_path)
        height, width = mask.shape[:2]
        boxes = boxes_from_multiclass_mask(
            mask,
            collapse_to_class=1 if args.class_mode == "binary" else None,
            min_area=args.min_area,
            margin=args.box_margin,
        )

        exposed_image_path = output_images_dir / image_path.name
        expose_image(image_path, exposed_image_path, args.link_mode)
        label_path = output_labels_dir / (image_path.stem + ".txt")
        write_yolo_label(label_path, boxes, image_width=width, image_height=height, class_mode=args.class_mode)

        coco_file_name = str((Path(split) / "images" / image_path.name).as_posix())
        if args.link_mode == "none":
            coco_file_name = str(image_path.resolve())
        coco_images.append({"id": image_id, "file_name": coco_file_name, "width": int(width), "height": int(height)})

        for box_index, box in enumerate(boxes, start=1):
            category_id = coco_category_id(box, args.class_mode)
            x, y, box_width, box_height = box.to_xywh()
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, box_width, box_height],
                    "area": int(box_width * box_height),
                    "iscrowd": 0,
                }
            )
            rows.append(
                {
                    "split": split,
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "label": str(label_path),
                    "box_index": box_index,
                    "class_id": category_id,
                    "class_name": categories.get(category_id, "class_{}".format(category_id)),
                    "x_min": box.x_min,
                    "y_min": box.y_min,
                    "x_max": box.x_max,
                    "y_max": box.y_max,
                    "component_area": box.area,
                    "box_area": box.box_area,
                }
            )
            annotation_id += 1

        if not boxes:
            rows.append(
                {
                    "split": split,
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "label": str(label_path),
                    "box_index": 0,
                    "class_id": "",
                    "class_name": "",
                    "x_min": "",
                    "y_min": "",
                    "x_max": "",
                    "y_max": "",
                    "component_area": 0,
                    "box_area": 0,
                }
            )

    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": class_id, "name": name} for class_id, name in sorted(categories.items()) if class_id in {ann["category_id"] for ann in coco_annotations} or args.class_mode == "binary"],
    }
    return rows, coco


def write_csv(rows: List[Dict[str, object]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_yaml(args, categories: Dict[int, str]):
    output_root = Path(args.output_root)
    names = [name for _, name in sorted(categories.items())]
    if args.class_mode == "per-class":
        names = names[: max(1, len([name for name in names if name]))]
    payload = [
        "path: {}".format(output_root.as_posix()),
        "train: train/images",
        "val: val/images",
        "names:",
    ]
    for index, name in enumerate(names):
        payload.append("  {}: {}".format(index, name))
    (output_root / "data.yaml").write_text("\n".join(payload) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    categories = category_mapping(args)
    all_rows = []
    summary = {
        "dataset_root": args.dataset_root,
        "output_root": args.output_root,
        "class_mode": args.class_mode,
        "min_area": args.min_area,
        "box_margin": args.box_margin,
        "splits": {},
    }

    for split in args.splits:
        rows, coco = convert_split(args, split, categories)
        all_rows.extend(rows)
        with (output_root / "annotations_{}.json".format(split)).open("w", encoding="utf-8") as handle:
            json.dump(coco, handle, indent=2, ensure_ascii=False)
        positive_images = len({row["image"] for row in rows if row.get("box_index")})
        summary["splits"][split] = {
            "images": len(coco["images"]),
            "boxes": len(coco["annotations"]),
            "positive_images": positive_images,
            "negative_images": len(coco["images"]) - positive_images,
        }

    write_csv(all_rows, output_root / "boxes.csv")
    write_dataset_yaml(args, categories)
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("Wrote detection dataset:", output_root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
