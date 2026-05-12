import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_IMAGES_DIR = "/root/datasets/HZY_HSPN_export_ds025/images"
DEFAULT_ANNOTATIONS_DIR = "/root/datasets/HZY_HSPN_export_ds025/annotations"
DEFAULT_OUTPUT_DIR = "/root/datasets/HZY_HSPN_export_ds025/debug_label_gallery_color"

DEFAULT_LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "未废弃肾小球": (0, 200, 80),
    "废弃肾小球": (80, 80, 80),
    "肾小球系膜细胞增生": (255, 180, 0),
    "毛细血管内细胞增生": (255, 90, 0),
    "细胞性新月体": (220, 0, 255),
    "纤维细胞性新月体": (150, 70, 255),
    "纤维性新月体": (0, 120, 255),
    "节段硬化": (255, 0, 0),
    "节段球囊粘连": (0, 200, 220),
    "纤维素样坏死": (120, 0, 0),
    "纤维素性血栓": (0, 0, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render color-coded local crop galleries for HZY annotation labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, help="Directory containing scene TIFF files")
    parser.add_argument("--annotations-dir", default=DEFAULT_ANNOTATIONS_DIR, help="Directory containing GeoJSON files")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for label gallery outputs")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=list(DEFAULT_LABEL_COLORS.keys()),
        help="Labels to render. Defaults to the known HZY HSPN annotation labels.",
    )
    parser.add_argument("--image-suffix", default=".tiff", help="Suffix for scene image files")
    parser.add_argument("--annotation-suffix", default=".json", help="Suffix for annotation files")
    parser.add_argument("--max-per-label", type=int, default=24, help="Maximum crops sampled for each label")
    parser.add_argument("--pad", type=int, default=512, help="Padding around each polygon crop in pixels")
    parser.add_argument("--crop-max-size", type=int, default=1400, help="Maximum saved crop width or height")
    parser.add_argument("--thumbnail-size", type=int, default=360, help="Thumbnail size in contact sheets")
    parser.add_argument("--fill-alpha", type=int, default=70, help="Polygon fill alpha in [0, 255]")
    parser.add_argument("--line-width", type=int, default=5, help="Polygon outline width")
    parser.add_argument("--jpeg-quality", type=int, default=96, help="Output JPG quality")
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed")
    parser.add_argument(
        "--slide-ids",
        nargs="*",
        default=[],
        help="Optional scene slide IDs to search. If omitted, all annotation JSON files are used.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip saving individual crop files that already exist.",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", text).strip("_")
    return cleaned or "label"


def normalize_rgb(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    while array.ndim > 3 and 1 in array.shape:
        array = np.squeeze(array)

    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    elif array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array[:3], 0, -1)
    elif array.ndim == 3 and array.shape[-1] >= 3:
        array = array[..., :3]
    else:
        raise ValueError("Could not normalize image array with shape {}".format(array.shape))

    if array.dtype == np.uint8:
        return array

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        if info.max <= 255:
            return array.astype(np.uint8)
        return np.clip(array.astype(np.float32) / float(info.max) * 255.0, 0, 255).astype(np.uint8)

    max_value = float(np.nanmax(array)) if array.size else 1.0
    min_value = float(np.nanmin(array)) if array.size else 0.0
    if max_value <= 1.0 and min_value >= 0.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def load_font(size: int = 22):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_text_safe(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, fill, font=None) -> None:
    try:
        draw.text(xy, text, fill=fill, font=font)
    except UnicodeEncodeError:
        draw.text(xy, text.encode("unicode_escape").decode("ascii"), fill=fill, font=font)


def get_label(feature: dict) -> str:
    properties = feature.get("properties") or {}
    classification = properties.get("classification") or {}
    if isinstance(classification, dict) and classification.get("name"):
        return str(classification["name"])
    return str(properties.get("name") or properties.get("label") or "")


def polygon_points(feature: dict) -> Optional[List[Tuple[float, float]]]:
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "Polygon":
        return None
    rings = geometry.get("coordinates") or []
    if not rings or len(rings[0]) < 3:
        return None
    return [(float(x), float(y)) for x, y in rings[0]]


def annotation_paths(annotations_dir: Path, annotation_suffix: str, slide_ids: Sequence[str]) -> List[Path]:
    if slide_ids:
        return [annotations_dir / "{}{}".format(slide_id, annotation_suffix) for slide_id in slide_ids]
    return sorted(annotations_dir.glob("*{}".format(annotation_suffix)))


def collect_examples(
    annotations_dir: Path,
    annotation_suffix: str,
    slide_ids: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, List[dict]]:
    label_set = set(labels)
    examples: Dict[str, List[dict]] = defaultdict(list)

    for json_path in annotation_paths(annotations_dir, annotation_suffix, slide_ids):
        if not json_path.exists():
            print("[skip] missing annotation:", json_path)
            continue

        slide_id = json_path.name[: -len(annotation_suffix)] if annotation_suffix else json_path.stem
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        for feature_index, feature in enumerate(data.get("features", [])):
            label = get_label(feature)
            if label not in label_set:
                continue
            points = polygon_points(feature)
            if not points:
                continue
            examples[label].append(
                {
                    "slide_id": slide_id,
                    "feature_index": feature_index,
                    "label": label,
                    "points": points,
                }
            )

    return examples


def sample_examples(examples: Dict[str, List[dict]], labels: Sequence[str], max_per_label: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    selected = []
    for label in labels:
        items = list(examples.get(label, []))
        rng.shuffle(items)
        chosen = items[:max_per_label] if max_per_label > 0 else items
        selected.extend(chosen)
        print("{} total={} selected={}".format(label, len(items), len(chosen)))
    return selected


def crop_bounds(points: Sequence[Tuple[float, float]], width: int, height: int, pad: int) -> Tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0 = max(0, int(min(xs)) - pad)
    y0 = max(0, int(min(ys)) - pad)
    x1 = min(width, int(max(xs)) + pad)
    y1 = min(height, int(max(ys)) + pad)
    return x0, y0, x1, y1


def color_for_label(label: str) -> Tuple[int, int, int]:
    return DEFAULT_LABEL_COLORS.get(label, (255, 0, 0))


def render_crop(
    image_array: np.ndarray,
    item: dict,
    output_dir: Path,
    crop_max_size: int,
    pad: int,
    fill_alpha: int,
    line_width: int,
    jpeg_quality: int,
    skip_existing: bool,
    font,
) -> Optional[dict]:
    height, width = image_array.shape[:2]
    points = item["points"]
    label = item["label"]
    x0, y0, x1, y1 = crop_bounds(points, width=width, height=height, pad=pad)
    if x1 <= x0 or y1 <= y0:
        return None

    label_dir = output_dir / safe_name(label)
    label_dir.mkdir(parents=True, exist_ok=True)
    out_name = "{}_feature{:04d}_{}.jpg".format(item["slide_id"], item["feature_index"], safe_name(label))
    out_path = label_dir / out_name
    if out_path.exists() and skip_existing:
        return {
            "label": label,
            "color_rgb": str(color_for_label(label)),
            "slide_id": item["slide_id"],
            "feature_index": item["feature_index"],
            "image_path": str(out_path),
            "crop_x0": x0,
            "crop_y0": y0,
            "crop_x1": x1,
            "crop_y1": y1,
        }

    crop = image_array[y0:y1, x0:x1].copy()
    image = Image.fromarray(crop).convert("RGBA")
    shifted = [(x - x0, y - y0) for x, y in points]
    red, green, blue = color_for_label(label)
    fill_color = (red, green, blue, max(0, min(fill_alpha, 255)))
    line_color = (red, green, blue, 255)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.polygon(shifted, fill=fill_color, outline=line_color)
    image.alpha_composite(overlay)

    draw = ImageDraw.Draw(image)
    draw.line(shifted + [shifted[0]], fill=line_color, width=line_width)
    draw.rectangle([10, 10, 52, 52], fill=line_color)
    draw_text_safe(draw, (62, 18), label, fill=(0, 0, 0, 255), font=font)

    rgb = image.convert("RGB")
    if crop_max_size > 0 and max(rgb.size) > crop_max_size:
        rgb.thumbnail((crop_max_size, crop_max_size))
    rgb.save(out_path, quality=jpeg_quality)

    return {
        "label": label,
        "color_rgb": str(color_for_label(label)),
        "slide_id": item["slide_id"],
        "feature_index": item["feature_index"],
        "image_path": str(out_path),
        "crop_x0": x0,
        "crop_y0": y0,
        "crop_x1": x1,
        "crop_y1": y1,
    }


def render_label_sheets(
    labels: Sequence[str],
    output_dir: Path,
    thumbnail_size: int,
    jpeg_quality: int,
    font,
) -> None:
    for label in labels:
        label_dir = output_dir / safe_name(label)
        paths = sorted(label_dir.glob("*.jpg"))
        if not paths:
            continue

        thumbs = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumbnail_size, thumbnail_size))
            canvas = Image.new("RGB", (thumbnail_size, thumbnail_size + 34), "white")
            canvas.paste(image, ((thumbnail_size - image.size[0]) // 2, 0))
            draw = ImageDraw.Draw(canvas)
            color = color_for_label(label)
            draw.rectangle([8, thumbnail_size + 8, 36, thumbnail_size + 28], fill=color)
            draw_text_safe(draw, (44, thumbnail_size + 8), label, fill=(0, 0, 0), font=font)
            thumbs.append(canvas)

        cols = 4
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumbnail_size, rows * (thumbnail_size + 34)), "white")
        for index, image in enumerate(thumbs):
            x = (index % cols) * thumbnail_size
            y = (index // cols) * (thumbnail_size + 34)
            sheet.paste(image, (x, y))

        sheet_path = output_dir / "gallery_{}.jpg".format(safe_name(label))
        sheet.save(sheet_path, quality=jpeg_quality)
        print("saved gallery:", sheet_path)


def render_legend(labels: Sequence[str], examples: Dict[str, List[dict]], output_dir: Path, jpeg_quality: int, font) -> None:
    row_height = 44
    legend = Image.new("RGB", (920, 44 + len(labels) * row_height), "white")
    draw = ImageDraw.Draw(legend)
    draw_text_safe(draw, (20, 12), "Label color legend", fill=(0, 0, 0), font=font)
    for index, label in enumerate(labels):
        y = 44 + index * row_height
        color = color_for_label(label)
        draw.rectangle([20, y, 60, y + 26], fill=color)
        text = "{}  RGB={}  n={}".format(label, color, len(examples.get(label, [])))
        draw_text_safe(draw, (76, y + 4), text, fill=(0, 0, 0), font=font)
    legend_path = output_dir / "label_color_legend.jpg"
    legend.save(legend_path, quality=jpeg_quality)
    print("saved legend:", legend_path)


def write_summary(rows: List[dict], output_path: Path) -> None:
    fieldnames = [
        "label",
        "color_rgb",
        "slide_id",
        "feature_index",
        "image_path",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = list(args.labels)
    examples = collect_examples(
        annotations_dir=annotations_dir,
        annotation_suffix=args.annotation_suffix,
        slide_ids=args.slide_ids,
        labels=labels,
    )
    selected = sample_examples(
        examples=examples,
        labels=labels,
        max_per_label=args.max_per_label,
        seed=args.seed,
    )

    selected_by_slide: Dict[str, List[dict]] = defaultdict(list)
    for item in selected:
        selected_by_slide[item["slide_id"]].append(item)

    import tifffile

    font = load_font(size=22)
    summary_rows = []
    for slide_index, (slide_id, items) in enumerate(sorted(selected_by_slide.items()), start=1):
        image_path = images_dir / "{}{}".format(slide_id, args.image_suffix)
        if not image_path.exists():
            print("[skip] missing image:", image_path)
            continue

        print("[{} / {}] reading {} items={}".format(slide_index, len(selected_by_slide), slide_id, len(items)))
        image_array = normalize_rgb(tifffile.imread(str(image_path)))
        for item in items:
            row = render_crop(
                image_array=image_array,
                item=item,
                output_dir=output_dir,
                crop_max_size=args.crop_max_size,
                pad=args.pad,
                fill_alpha=args.fill_alpha,
                line_width=args.line_width,
                jpeg_quality=args.jpeg_quality,
                skip_existing=args.skip_existing,
                font=font,
            )
            if row is not None:
                summary_rows.append(row)

    render_label_sheets(
        labels=labels,
        output_dir=output_dir,
        thumbnail_size=args.thumbnail_size,
        jpeg_quality=args.jpeg_quality,
        font=font,
    )
    render_legend(labels=labels, examples=examples, output_dir=output_dir, jpeg_quality=args.jpeg_quality, font=font)

    summary_path = output_dir / "label_gallery_summary.csv"
    write_summary(summary_rows, summary_path)
    print("Rendered crops:", len(summary_rows))
    print("Output directory:", output_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
