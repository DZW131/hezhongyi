import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_IMAGES_DIR = "/root/datasets/HZY_HSPN_export_ds025/images"
DEFAULT_ANNOTATIONS_DIR = "/root/datasets/HZY_HSPN_export_ds025/annotations"
DEFAULT_PREDICTIONS_DIR = "/root/Pytorch-UNet/Pytorch-UNet-master/predictions/hzy_scene_ds025"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render original HZY scene TIFF + doctor polygons + predicted mask overlays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--slide-ids",
        nargs="*",
        default=[],
        help=(
            "Scene slide IDs to render, for example 2026001_s0 2026002_s1. "
            "If omitted, slide IDs are discovered from prediction masks."
        ),
    )
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, help="Directory containing scene TIFF files")
    parser.add_argument("--annotations-dir", default=DEFAULT_ANNOTATIONS_DIR, help="Directory containing GeoJSON files")
    parser.add_argument("--predictions-dir", default=DEFAULT_PREDICTIONS_DIR, help="Directory containing predicted masks")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for overlay JPG files. Defaults to --predictions-dir.",
    )
    parser.add_argument("--image-suffix", default=".tiff", help="Suffix for scene image files")
    parser.add_argument("--annotation-suffix", default=".json", help="Suffix for annotation files")
    parser.add_argument("--prediction-suffix", default="_pred_mask.png", help="Suffix for predicted mask files")
    parser.add_argument("--output-suffix", default="_overlay_gt_pred.jpg", help="Suffix for output overlay files")
    parser.add_argument(
        "--preview-max-size",
        type=int,
        default=8000,
        help="Maximum preview width or height. Use 0 to keep original size.",
    )
    parser.add_argument("--gt-width", type=int, default=3, help="Doctor polygon line width")
    parser.add_argument("--pred-alpha", type=int, default=80, help="Prediction overlay alpha in [0, 255]")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="Output JPG quality")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of overlays to render")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip slide IDs with missing image, annotation, or prediction files instead of raising an error.",
    )
    return parser.parse_args()


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


def discover_slide_ids(predictions_dir: Path, prediction_suffix: str) -> List[str]:
    pattern = "*{}".format(prediction_suffix)
    slide_ids = []
    for path in sorted(predictions_dir.glob(pattern)):
        name = path.name
        slide_ids.append(name[: -len(prediction_suffix)])
    return slide_ids


def require_or_skip(paths: Sequence[Path], skip_missing: bool) -> bool:
    missing = [path for path in paths if not path.exists()]
    if not missing:
        return True
    if skip_missing:
        print("[skip] missing files:", ", ".join(str(path) for path in missing))
        return False
    raise FileNotFoundError("Missing required file(s): {}".format(", ".join(str(path) for path in missing)))


def resize_preview(image: Image.Image, preview_max_size: int) -> Tuple[Image.Image, float, float]:
    orig_w, orig_h = image.size
    preview = image.copy()
    if preview_max_size > 0:
        preview.thumbnail((preview_max_size, preview_max_size))
    sx = preview.size[0] / float(orig_w)
    sy = preview.size[1] / float(orig_h)
    return preview, sx, sy


def load_annotation_features(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and isinstance(data.get("features"), list):
        return [feature for feature in data["features"] if isinstance(feature, dict)]
    if isinstance(data, list):
        return [feature for feature in data if isinstance(feature, dict)]
    raise ValueError("Unsupported annotation JSON structure: {}".format(json_path))


def draw_prediction_overlay(
    preview: Image.Image,
    pred_path: Path,
    original_size: Tuple[int, int],
    pred_alpha: int,
) -> Image.Image:
    pred = Image.open(pred_path).convert("L")
    if pred.size != original_size:
        pred = pred.resize(original_size, Image.NEAREST)

    pred_small = pred.resize(preview.size, Image.NEAREST)
    alpha = np.asarray(pred_small) > 0
    blue_arr = np.zeros((preview.size[1], preview.size[0], 4), dtype=np.uint8)
    blue_arr[..., 2] = 255
    blue_arr[..., 1] = 120
    blue_arr[..., 3] = alpha.astype(np.uint8) * max(0, min(pred_alpha, 255))

    preview_rgba = preview.convert("RGBA")
    preview_rgba.alpha_composite(Image.fromarray(blue_arr, "RGBA"))
    return preview_rgba.convert("RGB")


def draw_gt_polygons(
    preview: Image.Image,
    features: Iterable[dict],
    scale_x: float,
    scale_y: float,
    line_width: int,
) -> None:
    draw = ImageDraw.Draw(preview)
    for feature in features:
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Polygon":
            continue

        rings = geometry.get("coordinates") or []
        if not rings:
            continue

        outer_ring = rings[0]
        points = [(float(x) * scale_x, float(y) * scale_y) for x, y in outer_ring]
        if len(points) >= 3:
            draw.line(points + [points[0]], fill="red", width=line_width)


def render_one_overlay(
    slide_id: str,
    images_dir: Path,
    annotations_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
    image_suffix: str,
    annotation_suffix: str,
    prediction_suffix: str,
    output_suffix: str,
    preview_max_size: int,
    gt_width: int,
    pred_alpha: int,
    jpeg_quality: int,
    skip_missing: bool,
) -> Optional[Path]:
    image_path = images_dir / "{}{}".format(slide_id, image_suffix)
    annotation_path = annotations_dir / "{}{}".format(slide_id, annotation_suffix)
    prediction_path = predictions_dir / "{}{}".format(slide_id, prediction_suffix)
    output_path = output_dir / "{}{}".format(slide_id, output_suffix)

    if not require_or_skip([image_path, annotation_path, prediction_path], skip_missing=skip_missing):
        return None

    import tifffile

    image_array = normalize_rgb(tifffile.imread(str(image_path)))
    image = Image.fromarray(image_array)
    preview, scale_x, scale_y = resize_preview(image, preview_max_size=preview_max_size)

    preview = draw_prediction_overlay(
        preview=preview,
        pred_path=prediction_path,
        original_size=image.size,
        pred_alpha=pred_alpha,
    )
    draw_gt_polygons(
        preview=preview,
        features=load_annotation_features(annotation_path),
        scale_x=scale_x,
        scale_y=scale_y,
        line_width=gt_width,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path, quality=jpeg_quality)
    return output_path


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_dir

    slide_ids = list(args.slide_ids) or discover_slide_ids(predictions_dir, args.prediction_suffix)
    if args.limit > 0:
        slide_ids = slide_ids[: args.limit]
    if not slide_ids:
        raise ValueError("No slide IDs found. Pass --slide-ids or create prediction masks first.")

    rendered = 0
    for index, slide_id in enumerate(slide_ids, start=1):
        print("[{} / {}] {}".format(index, len(slide_ids), slide_id))
        output_path = render_one_overlay(
            slide_id=slide_id,
            images_dir=images_dir,
            annotations_dir=annotations_dir,
            predictions_dir=predictions_dir,
            output_dir=output_dir,
            image_suffix=args.image_suffix,
            annotation_suffix=args.annotation_suffix,
            prediction_suffix=args.prediction_suffix,
            output_suffix=args.output_suffix,
            preview_max_size=args.preview_max_size,
            gt_width=args.gt_width,
            pred_alpha=args.pred_alpha,
            jpeg_quality=args.jpeg_quality,
            skip_missing=args.skip_missing,
        )
        if output_path is not None:
            rendered += 1
            print("saved:", output_path)

    print("Rendered:", rendered)
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
