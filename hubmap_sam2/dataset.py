from __future__ import annotations

from bisect import bisect_left, bisect_right
import csv
import json
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except ImportError:  # pragma: no cover - optional at import time
    cv2 = None

try:
    import tifffile
except ImportError:  # pragma: no cover - optional at import time
    tifffile = None


IMAGE_SUFFIXES = (".tif", ".tiff")
PREPARED_IMAGE_NAME = "00000.png"
PREPARED_MASK_NAME = "00000.png"
PREPARED_IMAGE_CANDIDATES = ("00000.png", "00000.jpg", "00000.jpeg")
DEFAULT_PALETTE = [0, 0, 0] + [value for idx in range(1, 256) for value in (idx, idx, idx)]


@dataclass
class PolygonRecord:
    record_id: int
    label: str
    polygons: List[List[List[Tuple[float, float]]]]
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    area: float


@dataclass
class PreparedSample:
    sample_id: str
    split: str
    slide_id: str
    image_path: Path
    annotation_path: Path
    metadata_path: Path


def require_tifffile() -> None:
    if tifffile is None:
        raise ImportError(
            "tifffile is required for HuBMAP preprocessing and WSI inference. "
            'Install the project with `pip install -e ".[hubmap]"`.'
        )


def normalize_label(label: Optional[str]) -> str:
    return (label or "").strip().lower()


def read_geojson_features(json_path: Path) -> List[Dict[str, object]]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("features"), list):
            return [item for item in data["features"] if isinstance(item, dict)]
        return [data]
    raise ValueError(f"Unsupported JSON structure in {json_path}")


def get_feature_label(feature: Dict[str, object]) -> str:
    properties = feature.get("properties") or {}
    if isinstance(properties, dict):
        classification = properties.get("classification") or {}
        if isinstance(classification, dict):
            label = classification.get("name") or classification.get("label")
            if label:
                return normalize_label(str(label))

        for key in ("name", "label", "objectType"):
            if properties.get(key):
                return normalize_label(str(properties[key]))

    for key in ("name", "label", "objectType"):
        if feature.get(key):
            return normalize_label(str(feature[key]))

    return ""


def geometry_to_polygons(geometry: Dict[str, object]) -> List[List[List[Tuple[float, float]]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates") or []

    if geometry_type == "Polygon":
        coordinate_groups = [coordinates]
    elif geometry_type == "MultiPolygon":
        coordinate_groups = coordinates
    else:
        coordinate_groups = []

    polygons = []
    for polygon in coordinate_groups:
        rings = []
        for ring in polygon or []:
            if len(ring) < 3:
                continue
            rings.append([(float(point[0]), float(point[1])) for point in ring])
        if rings:
            polygons.append(rings)
    return polygons


def ring_area_and_centroid(ring: Sequence[Tuple[float, float]]) -> Tuple[float, Tuple[float, float]]:
    if len(ring) < 3:
        if not ring:
            return 0.0, (0.0, 0.0)
        xs, ys = zip(*ring)
        return 0.0, (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))

    area_twice = 0.0
    cx = 0.0
    cy = 0.0
    for index, (x0, y0) in enumerate(ring):
        x1, y1 = ring[(index + 1) % len(ring)]
        cross = x0 * y1 - x1 * y0
        area_twice += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    if abs(area_twice) < 1e-6:
        xs, ys = zip(*ring)
        return 0.0, (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))

    area = area_twice / 2.0
    factor = 1.0 / (3.0 * area_twice)
    return abs(area), (cx * factor, cy * factor)


def polygon_bbox(polygons: Sequence[Sequence[Sequence[Tuple[float, float]]]]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for polygon in polygons:
        for ring in polygon:
            for x, y in ring:
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def polygon_centroid(polygons: Sequence[Sequence[Sequence[Tuple[float, float]]]]) -> Tuple[float, float]:
    weighted_x = 0.0
    weighted_y = 0.0
    total_area = 0.0
    for polygon in polygons:
        if not polygon:
            continue
        area, centroid = ring_area_and_centroid(polygon[0])
        if area <= 0:
            continue
        weighted_x += centroid[0] * area
        weighted_y += centroid[1] * area
        total_area += area

    if total_area <= 0:
        bbox = polygon_bbox(polygons)
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
    return (weighted_x / total_area, weighted_y / total_area)


def polygon_total_area(polygons: Sequence[Sequence[Sequence[Tuple[float, float]]]]) -> float:
    total = 0.0
    for polygon in polygons:
        if not polygon:
            continue
        outer_area, _ = ring_area_and_centroid(polygon[0])
        hole_area = sum(ring_area_and_centroid(ring)[0] for ring in polygon[1:])
        total += max(outer_area - hole_area, 0.0)
    return total


def point_in_ring(point: Tuple[float, float], ring: Sequence[Tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    for index, (x0, y0) in enumerate(ring):
        x1, y1 = ring[(index + 1) % len(ring)]
        intersects = ((y0 > y) != (y1 > y)) and (
            x < (x1 - x0) * (y - y0) / max((y1 - y0), 1e-12) + x0
        )
        if intersects:
            inside = not inside
    return inside


def point_in_polygon(point: Tuple[float, float], polygon: Sequence[Sequence[Tuple[float, float]]]) -> bool:
    if not polygon or not polygon[0] or not point_in_ring(point, polygon[0]):
        return False
    for hole in polygon[1:]:
        if hole and point_in_ring(point, hole):
            return False
    return True


def bbox_contains_point(bbox: Tuple[float, float, float, float], point: Tuple[float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    x, y = point
    return x0 <= x <= x1 and y0 <= y <= y1


def bboxes_intersect(
    bbox_a: Tuple[float, float, float, float],
    bbox_b: Tuple[float, float, float, float],
) -> bool:
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def point_in_any_polygon(point: Tuple[float, float], records: Sequence[PolygonRecord]) -> bool:
    for record in records:
        if not bbox_contains_point(record.bbox, point):
            continue
        for polygon in record.polygons:
            if point_in_polygon(point, polygon):
                return True
    return False


def load_polygon_records(
    json_path: Path,
    target_labels: Optional[Iterable[str]] = None,
    start_id: int = 1,
) -> List[PolygonRecord]:
    label_set = {normalize_label(label) for label in target_labels} if target_labels else None
    records = []
    for feature in read_geojson_features(json_path):
        label = get_feature_label(feature)
        if label_set and label not in label_set:
            continue
        polygons = geometry_to_polygons(feature.get("geometry") or {})
        if not polygons:
            continue
        record_id = start_id + len(records)
        records.append(
            PolygonRecord(
                record_id=record_id,
                label=label,
                polygons=polygons,
                bbox=polygon_bbox(polygons),
                centroid=polygon_centroid(polygons),
                area=polygon_total_area(polygons),
            )
        )
    return records


def filter_records_by_roi(
    records: Sequence[PolygonRecord],
    roi_records: Sequence[PolygonRecord],
) -> List[PolygonRecord]:
    if not roi_records:
        return list(records)
    return [record for record in records if point_in_any_polygon(record.centroid, roi_records)]


def find_slide_ids(images_dir: Path, limit_slides: int = 0) -> List[str]:
    slide_ids = sorted(path.stem for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if limit_slides > 0:
        return slide_ids[:limit_slides]
    return slide_ids


def find_existing_path(base_dir: Path, slide_id: str, suffixes: Sequence[str]) -> Optional[Path]:
    for suffix in suffixes:
        candidate = base_dir / f"{slide_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def infer_channel_axis(shape: Sequence[int]) -> Optional[int]:
    if len(shape) < 3:
        return None
    explicit_axes = [idx for idx, size in enumerate(shape) if size in (1, 3, 4)]
    if explicit_axes:
        if shape[-1] in (1, 3, 4):
            return len(shape) - 1
        if shape[0] in (1, 3, 4):
            return 0
        return explicit_axes[0]
    ranked_axes = sorted(range(len(shape)), key=lambda idx: shape[idx])
    smallest_axis = ranked_axes[0]
    second_smallest_axis = ranked_axes[1]
    if shape[smallest_axis] <= 16 or shape[smallest_axis] * 8 < shape[second_smallest_axis]:
        return smallest_axis
    return None


def normalize_slide_array(slide: np.ndarray) -> np.ndarray:
    array = np.asarray(slide)
    raw_shape = tuple(int(size) for size in array.shape)
    array = np.squeeze(array)
    if array.ndim == 0:
        array = array.reshape((1, 1))

    while array.ndim > 3:
        channel_axis = infer_channel_axis(array.shape)
        spatial_axes = set(sorted(range(array.ndim), key=lambda idx: array.shape[idx])[-2:])
        removable_axes = [idx for idx in range(array.ndim) if idx not in spatial_axes and idx != channel_axis]
        if not removable_axes:
            removable_axes = [idx for idx in range(array.ndim) if idx != channel_axis]
        axis_to_slice = min(removable_axes, key=lambda idx: array.shape[idx])
        array = np.take(array, indices=0, axis=axis_to_slice)
        array = np.squeeze(array)

    if array.ndim == 1:
        array = array[np.newaxis, :, np.newaxis]
    elif array.ndim == 2:
        array = array[..., np.newaxis]
    elif array.ndim == 3:
        channel_axis = infer_channel_axis(array.shape)
        if channel_axis is not None and channel_axis != 2:
            array = np.moveaxis(array, channel_axis, -1)
    else:
        raise ValueError(f"Unsupported slide array shape {raw_shape}")

    normalized_shape = tuple(int(size) for size in array.shape)
    if normalized_shape != raw_shape:
        logging.info("Normalized slide array shape from %s to %s", raw_shape, normalized_shape)
    return array


def open_slide_array(image_path: Path) -> np.ndarray:
    if tifffile is not None:
        try:
            slide = tifffile.memmap(str(image_path))
        except Exception:
            slide = tifffile.imread(str(image_path))
    else:
        slide = np.asarray(Image.open(image_path))
    return normalize_slide_array(slide)


def ensure_uint8_rgb(tile: np.ndarray) -> np.ndarray:
    array = np.asarray(tile)
    if array.ndim == 2:
        array = np.repeat(array[..., np.newaxis], 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] > 3:
        array = array[..., :3]

    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        max_value = float(array.max()) if array.size else 255.0
        scale = 255.0 / max(max_value, 1.0)
        array = np.clip(array.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return array


def estimate_tissue_coverage(tile: np.ndarray, white_threshold: float) -> float:
    tile_rgb = ensure_uint8_rgb(tile)
    grayscale = tile_rgb.mean(axis=2)
    return float((grayscale < white_threshold).mean())


def generate_positions(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last_position = length - tile_size
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions


def _intersecting_grid_positions(
    positions: Sequence[int],
    bbox_start: float,
    bbox_end: float,
    tile_size: int,
) -> List[int]:
    positions_list = positions if isinstance(positions, list) else list(positions)
    min_origin = int(np.floor(bbox_start - tile_size + 1))
    max_origin = int(np.ceil(bbox_end) - 1)
    lo = bisect_left(positions_list, min_origin)
    hi = bisect_right(positions_list, max_origin)
    return [int(value) for value in positions_list[lo:hi]]


def expand_tile_candidates(
    candidates: Sequence[Tuple[int, int]],
    width: int,
    height: int,
    tile_size: int,
    stride: int,
    radius: int = 0,
) -> List[Tuple[int, int]]:
    if radius <= 0 or not candidates:
        return sorted({(int(x), int(y)) for x, y in candidates}, key=lambda item: (item[1], item[0]))

    x_positions = generate_positions(width, tile_size, stride)
    y_positions = generate_positions(height, tile_size, stride)
    x_to_index = {int(value): index for index, value in enumerate(x_positions)}
    y_to_index = {int(value): index for index, value in enumerate(y_positions)}

    expanded = set()
    for x, y in candidates:
        x_index = x_to_index[int(x)]
        y_index = y_to_index[int(y)]
        for dx in range(-radius, radius + 1):
            nx_index = x_index + dx
            if nx_index < 0 or nx_index >= len(x_positions):
                continue
            for dy in range(-radius, radius + 1):
                ny_index = y_index + dy
                if ny_index < 0 or ny_index >= len(y_positions):
                    continue
                expanded.add((int(x_positions[nx_index]), int(y_positions[ny_index])))

    return sorted(expanded, key=lambda item: (item[1], item[0]))


def generate_candidate_tiles_for_boxes(
    boxes: Sequence[Tuple[float, float, float, float]],
    width: int,
    height: int,
    tile_size: int,
    stride: int,
    expand_radius: int = 0,
) -> List[Tuple[int, int]]:
    if not boxes:
        return []

    x_positions = generate_positions(width, tile_size, stride)
    y_positions = generate_positions(height, tile_size, stride)
    candidates = set()

    for bbox in boxes:
        x0, y0, x1, y1 = bbox
        candidate_x = _intersecting_grid_positions(x_positions, x0, x1, tile_size)
        candidate_y = _intersecting_grid_positions(y_positions, y0, y1, tile_size)
        for x in candidate_x:
            for y in candidate_y:
                candidates.add((int(x), int(y)))

    return expand_tile_candidates(
        sorted(candidates, key=lambda item: (item[1], item[0])),
        width=width,
        height=height,
        tile_size=tile_size,
        stride=stride,
        radius=expand_radius,
    )


def generate_candidate_tiles_for_records(
    records: Sequence[PolygonRecord],
    width: int,
    height: int,
    tile_size: int,
    stride: int,
    expand_radius: int = 0,
) -> List[Tuple[int, int]]:
    return generate_candidate_tiles_for_boxes(
        [record.bbox for record in records],
        width=width,
        height=height,
        tile_size=tile_size,
        stride=stride,
        expand_radius=expand_radius,
    )


def pad_image_tile(tile: np.ndarray, tile_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    tile = ensure_uint8_rgb(tile)
    height, width = tile.shape[:2]
    if height == tile_size and width == tile_size:
        return tile, (height, width)
    padded = np.full((tile_size, tile_size, tile.shape[2]), 255, dtype=np.uint8)
    padded[:height, :width] = tile
    return padded, (height, width)


def resize_image(image: np.ndarray, output_size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize((output_size, output_size), resample=Image.BICUBIC))


def resize_mask(mask: np.ndarray, output_size: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint16), mode="I;16")
    return np.asarray(image.resize((output_size, output_size), resample=Image.NEAREST)).astype(np.uint16)


def extract_tile(slide_array: np.ndarray, x: int, y: int, tile_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    crop = slide_array[y : y + tile_size, x : x + tile_size]
    return pad_image_tile(crop, tile_size)


def translate_polygon(
    polygon: Sequence[Sequence[Tuple[float, float]]],
    offset_x: float,
    offset_y: float,
) -> List[List[Tuple[int, int]]]:
    translated = []
    for ring in polygon:
        translated.append([(int(round(x - offset_x)), int(round(y - offset_y))) for x, y in ring])
    return translated


def rasterize_records_to_mask(
    records: Sequence[PolygonRecord],
    tile_bbox: Tuple[int, int, int, int],
    tile_size: int,
    value_lookup: Optional[Dict[int, int]] = None,
) -> np.ndarray:
    image = Image.new("I", (tile_size, tile_size), 0)
    draw = ImageDraw.Draw(image)
    tile_x0, tile_y0, _, _ = tile_bbox
    for record in records:
        value = value_lookup[record.record_id] if value_lookup is not None else 1
        for polygon in record.polygons:
            translated = translate_polygon(polygon, tile_x0, tile_y0)
            outer_ring = translated[0] if translated else []
            if len(outer_ring) >= 3:
                draw.polygon(outer_ring, outline=int(value), fill=int(value))
            for hole in translated[1:]:
                if len(hole) >= 3:
                    draw.polygon(hole, outline=0, fill=0)
    return np.asarray(image).astype(np.uint16)


def build_palette() -> List[int]:
    palette = list(DEFAULT_PALETTE)
    if len(palette) < 768:
        palette.extend([0] * (768 - len(palette)))
    return palette[:768]


def save_palette_png(mask: np.ndarray, output_path: Path) -> None:
    mask_uint8 = np.clip(mask, 0, 255).astype(np.uint8)
    image = Image.fromarray(mask_uint8, mode="P")
    image.putpalette(build_palette())
    image.save(output_path)


def save_image_png(image: np.ndarray, output_path: Path) -> None:
    Image.fromarray(ensure_uint8_rgb(image)).save(output_path)


def load_mask_array(mask_path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def _connected_components(mask: np.ndarray) -> Tuple[int, np.ndarray]:
    binary_mask = (mask > 0).astype(np.uint8)
    if binary_mask.max() == 0:
        return 1, np.zeros_like(binary_mask, dtype=np.int32)

    if cv2 is not None:
        num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
        return int(num_labels), labels.astype(np.int32)

    height, width = binary_mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    current_label = 0
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(height):
        for x in range(width):
            if binary_mask[y, x] == 0 or labels[y, x] != 0:
                continue
            current_label += 1
            queue = [(y, x)]
            labels[y, x] = current_label
            while queue:
                cy, cx = queue.pop()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if binary_mask[ny, nx] == 0 or labels[ny, nx] != 0:
                        continue
                    labels[ny, nx] = current_label
                    queue.append((ny, nx))
    return current_label + 1, labels


def remap_mask_to_instance_map(
    mask: np.ndarray,
    mask_format: str = "auto",
    min_instance_area: int = 32,
    max_instances: int = 254,
) -> Tuple[np.ndarray, str]:
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]

    unique_values = [int(value) for value in np.unique(mask_array) if int(value) > 0]
    if not unique_values:
        return np.zeros(mask_array.shape, dtype=np.uint16), "empty"

    inferred_format = mask_format
    if mask_format == "auto":
        inferred_format = "binary" if len(unique_values) <= 1 else "instance"

    instance_map = np.zeros(mask_array.shape, dtype=np.uint16)
    next_instance_id = 1

    if inferred_format == "binary":
        _, labels = _connected_components(mask_array > 0)
        component_ids = [int(value) for value in np.unique(labels) if int(value) > 0]
        for component_id in component_ids:
            component_mask = labels == component_id
            if int(component_mask.sum()) < min_instance_area:
                continue
            if next_instance_id > max_instances:
                break
            instance_map[component_mask] = next_instance_id
            next_instance_id += 1
        return instance_map, "binary"

    for value in unique_values:
        _, labels = _connected_components(mask_array == value)
        component_ids = [int(component_id) for component_id in np.unique(labels) if int(component_id) > 0]
        for component_id in component_ids:
            component_mask = labels == component_id
            if int(component_mask.sum()) < min_instance_area:
                continue
            if next_instance_id > max_instances:
                break
            instance_map[component_mask] = next_instance_id
            next_instance_id += 1
        if next_instance_id > max_instances:
            break
    return instance_map, "instance"


def load_mask_value_cache(mask_dir: Path) -> Optional[Dict[str, object]]:
    cache_path = mask_dir / ".mask_values_cache.json"
    if not cache_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8") as handle:
        cache = json.load(handle)
    return cache if isinstance(cache, dict) else None


def infer_mask_format_from_cache(mask_dir: Path) -> Optional[str]:
    cache = load_mask_value_cache(mask_dir)
    if cache is None:
        return None
    values = [int(value) for value in cache.get("mask_values", [])]
    non_zero_values = [value for value in values if value > 0]
    if not non_zero_values:
        return "binary"
    return "binary" if len(non_zero_values) <= 1 else "instance"


def link_or_copy_file(source_path: Path, target_path: Path, mode: str = "auto") -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()

    modes = [mode]
    if mode == "auto":
        modes = ["hardlink", "copy"]

    last_error = None
    for current_mode in modes:
        try:
            if current_mode == "hardlink":
                os.link(source_path, target_path)
            elif current_mode == "symlink":
                os.symlink(source_path, target_path)
            elif current_mode == "copy":
                shutil.copy2(source_path, target_path)
            else:
                raise ValueError(f"Unsupported link mode: {current_mode}")
            return current_mode
        except OSError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Could not materialize {source_path} to {target_path}")


def assign_slide_splits(
    slide_ids: Sequence[str],
    split_csv: Optional[Path],
    val_ratio: float,
    seed: int,
) -> Dict[str, str]:
    if split_csv is not None and split_csv.exists():
        split_map: Dict[str, str] = {}
        with split_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                slide_id = row.get("slide_id") or row.get("id")
                split = normalize_label(row.get("split"))
                if slide_id and split in {"train", "val"}:
                    split_map[slide_id] = split
        return {slide_id: split_map.get(slide_id, "train") for slide_id in slide_ids}

    if val_ratio <= 0 or len(slide_ids) < 2:
        return {slide_id: "train" for slide_id in slide_ids}

    rng = random.Random(seed)
    shuffled = list(slide_ids)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_slides = set(shuffled[:val_count])
    return {slide_id: ("val" if slide_id in val_slides else "train") for slide_id in slide_ids}


def write_json(data: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def discover_prepared_samples(dataset_root: Path, split: str) -> List[PreparedSample]:
    image_root = dataset_root / split / "JPEGImages"
    annotation_root = dataset_root / split / "Annotations"
    metadata_root = dataset_root / split / "Metadata"
    if not image_root.exists():
        return []

    samples = []
    for sample_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        sample_id = sample_dir.name
        image_path = None
        for image_name in PREPARED_IMAGE_CANDIDATES:
            candidate = sample_dir / image_name
            if candidate.exists():
                image_path = candidate
                break
        annotation_path = annotation_root / sample_id / PREPARED_MASK_NAME
        metadata_path = metadata_root / f"{sample_id}.json"
        if image_path is not None and annotation_path.exists() and metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            samples.append(
                PreparedSample(
                    sample_id=sample_id,
                    split=split,
                    slide_id=str(metadata.get("slide_id", "")),
                    image_path=image_path,
                    annotation_path=annotation_path,
                    metadata_path=metadata_path,
                )
            )
    return samples


def load_prepared_sample(sample: PreparedSample) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    image = np.array(Image.open(sample.image_path).convert("RGB"), copy=True)
    annotation = np.array(Image.open(sample.annotation_path), copy=True)
    with sample.metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return image, annotation.astype(np.int32), metadata
