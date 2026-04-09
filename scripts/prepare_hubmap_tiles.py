import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

try:
    import tifffile
except ImportError:
    tifffile = None


IMAGE_SUFFIXES = ('.tif', '.tiff')
MASK_SUFFIXES = ('.png', '.tif', '.tiff')
ANNOTATION_KEYS = ('encoding', 'rle', 'segmentation')
AVAILABLE_ANNOTATION_FORMATS = ('auto', 'csv-rle', 'mask', 'json-polygons')
AVAILABLE_MISSING_ROI_POLICIES = ('skip-slide', 'ignore-roi', 'error')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare HuBMAP training tiles from TIFF slides and annotations.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--images-dir', type=str, required=True, help='Directory containing HuBMAP TIFF slides')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory where the tiled dataset will be written')
    parser.add_argument('--annotation-format', choices=AVAILABLE_ANNOTATION_FORMATS, default='auto',
                        help='Annotation source format. auto detects mask, CSV-RLE, or JSON polygons')
    parser.add_argument('--annotations-csv', type=str, default='',
                        help='CSV file containing slide ids and RLE masks')
    parser.add_argument('--annotations-dir', type=str, default='',
                        help='Directory containing per-slide annotation JSON files. Defaults to images-dir')
    parser.add_argument('--annotation-json-suffix', type=str, default='.json',
                        help='Suffix for glomerulus annotation JSON files')
    parser.add_argument('--target-labels', nargs='+', default=['glomerulus'],
                        help='Annotation labels that should be rasterized into the positive mask')
    parser.add_argument('--mask-dir', type=str, default='',
                        help='Optional directory containing pre-rendered masks. Overrides other annotation sources')
    parser.add_argument('--anatomical-dir', type=str, default='',
                        help='Directory containing anatomical structure JSON files. Defaults to images-dir')
    parser.add_argument('--anatomical-json-suffix', type=str, default='-anatomical-structure.json',
                        help='Suffix for anatomical ROI JSON files')
    parser.add_argument('--roi-labels', nargs='*', default=[],
                        help='Optional anatomical labels to keep, for example Cortex')
    parser.add_argument('--min-roi-coverage', type=float, default=0.05,
                        help='Minimum anatomical ROI coverage required to keep a tile when roi-labels are used')
    parser.add_argument('--missing-roi-policy', choices=AVAILABLE_MISSING_ROI_POLICIES, default='skip-slide',
                        help='What to do when a slide does not contain the requested ROI labels')
    parser.add_argument('--tile-size', type=int, default=1024, help='Tile size before optional resizing')
    parser.add_argument('--stride', type=int, default=1024, help='Sliding-window stride')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='Optional scale factor applied before saving each tile')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation slide ratio used when no split file is provided')
    parser.add_argument('--split-csv', type=str, default='',
                        help='Optional CSV with columns slide_id,split to control train/val assignment')
    parser.add_argument('--min-tissue-coverage', type=float, default=0.05,
                        help='Minimum non-white tissue coverage required to keep a tile')
    parser.add_argument('--white-threshold', type=float, default=230.0,
                        help='Mean intensity threshold used to estimate white background pixels')
    parser.add_argument('--min-positive-pixels', type=int, default=64,
                        help='Minimum positive mask pixels required to force-keep a tile')
    parser.add_argument('--min-mask-coverage', type=float, default=0.0,
                        help='Minimum positive mask coverage used to mark a tile as positive')
    parser.add_argument('--negative-ratio', type=float, default=2.0,
                        help='How many negative tiles to keep per positive tile')
    parser.add_argument('--max-background-tiles-per-slide', type=int, default=200,
                        help='Negative tile cap for slides without positive tiles')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for slide splitting and negative sampling')
    parser.add_argument('--limit-slides', type=int, default=0,
                        help='Optional cap on processed slide count for quick experiments')
    return parser.parse_args()


def require_tifffile():
    if tifffile is None:
        raise ImportError(
            'tifffile is required for HuBMAP preprocessing. Install dependencies with '
            '`pip install -r requirements.txt`.'
        )


def read_annotation_table(csv_path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if not csv_path:
        return {}

    annotations = {}
    with csv_path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            slide_id = row.get('id') or row.get('slide_id') or row.get('image_id')
            if not slide_id:
                continue

            encoding = ''
            for key in ANNOTATION_KEYS:
                if key in row:
                    encoding = row.get(key, '') or ''
                    break

            annotations[slide_id] = {
                'encoding': encoding.strip(),
                'row': row,
            }
    return annotations


def read_split_table(split_csv: Optional[Path]) -> Dict[str, str]:
    if not split_csv:
        return {}

    split_map = {}
    with split_csv.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            slide_id = row.get('slide_id') or row.get('id')
            split = (row.get('split') or '').strip().lower()
            if slide_id and split in ('train', 'val'):
                split_map[slide_id] = split
    return split_map


def discover_slide_ids(images_dir: Path, limit_slides: int, annotations: Dict[str, Dict[str, object]], annotation_format: str):
    image_ids = [path.stem for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
    if annotation_format == 'csv-rle' and annotations:
        slide_ids = [slide_id for slide_id in image_ids if slide_id in annotations]
    else:
        slide_ids = image_ids

    slide_ids = sorted(slide_ids)
    if limit_slides > 0:
        slide_ids = slide_ids[:limit_slides]
    return slide_ids


def assign_slide_splits(slide_ids: Sequence[str], split_map: Dict[str, str], val_ratio: float, seed: int):
    if split_map:
        assignments = {}
        for slide_id in slide_ids:
            assignments[slide_id] = split_map.get(slide_id, 'train')
        return assignments

    if val_ratio <= 0 or len(slide_ids) < 2:
        return {slide_id: 'train' for slide_id in slide_ids}

    rng = random.Random(seed)
    shuffled = list(slide_ids)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_slides = set(shuffled[:val_count])
    return {slide_id: ('val' if slide_id in val_slides else 'train') for slide_id in slide_ids}


def find_image_path(images_dir: Path, slide_id: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / '{}{}'.format(slide_id, suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError('Could not find TIFF slide for id {}'.format(slide_id))


def find_mask_path(mask_dir: Path, slide_id: str) -> Optional[Path]:
    for suffix in MASK_SUFFIXES:
        candidate = mask_dir / '{}{}'.format(slide_id, suffix)
        if candidate.exists():
            return candidate
    return None


def find_json_path(base_dir: Path, slide_id: str, suffix: str) -> Optional[Path]:
    candidate = base_dir / '{}{}'.format(slide_id, suffix)
    if candidate.exists():
        return candidate
    return None


def open_slide_array(image_path: Path) -> np.ndarray:
    require_tifffile()
    try:
        slide = tifffile.memmap(str(image_path))
    except Exception:
        slide = tifffile.imread(str(image_path))

    if slide.ndim == 2:
        slide = slide[..., np.newaxis]
    elif slide.ndim == 3 and slide.shape[0] in (3, 4) and slide.shape[-1] not in (3, 4):
        slide = np.moveaxis(slide, 0, -1)

    return slide


def read_geojson_features(json_path: Path) -> List[Dict[str, object]]:
    with json_path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        if isinstance(data.get('features'), list):
            return [item for item in data['features'] if isinstance(item, dict)]
        return [data]

    raise ValueError('Unsupported JSON annotation structure in {}'.format(json_path))


def normalize_label(label: Optional[str]) -> str:
    return (label or '').strip().lower()


def get_feature_label(feature: Dict[str, object]) -> str:
    properties = feature.get('properties') or {}
    if isinstance(properties, dict):
        classification = properties.get('classification') or {}
        if isinstance(classification, dict):
            label = classification.get('name') or classification.get('label')
            if label:
                return normalize_label(str(label))

        for key in ('name', 'label', 'objectType'):
            if key in properties and properties[key]:
                return normalize_label(str(properties[key]))

    for key in ('name', 'label', 'objectType'):
        if key in feature and feature[key]:
            return normalize_label(str(feature[key]))

    return ''


def point_to_tuple(point) -> tuple:
    return (int(round(point[0])), int(round(point[1])))


def polygon_rings_from_geometry(geometry: Dict[str, object]) -> List[List[List[float]]]:
    geometry_type = geometry.get('type')
    coordinates = geometry.get('coordinates') or []

    if geometry_type == 'Polygon':
        return [coordinates]

    if geometry_type == 'MultiPolygon':
        return coordinates

    return []


def rasterize_features(image_shape, features: Sequence[Dict[str, object]], target_labels: Optional[Iterable[str]]) -> np.ndarray:
    height, width = image_shape[:2]
    target_label_set = {normalize_label(label) for label in target_labels} if target_labels else set()
    # Mode "1" uses a 1-bit bitmap, which is much lighter than an 8-bit full-slide mask.
    mask_image = Image.new('1', (width, height), 0)
    draw = ImageDraw.Draw(mask_image)

    for feature in features:
        label = get_feature_label(feature)
        if target_label_set and label not in target_label_set:
            continue

        geometry = feature.get('geometry') or {}
        polygons = polygon_rings_from_geometry(geometry)
        for polygon in polygons:
            if not polygon:
                continue

            outer_ring = polygon[0] if polygon else []
            if len(outer_ring) >= 3:
                draw.polygon([point_to_tuple(point) for point in outer_ring], outline=1, fill=1)

            for hole in polygon[1:]:
                if len(hole) >= 3:
                    draw.polygon([point_to_tuple(point) for point in hole], outline=0, fill=0)

    return (np.asarray(mask_image) > 0).astype(np.uint8)


def load_mask_from_mask_dir(mask_dir: Path, slide_id: str) -> np.ndarray:
    mask_path = find_mask_path(mask_dir, slide_id)
    if mask_path is None:
        raise FileNotFoundError('No rendered mask found for slide {}'.format(slide_id))

    if mask_path.suffix.lower() in ('.tif', '.tiff'):
        require_tifffile()
        mask = tifffile.imread(str(mask_path))
    else:
        mask = np.asarray(Image.open(mask_path))

    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def decode_rle_mask(rle: str, image_shape) -> np.ndarray:
    height, width = image_shape
    if not rle:
        return np.zeros((height, width), dtype=np.uint8)

    mask = np.zeros(height * width, dtype=np.uint8)
    numbers = np.asarray([int(part) for part in rle.split()], dtype=np.int64)
    starts = numbers[0::2] - 1
    lengths = numbers[1::2]
    ends = starts + lengths

    for start, end in zip(starts, ends):
        mask[start:end] = 1

    return mask.reshape((height, width), order='F')


def resolve_annotation_format(
    requested_format: str,
    slide_id: str,
    annotations_csv: Dict[str, Dict[str, object]],
    mask_dir: Optional[Path],
    annotations_dir: Path,
    annotation_json_suffix: str,
) -> str:
    if mask_dir and find_mask_path(mask_dir, slide_id) is not None:
        return 'mask'

    if requested_format != 'auto':
        return requested_format

    if slide_id in annotations_csv:
        return 'csv-rle'

    if find_json_path(annotations_dir, slide_id, annotation_json_suffix) is not None:
        return 'json-polygons'

    raise FileNotFoundError('Could not infer annotation source for slide {}'.format(slide_id))


def load_mask(
    slide_id: str,
    image_shape,
    annotation_format: str,
    annotations_csv: Dict[str, Dict[str, object]],
    annotations_dir: Path,
    annotation_json_suffix: str,
    target_labels: Sequence[str],
    mask_dir: Optional[Path],
) -> np.ndarray:
    if annotation_format == 'mask':
        if mask_dir is None:
            raise ValueError('mask annotation format requires --mask-dir')
        return load_mask_from_mask_dir(mask_dir, slide_id)

    if annotation_format == 'csv-rle':
        if slide_id not in annotations_csv:
            raise FileNotFoundError('No RLE annotation found for slide {}'.format(slide_id))
        encoding = annotations_csv[slide_id]['encoding']
        return decode_rle_mask(str(encoding), image_shape[:2])

    if annotation_format == 'json-polygons':
        annotation_path = find_json_path(annotations_dir, slide_id, annotation_json_suffix)
        if annotation_path is None:
            raise FileNotFoundError('No annotation JSON found for slide {}'.format(slide_id))
        logging.info('Reading glomerulus annotations from %s', annotation_path)
        features = read_geojson_features(annotation_path)
        logging.info('Loaded %s annotation features for slide %s', len(features), slide_id)
        return rasterize_features(image_shape, features, target_labels=target_labels)

    raise ValueError('Unsupported annotation format {}'.format(annotation_format))


def load_roi_mask(
    slide_id: str,
    image_shape,
    anatomical_dir: Path,
    anatomical_json_suffix: str,
    roi_labels: Sequence[str],
    missing_roi_policy: str,
) -> Optional[np.ndarray]:
    if not roi_labels:
        return None

    anatomical_path = find_json_path(anatomical_dir, slide_id, anatomical_json_suffix)
    if anatomical_path is None:
        message = 'No anatomical JSON found for slide {}'.format(slide_id)
        if missing_roi_policy == 'ignore-roi':
            logging.warning('%s. Continuing without ROI filtering for this slide.', message)
            return None
        if missing_roi_policy == 'skip-slide':
            logging.warning('%s. Skipping this slide.', message)
            return None
        raise FileNotFoundError(message)

    logging.info('Reading anatomical annotations from %s', anatomical_path)
    features = read_geojson_features(anatomical_path)
    logging.info('Loaded %s anatomical features for slide %s', len(features), slide_id)
    roi_mask = rasterize_features(image_shape, features, target_labels=roi_labels)
    if roi_mask.max() == 0:
        message = 'Anatomical JSON {} did not contain any ROI labels matching {}'.format(
            anatomical_path,
            ', '.join(roi_labels),
        )
        if missing_roi_policy == 'ignore-roi':
            logging.warning('%s. Continuing without ROI filtering for this slide.', message)
            return None
        if missing_roi_policy == 'skip-slide':
            logging.warning('%s. Skipping this slide.', message)
            return None
        raise ValueError(message)
    return roi_mask


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
    tissue_mask = grayscale < white_threshold
    return float(tissue_mask.mean())


def generate_positions(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))
    last_position = length - tile_size
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions


def extract_crop(array: np.ndarray, x: int, y: int, tile_size: int):
    return array[y:y + tile_size, x:x + tile_size]


def pad_image_tile(tile: np.ndarray, tile_size: int) -> np.ndarray:
    tile = ensure_uint8_rgb(tile)
    height, width = tile.shape[:2]
    if height == tile_size and width == tile_size:
        return tile

    padded = np.full((tile_size, tile_size, tile.shape[2]), 255, dtype=np.uint8)
    padded[:height, :width] = tile
    return padded


def pad_mask_tile(mask_tile: np.ndarray, tile_size: int) -> np.ndarray:
    height, width = mask_tile.shape[:2]
    if height == tile_size and width == tile_size:
        return mask_tile.astype(np.uint8)

    padded = np.zeros((tile_size, tile_size), dtype=np.uint8)
    padded[:height, :width] = mask_tile.astype(np.uint8)
    return padded


def resize_tile_pair(image_tile: np.ndarray, mask_tile: np.ndarray, downsample: float):
    if downsample == 1.0:
        return image_tile, mask_tile

    output_size = int(round(image_tile.shape[0] * downsample))
    output_size = max(output_size, 1)

    image = Image.fromarray(image_tile)
    mask = Image.fromarray((mask_tile.astype(np.uint8) * 255))
    image = image.resize((output_size, output_size), resample=Image.BICUBIC)
    mask = mask.resize((output_size, output_size), resample=Image.NEAREST)
    return np.asarray(image), (np.asarray(mask) > 0).astype(np.uint8)


def collect_tile_records(slide_array, mask_array, roi_mask, slide_id: str, split: str, args):
    height, width = mask_array.shape
    positive_tiles = []
    negative_tiles = []
    y_positions = generate_positions(height, args.tile_size, args.stride)
    x_positions = generate_positions(width, args.tile_size, args.stride)

    for y in y_positions:
        for x in x_positions:
            image_crop = extract_crop(slide_array, x, y, args.tile_size)
            mask_crop = extract_crop(mask_array, x, y, args.tile_size)
            roi_crop = extract_crop(roi_mask, x, y, args.tile_size) if roi_mask is not None else None

            tissue_coverage = estimate_tissue_coverage(image_crop, args.white_threshold)
            if tissue_coverage < args.min_tissue_coverage:
                continue

            roi_coverage = float(roi_crop.mean()) if roi_crop is not None else 1.0
            if roi_crop is not None and roi_coverage < args.min_roi_coverage:
                continue

            positive_pixels = int(mask_crop.sum())
            mask_coverage = positive_pixels / float(mask_crop.size)
            has_mask_coverage_match = args.min_mask_coverage > 0 and mask_coverage >= args.min_mask_coverage
            record = {
                'slide_id': slide_id,
                'split': split,
                'x': x,
                'y': y,
                'width': int(image_crop.shape[1]),
                'height': int(image_crop.shape[0]),
                'tissue_coverage': round(tissue_coverage, 6),
                'roi_coverage': round(roi_coverage, 6),
                'mask_coverage': round(mask_coverage, 6),
                'positive_pixels': positive_pixels,
                'is_positive': int(positive_pixels >= args.min_positive_pixels or has_mask_coverage_match),
            }

            if record['is_positive']:
                positive_tiles.append(record)
            else:
                negative_tiles.append(record)

    return positive_tiles, negative_tiles


def select_tiles(positive_tiles, negative_tiles, rng: random.Random, negative_ratio: float, max_background_tiles: int):
    if positive_tiles:
        negative_cap = int(math.ceil(len(positive_tiles) * max(negative_ratio, 0.0)))
    else:
        negative_cap = max(max_background_tiles, 0)

    if negative_cap > 0 and len(negative_tiles) > negative_cap:
        negative_tiles = rng.sample(negative_tiles, negative_cap)
    elif negative_cap == 0:
        negative_tiles = []

    selected_tiles = positive_tiles + negative_tiles
    selected_tiles.sort(key=lambda item: (item['y'], item['x']))
    return selected_tiles


def save_tiles(slide_array, mask_array, selected_tiles, output_dir: Path, tile_size: int, downsample: float):
    manifest_rows = []
    for record in selected_tiles:
        x = record['x']
        y = record['y']

        image_crop = extract_crop(slide_array, x, y, tile_size)
        mask_crop = extract_crop(mask_array, x, y, tile_size)

        image_tile = pad_image_tile(image_crop, tile_size)
        mask_tile = pad_mask_tile(mask_crop, tile_size)
        image_tile, mask_tile = resize_tile_pair(image_tile, mask_tile, downsample)

        tile_name = '{slide}_{x}_{y}'.format(
            slide=record['slide_id'],
            x=x,
            y=y,
        )

        image_path = output_dir / record['split'] / 'images' / '{}.jpg'.format(tile_name)
        mask_path = output_dir / record['split'] / 'masks' / '{}.png'.format(tile_name)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(image_tile).save(image_path, quality=95)
        Image.fromarray((mask_tile.astype(np.uint8) * 255)).save(mask_path)

        manifest_rows.append({
            'tile_id': tile_name,
            'slide_id': record['slide_id'],
            'split': record['split'],
            'x': record['x'],
            'y': record['y'],
            'tile_size': tile_size,
            'saved_size': image_tile.shape[0],
            'tissue_coverage': record['tissue_coverage'],
            'roi_coverage': record['roi_coverage'],
            'mask_coverage': record['mask_coverage'],
            'positive_pixels': record['positive_pixels'],
            'is_positive': record['is_positive'],
            'image_path': str(image_path.relative_to(output_dir)),
            'mask_path': str(mask_path.relative_to(output_dir)),
        })

    return manifest_rows


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text('', encoding='utf-8')
        return

    fieldnames = list(rows[0].keys())
    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    annotations_csv_path = Path(args.annotations_csv) if args.annotations_csv else None
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else images_dir
    anatomical_dir = Path(args.anatomical_dir) if args.anatomical_dir else images_dir
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    split_csv = Path(args.split_csv) if args.split_csv else None

    if args.tile_size <= 0 or args.stride <= 0:
        raise ValueError('tile-size and stride must both be positive integers.')
    if not (0 < args.downsample <= 1.0):
        raise ValueError('downsample must be in the interval (0, 1].')
    require_tifffile()

    annotations_csv = read_annotation_table(annotations_csv_path)
    split_map = read_split_table(split_csv)
    slide_ids = discover_slide_ids(
        images_dir=images_dir,
        limit_slides=args.limit_slides,
        annotations=annotations_csv,
        annotation_format=args.annotation_format,
    )
    slide_splits = assign_slide_splits(slide_ids, split_map, args.val_ratio, args.seed)
    rng = random.Random(args.seed)

    tile_manifest = []
    slide_manifest = []
    target_labels = [normalize_label(label) for label in args.target_labels]
    roi_labels = [normalize_label(label) for label in args.roi_labels]

    for slide_id in tqdm(slide_ids, desc='Slides', unit='slide'):
        split = slide_splits.get(slide_id, 'train')
        image_path = find_image_path(images_dir, slide_id)
        logging.info('Opening slide %s from %s', slide_id, image_path)
        slide_array = open_slide_array(image_path)
        logging.info(
            'Slide %s dimensions: width=%s, height=%s, channels=%s',
            slide_id,
            slide_array.shape[1],
            slide_array.shape[0],
            slide_array.shape[2] if slide_array.ndim == 3 else 1,
        )

        resolved_annotation_format = resolve_annotation_format(
            requested_format=args.annotation_format,
            slide_id=slide_id,
            annotations_csv=annotations_csv,
            mask_dir=mask_dir,
            annotations_dir=annotations_dir,
            annotation_json_suffix=args.annotation_json_suffix,
        )
        logging.info('Slide %s annotation format resolved to %s', slide_id, resolved_annotation_format)
        mask_array = load_mask(
            slide_id=slide_id,
            image_shape=slide_array.shape,
            annotation_format=resolved_annotation_format,
            annotations_csv=annotations_csv,
            annotations_dir=annotations_dir,
            annotation_json_suffix=args.annotation_json_suffix,
            target_labels=target_labels,
            mask_dir=mask_dir,
        )
        logging.info('Finished rasterizing glomerulus mask for slide %s', slide_id)
        roi_mask = load_roi_mask(
            slide_id=slide_id,
            image_shape=slide_array.shape,
            anatomical_dir=anatomical_dir,
            anatomical_json_suffix=args.anatomical_json_suffix,
            roi_labels=roi_labels,
            missing_roi_policy=args.missing_roi_policy,
        )
        if roi_labels and args.missing_roi_policy == 'skip-slide' and roi_mask is None:
            slide_manifest.append({
                'slide_id': slide_id,
                'split': split,
                'annotation_format': resolved_annotation_format,
                'width': int(slide_array.shape[1]),
                'height': int(slide_array.shape[0]),
                'positive_tiles': 0,
                'negative_tiles_kept': 0,
                'total_tiles_kept': 0,
                'status': 'skipped_missing_roi',
            })
            del mask_array
            del slide_array
            continue
        if roi_mask is not None:
            logging.info('Finished rasterizing ROI mask for slide %s', slide_id)

        positive_tiles, negative_tiles = collect_tile_records(
            slide_array=slide_array,
            mask_array=mask_array,
            roi_mask=roi_mask,
            slide_id=slide_id,
            split=split,
            args=args,
        )
        logging.info(
            'Collected tile candidates for %s: %s positive, %s negative',
            slide_id,
            len(positive_tiles),
            len(negative_tiles),
        )
        selected_tiles = select_tiles(
            positive_tiles=positive_tiles,
            negative_tiles=negative_tiles,
            rng=rng,
            negative_ratio=args.negative_ratio,
            max_background_tiles=args.max_background_tiles_per_slide,
        )
        logging.info('Selected %s tiles for slide %s', len(selected_tiles), slide_id)
        tile_rows = save_tiles(slide_array, mask_array, selected_tiles, output_dir, args.tile_size, args.downsample)
        tile_manifest.extend(tile_rows)

        slide_manifest.append({
            'slide_id': slide_id,
            'split': split,
            'annotation_format': resolved_annotation_format,
            'width': int(slide_array.shape[1]),
            'height': int(slide_array.shape[0]),
            'positive_tiles': len(positive_tiles),
            'negative_tiles_kept': sum(1 for row in tile_rows if not row['is_positive']),
            'total_tiles_kept': len(tile_rows),
            'status': 'processed',
        })

        logging.info(
            'Processed %s with %s: %s positives, %s negatives kept, split=%s',
            slide_id,
            resolved_annotation_format,
            len(positive_tiles),
            sum(1 for row in tile_rows if not row['is_positive']),
            split,
        )

        del mask_array
        if roi_mask is not None:
            del roi_mask

    manifests_dir = output_dir / 'manifests'
    write_csv(tile_manifest, manifests_dir / 'tiles.csv')
    write_csv(slide_manifest, manifests_dir / 'slides.csv')

    summary = {
        'slides': len(slide_ids),
        'annotation_format': args.annotation_format,
        'target_labels': target_labels,
        'roi_labels': roi_labels,
        'missing_roi_policy': args.missing_roi_policy,
        'train_tiles': sum(1 for row in tile_manifest if row.get('split') == 'train'),
        'val_tiles': sum(1 for row in tile_manifest if row.get('split') == 'val'),
        'positive_tiles': sum(int(row.get('is_positive', 0)) for row in tile_manifest),
        'negative_tiles': sum(1 for row in tile_manifest if not row.get('is_positive')),
        'processed_slides': sum(1 for row in slide_manifest if row.get('status') == 'processed'),
        'skipped_slides': sum(1 for row in slide_manifest if row.get('status') == 'skipped_missing_roi'),
        'tile_size': args.tile_size,
        'stride': args.stride,
        'downsample': args.downsample,
    }

    with (manifests_dir / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    logging.info('Finished. Tiled dataset written to %s', output_dir)
