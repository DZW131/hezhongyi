import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import tifffile
except ImportError:
    tifffile = None


IMAGE_SUFFIXES = ('.tif', '.tiff')
MASK_SUFFIXES = ('.png', '.tif', '.tiff')
ANNOTATION_KEYS = ('encoding', 'rle', 'segmentation')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Tile HuBMAP WSI TIFF files into train/val jpg/png patches for U-Net training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--images-dir', type=str, required=True, help='Directory containing HuBMAP TIFF slides')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory where tiled dataset will be written')
    parser.add_argument('--annotations-csv', type=str, default='',
                        help='CSV file containing slide ids and RLE masks')
    parser.add_argument('--mask-dir', type=str, default='',
                        help='Optional directory containing pre-rendered masks. Overrides RLE decoding when present')
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


def read_annotation_table(csv_path: Path):
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


def read_split_table(split_csv: Path):
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


def discover_slide_ids(images_dir: Path, annotations, limit_slides: int):
    image_ids = [path.stem for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
    if annotations:
        slide_ids = [slide_id for slide_id in image_ids if slide_id in annotations]
    else:
        slide_ids = image_ids

    slide_ids = sorted(slide_ids)
    if limit_slides > 0:
        slide_ids = slide_ids[:limit_slides]
    return slide_ids


def assign_slide_splits(slide_ids, split_map, val_ratio: float, seed: int):
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


def find_image_path(images_dir: Path, slide_id: str):
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / '{}{}'.format(slide_id, suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError('Could not find TIFF slide for id {}'.format(slide_id))


def find_mask_path(mask_dir: Path, slide_id: str):
    for suffix in MASK_SUFFIXES:
        candidate = mask_dir / '{}{}'.format(slide_id, suffix)
        if candidate.exists():
            return candidate
    return None


def open_slide_array(image_path: Path):
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


def load_mask(mask_dir: Path, annotations, slide_id: str, image_shape):
    if mask_dir:
        mask_path = find_mask_path(mask_dir, slide_id)
        if mask_path is not None:
            if mask_path.suffix.lower() in ('.tif', '.tiff'):
                require_tifffile()
                mask = tifffile.imread(str(mask_path))
            else:
                mask = np.asarray(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = (mask > 0).astype(np.uint8)
            return mask

    if slide_id not in annotations:
        raise FileNotFoundError('No annotation found for slide {}'.format(slide_id))

    encoding = annotations[slide_id]['encoding']
    return decode_rle_mask(encoding, image_shape[:2])


def decode_rle_mask(rle: str, image_shape):
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


def ensure_uint8_rgb(tile: np.ndarray):
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


def estimate_tissue_coverage(tile: np.ndarray, white_threshold: float):
    tile_rgb = ensure_uint8_rgb(tile)
    grayscale = tile_rgb.mean(axis=2)
    tissue_mask = grayscale < white_threshold
    return float(tissue_mask.mean())


def generate_positions(length: int, tile_size: int, stride: int):
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))
    last_position = length - tile_size
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions


def extract_crop(array: np.ndarray, x: int, y: int, tile_size: int):
    return array[y:y + tile_size, x:x + tile_size]


def pad_image_tile(tile: np.ndarray, tile_size: int):
    tile = ensure_uint8_rgb(tile)
    height, width = tile.shape[:2]
    if height == tile_size and width == tile_size:
        return tile

    padded = np.full((tile_size, tile_size, tile.shape[2]), 255, dtype=np.uint8)
    padded[:height, :width] = tile
    return padded


def pad_mask_tile(mask_tile: np.ndarray, tile_size: int):
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


def collect_tile_records(slide_array, mask_array, slide_id: str, split: str, args):
    height, width = mask_array.shape
    positive_tiles = []
    negative_tiles = []
    y_positions = generate_positions(height, args.tile_size, args.stride)
    x_positions = generate_positions(width, args.tile_size, args.stride)

    for y in y_positions:
        for x in x_positions:
            image_crop = extract_crop(slide_array, x, y, args.tile_size)
            mask_crop = extract_crop(mask_array, x, y, args.tile_size)
            tissue_coverage = estimate_tissue_coverage(image_crop, args.white_threshold)
            if tissue_coverage < args.min_tissue_coverage:
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

        tile_name = '{slide}__x{x:05d}_y{y:05d}_s{size}'.format(
            slide=record['slide_id'],
            x=x,
            y=y,
            size=tile_size,
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
    annotations_csv = Path(args.annotations_csv) if args.annotations_csv else None
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    split_csv = Path(args.split_csv) if args.split_csv else None

    if args.tile_size <= 0 or args.stride <= 0:
        raise ValueError('tile-size and stride must both be positive integers.')
    if not (0 < args.downsample <= 1.0):
        raise ValueError('downsample must be in the interval (0, 1].')
    require_tifffile()

    annotations = read_annotation_table(annotations_csv) if annotations_csv else {}
    split_map = read_split_table(split_csv) if split_csv else {}
    slide_ids = discover_slide_ids(images_dir, annotations, args.limit_slides)
    slide_splits = assign_slide_splits(slide_ids, split_map, args.val_ratio, args.seed)
    rng = random.Random(args.seed)

    tile_manifest = []
    slide_manifest = []

    for slide_id in tqdm(slide_ids, desc='Slides', unit='slide'):
        split = slide_splits.get(slide_id, 'train')
        image_path = find_image_path(images_dir, slide_id)
        slide_array = open_slide_array(image_path)
        mask_array = load_mask(mask_dir, annotations, slide_id, slide_array.shape)

        positive_tiles, negative_tiles = collect_tile_records(slide_array, mask_array, slide_id, split, args)
        selected_tiles = select_tiles(
            positive_tiles=positive_tiles,
            negative_tiles=negative_tiles,
            rng=rng,
            negative_ratio=args.negative_ratio,
            max_background_tiles=args.max_background_tiles_per_slide,
        )
        tile_rows = save_tiles(slide_array, mask_array, selected_tiles, output_dir, args.tile_size, args.downsample)
        tile_manifest.extend(tile_rows)

        slide_manifest.append({
            'slide_id': slide_id,
            'split': split,
            'width': int(slide_array.shape[1]),
            'height': int(slide_array.shape[0]),
            'positive_tiles': len(positive_tiles),
            'negative_tiles_kept': sum(1 for row in tile_rows if not row['is_positive']),
            'total_tiles_kept': len(tile_rows),
        })

        logging.info(
            'Processed %s: %s positives, %s negatives kept, split=%s',
            slide_id,
            len(positive_tiles),
            sum(1 for row in tile_rows if not row['is_positive']),
            split,
        )

    manifests_dir = output_dir / 'manifests'
    write_csv(tile_manifest, manifests_dir / 'tiles.csv')
    write_csv(slide_manifest, manifests_dir / 'slides.csv')

    summary = {
        'slides': len(slide_ids),
        'train_tiles': sum(1 for row in tile_manifest if row.get('split') == 'train'),
        'val_tiles': sum(1 for row in tile_manifest if row.get('split') == 'val'),
        'positive_tiles': sum(int(row.get('is_positive', 0)) for row in tile_manifest),
        'negative_tiles': sum(1 for row in tile_manifest if not row.get('is_positive')),
        'tile_size': args.tile_size,
        'stride': args.stride,
        'downsample': args.downsample,
    }

    with (manifests_dir / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    logging.info('Finished. Tiled dataset written to %s', output_dir)
