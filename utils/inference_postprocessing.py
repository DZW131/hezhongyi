import logging
from typing import Optional

import numpy as np

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


_SCIPY_WARNING_EMITTED = False


def estimate_tissue_mask(tile: np.ndarray, white_threshold: float = 230.0) -> np.ndarray:
    array = np.asarray(tile)
    if array.ndim == 2:
        grayscale = array.astype(np.float32)
    else:
        grayscale = array.astype(np.float32).mean(axis=2)
    return grayscale < white_threshold


def align_binary_mask_shape(mask: np.ndarray, target_shape) -> np.ndarray:
    array = (np.asarray(mask) > 0).astype(np.uint8)
    target_h, target_w = target_shape[:2]
    aligned = np.zeros((target_h, target_w), dtype=np.uint8)
    copy_h = min(target_h, array.shape[0])
    copy_w = min(target_w, array.shape[1])
    aligned[:copy_h, :copy_w] = array[:copy_h, :copy_w]
    return aligned


def warn_scipy_unavailable() -> None:
    global _SCIPY_WARNING_EMITTED
    if not _SCIPY_WARNING_EMITTED:
        logging.warning(
            'scipy is not available, so morphology and connected-component postprocessing were skipped. '
            'Install scipy if you want to use hole filling, closing, or component filtering.'
        )
        _SCIPY_WARNING_EMITTED = True


def disk_structure(radius: int) -> np.ndarray:
    radius = max(int(radius), 1)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def apply_binary_postprocessing(
    mask: np.ndarray,
    tissue_mask: Optional[np.ndarray] = None,
    fill_holes: bool = False,
    closing_radius: int = 0,
    min_component_area: int = 0,
    max_component_area: int = 0,
    max_component_extent: int = 0,
) -> np.ndarray:
    output = (np.asarray(mask) > 0).astype(np.uint8)

    if tissue_mask is not None:
        tissue_mask_aligned = align_binary_mask_shape(tissue_mask, output.shape)
        output = output * tissue_mask_aligned

    needs_scipy = (
        fill_holes
        or closing_radius > 0
        or min_component_area > 0
        or max_component_area > 0
        or max_component_extent > 0
    )
    if not needs_scipy:
        return output

    if ndimage is None:
        warn_scipy_unavailable()
        return output

    if fill_holes:
        output = ndimage.binary_fill_holes(output > 0).astype(np.uint8)

    if closing_radius > 0:
        structure = disk_structure(closing_radius)
        output = ndimage.binary_closing(output > 0, structure=structure).astype(np.uint8)
        if fill_holes:
            output = ndimage.binary_fill_holes(output > 0).astype(np.uint8)

    if min_component_area <= 0 and max_component_area <= 0 and max_component_extent <= 0:
        return output

    labeled, num_components = ndimage.label(output)
    if num_components == 0:
        return output

    kept = np.zeros_like(output, dtype=np.uint8)
    objects = ndimage.find_objects(labeled)

    for component_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue

        component_mask = labeled[slices] == component_id
        area = int(component_mask.sum())
        height = int(slices[0].stop - slices[0].start)
        width = int(slices[1].stop - slices[1].start)

        if min_component_area > 0 and area < min_component_area:
            continue
        if max_component_area > 0 and area > max_component_area:
            continue
        if max_component_extent > 0 and (height > max_component_extent or width > max_component_extent):
            continue

        kept_slice = kept[slices]
        kept_slice[component_mask] = 1

    return kept
