import logging
from typing import Dict

import numpy as np

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


_SCIPY_WARNING_EMITTED = False


def warn_scipy_unavailable() -> None:
    global _SCIPY_WARNING_EMITTED
    if not _SCIPY_WARNING_EMITTED:
        logging.warning(
            'scipy is not available, so morphology and connected-component postprocessing were skipped. '
            'Install scipy if you want to use hole filling, boundary smoothing, or small-component filtering.'
        )
        _SCIPY_WARNING_EMITTED = True


def _close_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    structure = disk_structure(radius)
    return ndimage.binary_closing(mask > 0, structure=structure).astype(np.uint8)


def disk_structure(radius: int) -> np.ndarray:
    radius = max(int(radius), 1)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _remove_small_components(mask: np.ndarray, min_component_area: int = 0) -> np.ndarray:
    labeled, num_components = ndimage.label(mask > 0)
    if num_components == 0:
        return mask.astype(np.uint8)

    kept = np.zeros_like(mask, dtype=np.uint8)
    objects = ndimage.find_objects(labeled)

    for component_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue

        component_mask = labeled[slices] == component_id
        area = int(component_mask.sum())
        if min_component_area > 0 and area < min_component_area:
            continue

        kept_slice = kept[slices]
        kept_slice[component_mask] = 1

    return kept


def _fill_component_holes(mask: np.ndarray, repair_radius: int = 0) -> np.ndarray:
    labeled, num_components = ndimage.label(mask > 0)
    if num_components == 0:
        return mask.astype(np.uint8)

    filled_mask = np.zeros_like(mask, dtype=np.uint8)
    objects = ndimage.find_objects(labeled)
    pad = max(int(repair_radius), 1)

    for component_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue

        y0 = max(slices[0].start - pad, 0)
        y1 = min(slices[0].stop + pad, mask.shape[0])
        x0 = max(slices[1].start - pad, 0)
        x1 = min(slices[1].stop + pad, mask.shape[1])
        region = (slice(y0, y1), slice(x0, x1))
        component = labeled[region] == component_id

        source = component
        if repair_radius > 0:
            source = _close_binary(component, radius=repair_radius)

        filled_component = ndimage.binary_fill_holes(source > 0)
        filled_slice = filled_mask[region]
        filled_slice[filled_component] = 1

    return filled_mask


def summarize_binary_mask(mask: np.ndarray) -> Dict[str, int]:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if ndimage is None:
        return {
            "foreground_pixels": int(binary.sum()),
            "components": -1,
        }

    labeled, num_components = ndimage.label(binary)
    summary = {
        "foreground_pixels": int(binary.sum()),
        "components": int(num_components),
    }
    if num_components == 0:
        return summary

    areas = []
    extents = []
    objects = ndimage.find_objects(labeled)
    for component_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue

        component_mask = labeled[slices] == component_id
        areas.append(int(component_mask.sum()))
        height = int(slices[0].stop - slices[0].start)
        width = int(slices[1].stop - slices[1].start)
        extents.append(max(height, width))

    if not areas:
        return summary

    area_values = np.asarray(areas, dtype=np.float64)
    extent_values = np.asarray(extents, dtype=np.float64)
    summary.update(
        {
            "component_area_min": int(area_values.min()),
            "component_area_median": int(np.median(area_values)),
            "component_area_max": int(area_values.max()),
            "component_extent_min": int(extent_values.min()),
            "component_extent_median": int(np.median(extent_values)),
            "component_extent_max": int(extent_values.max()),
        }
    )
    return summary


def apply_binary_postprocessing(
    mask: np.ndarray,
    fill_holes: bool = False,
    hole_repair_radius: int = 0,
    smooth_radius: int = 0,
    min_component_area: int = 0,
) -> np.ndarray:
    output = (np.asarray(mask) > 0).astype(np.uint8)

    needs_scipy = (
        fill_holes
        or hole_repair_radius > 0
        or smooth_radius > 0
        or min_component_area > 0
    )
    if not needs_scipy:
        return output

    if ndimage is None:
        warn_scipy_unavailable()
        return output

    if min_component_area > 0:
        output = _remove_small_components(output, min_component_area=min_component_area)

    if fill_holes:
        output = _fill_component_holes(output, repair_radius=hole_repair_radius)

    if smooth_radius > 0:
        output = _close_binary(output, radius=smooth_radius)
        structure = disk_structure(smooth_radius)
        output = ndimage.binary_opening(output > 0, structure=structure).astype(np.uint8)
        if fill_holes:
            output = ndimage.binary_fill_holes(output > 0).astype(np.uint8)

    if min_component_area > 0:
        output = _remove_small_components(output, min_component_area=min_component_area)

    return output
