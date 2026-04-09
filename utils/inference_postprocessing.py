import logging
from typing import Optional

import numpy as np

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


def estimate_tissue_mask(tile: np.ndarray, white_threshold: float = 230.0) -> np.ndarray:
    array = np.asarray(tile)
    if array.ndim == 2:
        grayscale = array.astype(np.float32)
    else:
        grayscale = array.astype(np.float32).mean(axis=2)
    return grayscale < white_threshold


def apply_binary_postprocessing(
    mask: np.ndarray,
    tissue_mask: Optional[np.ndarray] = None,
    min_component_area: int = 0,
    max_component_area: int = 0,
    max_component_extent: int = 0,
) -> np.ndarray:
    output = (np.asarray(mask) > 0).astype(np.uint8)

    if tissue_mask is not None:
        output = output * (np.asarray(tissue_mask) > 0).astype(np.uint8)

    if min_component_area <= 0 and max_component_area <= 0 and max_component_extent <= 0:
        return output

    if ndimage is None:
        logging.warning(
            'scipy is not available, so connected-component filtering was skipped. '
            'Install scipy if you want to use min/max component filtering.'
        )
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
