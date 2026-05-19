import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


_SCIPY_WARNING_EMITTED = False


def estimate_tissue_mask(
    tile: np.ndarray,
    white_threshold: float = 230.0,
    black_threshold: float = 0.0,
) -> np.ndarray:
    array = np.asarray(tile)
    if array.ndim == 2:
        grayscale = array.astype(np.float32)
    else:
        grayscale = array.astype(np.float32).mean(axis=2)
    tissue = grayscale < white_threshold
    if black_threshold > 0:
        tissue &= grayscale > black_threshold
    return tissue


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
            'Install scipy if you want to use hole filling, morphology, component filtering, or touching splits.'
        )
        _SCIPY_WARNING_EMITTED = True


def disk_structure(radius: int) -> np.ndarray:
    radius = max(int(radius), 1)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _keep_components_by_rules(
    mask: np.ndarray,
    min_component_area: int = 0,
    max_component_area: int = 0,
    max_component_extent: int = 0,
) -> np.ndarray:
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


def _component_local_maxima(distance: np.ndarray, min_distance: int, max_markers: int) -> List[Tuple[int, int, float]]:
    if distance.size == 0:
        return []

    min_distance = max(int(min_distance), 1)
    footprint_size = 2 * min_distance + 1
    local_max = distance == ndimage.maximum_filter(distance, size=footprint_size)
    local_max &= distance >= float(min_distance)

    labeled_peaks, num_peaks = ndimage.label(local_max)
    if num_peaks == 0:
        return []

    peaks: List[Tuple[int, int, float]] = []
    objects = ndimage.find_objects(labeled_peaks)
    for peak_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue

        peak_mask = labeled_peaks[slices] == peak_id
        values = distance[slices][peak_mask]
        if values.size == 0:
            continue

        coords = np.argwhere(peak_mask)
        best_index = int(np.argmax(values))
        local_y, local_x = coords[best_index]
        y = int(local_y + slices[0].start)
        x = int(local_x + slices[1].start)
        peaks.append((y, x, float(values[best_index])))

    peaks.sort(key=lambda item: item[2], reverse=True)
    if max_markers > 0:
        peaks = peaks[:max_markers]
    return peaks


def split_touching_components(
    mask: np.ndarray,
    min_component_area: int = 0,
    min_peak_distance: int = 32,
    max_markers: int = 4,
) -> np.ndarray:
    if ndimage is None:
        warn_scipy_unavailable()
        return (np.asarray(mask) > 0).astype(np.uint8)

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    labeled, num_components = ndimage.label(binary)
    if num_components == 0:
        return binary

    output = np.zeros_like(binary, dtype=np.uint8)
    objects = ndimage.find_objects(labeled)
    for component_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue

        component = labeled[slices] == component_id
        area = int(component.sum())
        if min_component_area > 0 and area < min_component_area:
            output_slice = output[slices]
            output_slice[component] = 1
            continue

        distance = ndimage.distance_transform_edt(component)
        peaks = _component_local_maxima(
            distance=distance,
            min_distance=min_peak_distance,
            max_markers=max_markers,
        )
        if len(peaks) < 2:
            output_slice = output[slices]
            output_slice[component] = 1
            continue

        marker_image = np.zeros(component.shape, dtype=np.int32)
        for marker_id, (y, x, _value) in enumerate(peaks, start=1):
            marker_image[y, x] = marker_id

        # Voronoi split inside the foreground component. This is intentionally
        # conservative: it separates clearly multi-lobed masks without adding
        # extra foreground pixels.
        _, nearest_indices = ndimage.distance_transform_edt(marker_image == 0, return_indices=True)
        nearest_marker_ids = marker_image[nearest_indices[0], nearest_indices[1]]
        split_labels = np.where(component, nearest_marker_ids, 0)
        boundary = np.zeros_like(component, dtype=bool)
        boundary[:, 1:] |= (split_labels[:, 1:] != split_labels[:, :-1]) & (split_labels[:, 1:] > 0) & (split_labels[:, :-1] > 0)
        boundary[:, :-1] |= (split_labels[:, 1:] != split_labels[:, :-1]) & (split_labels[:, 1:] > 0) & (split_labels[:, :-1] > 0)
        boundary[1:, :] |= (split_labels[1:, :] != split_labels[:-1, :]) & (split_labels[1:, :] > 0) & (split_labels[:-1, :] > 0)
        boundary[:-1, :] |= (split_labels[1:, :] != split_labels[:-1, :]) & (split_labels[1:, :] > 0) & (split_labels[:-1, :] > 0)
        split_binary = component & ~boundary
        output_slice = output[slices]
        output_slice[split_binary] = 1

    return output


def summarize_binary_mask(mask: np.ndarray) -> Dict[str, int]:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if ndimage is None:
        return {
            "foreground_pixels": int(binary.sum()),
            "components": -1,
        }
    _labeled, num_components = ndimage.label(binary)
    return {
        "foreground_pixels": int(binary.sum()),
        "components": int(num_components),
    }


def apply_binary_postprocessing(
    mask: np.ndarray,
    tissue_mask: Optional[np.ndarray] = None,
    fill_holes: bool = False,
    closing_radius: int = 0,
    opening_radius: int = 0,
    min_component_area: int = 0,
    max_component_area: int = 0,
    max_component_extent: int = 0,
    split_touching: bool = False,
    split_min_component_area: int = 0,
    split_min_peak_distance: int = 32,
    split_max_markers: int = 4,
) -> np.ndarray:
    output = (np.asarray(mask) > 0).astype(np.uint8)

    if tissue_mask is not None:
        tissue_mask_aligned = align_binary_mask_shape(tissue_mask, output.shape)
        output = output * tissue_mask_aligned

    needs_scipy = (
        fill_holes
        or closing_radius > 0
        or opening_radius > 0
        or min_component_area > 0
        or max_component_area > 0
        or max_component_extent > 0
        or split_touching
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

    if opening_radius > 0:
        structure = disk_structure(opening_radius)
        output = ndimage.binary_opening(output > 0, structure=structure).astype(np.uint8)

    if min_component_area <= 0 and max_component_area <= 0 and max_component_extent <= 0:
        filtered = output
    else:
        filtered = _keep_components_by_rules(
            output,
            min_component_area=min_component_area,
            max_component_area=max_component_area,
            max_component_extent=max_component_extent,
        )

    if not split_touching:
        return filtered

    return split_touching_components(
        filtered,
        min_component_area=split_min_component_area,
        min_peak_distance=split_min_peak_distance,
        max_markers=split_max_markers,
    )
