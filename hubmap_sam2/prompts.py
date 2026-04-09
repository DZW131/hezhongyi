from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency at import time
    cv2 = None


@dataclass
class PromptRecord:
    object_id: int
    point: Optional[Tuple[float, float]]
    box: Optional[Tuple[float, float, float, float]]
    area: int
    mask: np.ndarray


def mask_to_box(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def distance_transform_point(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    binary = (mask > 0).astype(np.uint8)
    if binary.max() == 0:
        return None

    if cv2 is not None:
        distance = cv2.distanceTransform(binary, distanceType=cv2.DIST_L2, maskSize=5)
        max_index = int(distance.argmax())
        y, x = divmod(max_index, distance.shape[1])
        return (float(x), float(y))

    ys, xs = np.nonzero(binary)
    return (float(xs.mean()), float(ys.mean()))


def instance_prompts_from_instance_map(instance_map: np.ndarray) -> List[PromptRecord]:
    prompts = []
    object_ids = [int(object_id) for object_id in np.unique(instance_map) if int(object_id) > 0]
    for object_id in object_ids:
        mask = instance_map == object_id
        prompts.append(
            PromptRecord(
                object_id=object_id,
                point=distance_transform_point(mask),
                box=mask_to_box(mask),
                area=int(mask.sum()),
                mask=mask,
            )
        )
    return prompts


def _connected_components_with_fallback(binary_mask: np.ndarray) -> Tuple[int, np.ndarray]:
    mask = (binary_mask > 0).astype(np.uint8)
    if cv2 is not None:
        num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
        return int(num_labels), labels.astype(np.int32)

    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    current_label = 0
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(height):
        for x in range(width):
            if mask[y, x] == 0 or labels[y, x] != 0:
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
                    if mask[ny, nx] == 0 or labels[ny, nx] != 0:
                        continue
                    labels[ny, nx] = current_label
                    queue.append((ny, nx))
    return current_label + 1, labels


def prompts_from_binary_mask(
    binary_mask: np.ndarray,
    min_component_area: int = 32,
) -> List[PromptRecord]:
    _, labels = _connected_components_with_fallback(binary_mask)
    prompts: List[PromptRecord] = []
    for object_id in [int(value) for value in np.unique(labels) if int(value) > 0]:
        component_mask = labels == object_id
        area = int(component_mask.sum())
        if area < min_component_area:
            continue
        prompts.append(
            PromptRecord(
                object_id=object_id,
                point=distance_transform_point(component_mask),
                box=mask_to_box(component_mask),
                area=area,
                mask=component_mask,
            )
        )
    return prompts


def prompts_from_metadata(
    metadata: Dict[str, object],
    fallback_instance_map: Optional[np.ndarray] = None,
) -> List[PromptRecord]:
    object_rows = metadata.get("objects") or []
    prompts: List[PromptRecord] = []
    for row in object_rows:
        if not isinstance(row, dict):
            continue
        object_id = int(row.get("tile_object_id", 0))
        point = row.get("point_xy")
        box = row.get("box_xyxy")
        prompts.append(
            PromptRecord(
                object_id=object_id,
                point=(float(point[0]), float(point[1])) if point else None,
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])) if box else None,
                area=int(row.get("visible_area", 0)),
                mask=(fallback_instance_map == object_id) if fallback_instance_map is not None else np.zeros((1, 1), dtype=bool),
            )
        )

    if prompts or fallback_instance_map is None:
        return prompts
    return instance_prompts_from_instance_map(fallback_instance_map)

