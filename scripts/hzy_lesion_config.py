from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class LesionLabel:
    label: str
    slug: str
    count: int
    trainable_default: bool


LESION_LABELS: List[LesionLabel] = [
    LesionLabel("废弃肾小球", "discarded_glomerulus", 39, True),
    LesionLabel("肾小球系膜细胞增生", "mesangial_hypercellularity", 308, True),
    LesionLabel("毛细血管内细胞增生", "endocapillary_hypercellularity", 179, True),
    LesionLabel("细胞性新月体", "cellular_crescent", 56, True),
    LesionLabel("纤维细胞性新月体", "fibrocellular_crescent", 42, True),
    LesionLabel("纤维性新月体", "fibrous_crescent", 22, True),
    LesionLabel("节段硬化", "segmental_sclerosis", 30, True),
    LesionLabel("节段球囊粘连", "segmental_capsular_adhesion", 16, False),
    LesionLabel("纤维素样坏死", "fibrinoid_necrosis", 4, False),
    LesionLabel("纤维素性血栓", "fibrin_thrombus", 1, False),
]


def labels_by_slug() -> Dict[str, LesionLabel]:
    return {item.slug: item for item in LESION_LABELS}


def labels_by_name() -> Dict[str, LesionLabel]:
    return {item.label: item for item in LESION_LABELS}


def resolve_lesion_labels(preset: str, requested: Iterable[str]) -> List[LesionLabel]:
    by_slug = labels_by_slug()
    by_name = labels_by_name()
    requested_items = list(requested)

    if requested_items:
        resolved = []
        for item in requested_items:
            if item in by_slug:
                resolved.append(by_slug[item])
            elif item in by_name:
                resolved.append(by_name[item])
            else:
                valid = sorted(list(by_slug.keys()) + list(by_name.keys()))
                raise ValueError("Unknown lesion label '{}'. Valid values: {}".format(item, ", ".join(valid)))
        return resolved

    if preset == "all":
        return list(LESION_LABELS)
    if preset == "trainable":
        return [item for item in LESION_LABELS if item.trainable_default]
    if preset == "rare":
        return [item for item in LESION_LABELS if not item.trainable_default]

    raise ValueError("Unsupported preset '{}'".format(preset))
