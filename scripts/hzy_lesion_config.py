from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class LesionLabel:
    label: str
    slug: str
    count: int
    class_id: Optional[int] = None


@dataclass(frozen=True)
class LesionTask:
    name: str
    slug: str
    labels: List[LesionLabel]
    class_names: Optional[Dict[int, str]] = None

    @property
    def num_classes(self) -> int:
        explicit_class_ids = [label.class_id for label in self.labels if label.class_id is not None]
        if explicit_class_ids:
            return max(explicit_class_ids) + 1
        return len(self.labels) + 1

    @property
    def class_map(self) -> Dict[str, int]:
        return {
            label.label: label.class_id if label.class_id is not None else index
            for index, label in enumerate(self.labels, start=1)
        }

    @property
    def class_mapping(self) -> Dict[str, str]:
        if self.class_names:
            foreground = {
                str(class_id): self.class_names[class_id]
                for class_id in sorted(self.class_names)
            }
        else:
            foreground = {
                str(index): label.label
                for index, label in enumerate(self.labels, start=1)
            }
        return {"0": "background", **foreground}


GLOMERULUS_LABELS: List[LesionLabel] = [
    LesionLabel("未废弃肾小球", "non_discarded_glomerulus", 1325),
    LesionLabel("废弃肾小球", "discarded_glomerulus", 39),
]

PROLIFERATION_LABELS: List[LesionLabel] = [
    LesionLabel("肾小球系膜细胞增生", "mesangial_hypercellularity", 308),
    LesionLabel("毛细血管内细胞增生", "endocapillary_hypercellularity", 179),
]

PROLIFERATION_BINARY_LABELS: List[LesionLabel] = [
    LesionLabel(label.label, label.slug, label.count, class_id=1)
    for label in PROLIFERATION_LABELS
]

CRESCENT_LABELS: List[LesionLabel] = [
    LesionLabel("细胞性新月体", "cellular_crescent", 56),
    LesionLabel("纤维细胞性新月体", "fibrocellular_crescent", 42),
    LesionLabel("纤维性新月体", "fibrous_crescent", 22),
]

CRESCENT_BINARY_LABELS: List[LesionLabel] = [
    LesionLabel(label.label, label.slug, label.count, class_id=1)
    for label in CRESCENT_LABELS
]

OTHER_LESION_LABELS: List[LesionLabel] = [
    LesionLabel("节段硬化", "segmental_sclerosis", 30),
    LesionLabel("节段球囊粘连", "segmental_capsular_adhesion", 16),
    LesionLabel("纤维素样坏死", "fibrinoid_necrosis", 4),
    LesionLabel("纤维素性血栓", "fibrin_thrombus", 1),
]

LESION_TASKS: List[LesionTask] = [
    LesionTask("proliferation binary", "proliferation_binary", PROLIFERATION_BINARY_LABELS, class_names={1: "proliferation"}),
    LesionTask("细胞增生类病变", "proliferation", PROLIFERATION_LABELS),
    LesionTask("新月体类病变", "crescent", CRESCENT_LABELS),
    LesionTask("新月体二分类", "crescent_binary", CRESCENT_BINARY_LABELS, class_names={1: "新月体"}),
    LesionTask("其他病变", "other_lesions", OTHER_LESION_LABELS),
]


def tasks_by_slug() -> Dict[str, LesionTask]:
    return {task.slug: task for task in LESION_TASKS}


def resolve_lesion_tasks(requested_tasks: Iterable[str]) -> List[LesionTask]:
    requested = list(requested_tasks)
    if not requested:
        return list(LESION_TASKS)

    by_slug = tasks_by_slug()
    by_name = {task.name: task for task in LESION_TASKS}
    resolved = []
    for item in requested:
        if item in by_slug:
            resolved.append(by_slug[item])
        elif item in by_name:
            resolved.append(by_name[item])
        else:
            valid = sorted(list(by_slug.keys()) + list(by_name.keys()))
            raise ValueError("Unknown lesion task '{}'. Valid values: {}".format(item, ", ".join(valid)))
    return resolved
