import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from hzy_lesion_config import GLOMERULUS_LABELS, LESION_TASKS
from prepare_hubmap_tiles import get_feature_label, read_geojson_features


DEFAULT_ANNOTATIONS_DIR = "/root/datasets/HZY_HSPN_export_ds025/annotations"
DEFAULT_OUTPUT_DIR = "/root/datasets/HZY_HSPN_lesion_tasks/analysis"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze HZY lesion annotation distribution before grouped lesion training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotations-dir", default=DEFAULT_ANNOTATIONS_DIR,
                        help="Directory containing scene JSON annotations")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for analysis CSV/JSON outputs")
    return parser.parse_args()


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    json_paths = sorted(annotations_dir.glob("*.json"))

    all_task_labels = {
        label.label: task.slug
        for task in LESION_TASKS
        for label in task.labels
    }
    glomerulus_labels = {label.label for label in GLOMERULUS_LABELS}
    label_counts = Counter()
    label_slide_sets = defaultdict(set)
    task_counts = Counter()
    task_slide_sets = defaultdict(set)
    unassigned_counts = Counter()

    for path in json_paths:
        slide_id = path.stem
        features = read_geojson_features(path)
        seen_labels = set()
        seen_tasks = set()

        for feature in features:
            label = get_feature_label(feature)
            if not label:
                continue

            label_counts[label] += 1
            seen_labels.add(label)
            task_slug = all_task_labels.get(label)
            if task_slug:
                task_counts[task_slug] += 1
                seen_tasks.add(task_slug)
            elif label not in glomerulus_labels:
                unassigned_counts[label] += 1

        for label in seen_labels:
            label_slide_sets[label].add(slide_id)
        for task_slug in seen_tasks:
            task_slide_sets[task_slug].add(slide_id)

    label_rows = []
    for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0])):
        if label in all_task_labels:
            group = all_task_labels[label]
        elif label in glomerulus_labels:
            group = "glomerulus_reference"
        else:
            group = "unassigned"
        label_rows.append(
            {
                "label": label,
                "group": group,
                "feature_count": int(count),
                "scene_count": len(label_slide_sets[label]),
            }
        )

    task_rows = []
    for task in LESION_TASKS:
        task_rows.append(
            {
                "task_name": task.name,
                "task_slug": task.slug,
                "num_classes": task.num_classes,
                "feature_count": int(task_counts[task.slug]),
                "scene_count": len(task_slide_sets[task.slug]),
                "labels": json.dumps([label.label for label in task.labels], ensure_ascii=False),
            }
        )

    summary = {
        "annotations_dir": str(annotations_dir),
        "scene_json_count": len(json_paths),
        "task_rows": task_rows,
        "label_rows": label_rows,
        "unassigned_counts": dict(unassigned_counts),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(task_rows, output_dir / "task_distribution.csv")
    write_csv(label_rows, output_dir / "label_distribution.csv")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("Analyzed {} annotation JSON files".format(len(json_paths)))
    print("Task distribution:", output_dir / "task_distribution.csv")
    print("Label distribution:", output_dir / "label_distribution.csv")
    print("Summary:", output_dir / "summary.json")


if __name__ == "__main__":
    main()
