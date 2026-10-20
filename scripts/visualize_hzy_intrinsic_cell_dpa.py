"""Render actual DPA-P2PNet predictions for held-out intrinsic-cell crops."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


CLASSES = ["endothelial_cell", "mesangial_cell", "podocyte"]
CLASS_LABELS = ["Endothelial", "Mesangial", "Podocyte"]
COLORS = [(54, 67, 244), (44, 160, 44), (231, 166, 0)]  # BGR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-distance", type=float, default=10.0)
    return parser.parse_args()


def deduplicate(points: np.ndarray, scores: np.ndarray, interval: float) -> tuple[np.ndarray, np.ndarray]:
    fused = np.zeros(len(points), dtype=bool)
    result_points, result_classes = [], []
    for index in range(len(points)):
        if fused[index]:
            continue
        nearby = np.flatnonzero(np.linalg.norm(points[index:] - points[index], axis=1) < interval) + index
        fused[nearby] = True
        local_scores = scores[nearby]
        best_flat = int(np.argmax(local_scores))
        local_point, local_class = np.unravel_index(best_flat, local_scores.shape)
        result_points.append(points[nearby[local_point]])
        result_classes.append(local_class)
    return np.asarray(result_points), np.asarray(result_classes)


def match_count(predictions: np.ndarray, targets: np.ndarray, distance: float) -> int:
    if len(predictions) == 0 or len(targets) == 0:
        return 0
    from scipy.optimize import linear_sum_assignment

    distances = np.linalg.norm(predictions[:, None, :] - targets[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(distances)
    return int(np.sum(distances[rows, cols] < distance))


def crop_metrics(pred_points: np.ndarray, gt_by_class: list[np.ndarray], distance: float) -> dict:
    gt_points = np.concatenate(gt_by_class, axis=0) if any(len(x) for x in gt_by_class) else np.empty((0, 2))
    matched = match_count(pred_points, gt_points, distance)
    precision = matched / len(pred_points) if len(pred_points) else 0.0
    recall = matched / len(gt_points) if len(gt_points) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"pred_count": int(len(pred_points)), "gt_count": int(len(gt_points)), "matched": matched,
            "precision": precision, "recall": recall, "f1": f1}


def draw_points(image: np.ndarray, points: np.ndarray, classes: np.ndarray, style: str) -> np.ndarray:
    canvas = image.copy()
    scale = image.shape[0] / 1024.0
    for point, category in zip(points, classes):
        x, y = (np.asarray(point) * scale).round().astype(int)
        color = COLORS[int(category)]
        if style == "gt":
            cv2.circle(canvas, (x, y), 4, color, 1, lineType=cv2.LINE_AA)
        else:
            cv2.line(canvas, (x - 3, y - 3), (x + 3, y + 3), color, 2, lineType=cv2.LINE_AA)
            cv2.line(canvas, (x - 3, y + 3), (x + 3, y - 3), color, 2, lineType=cv2.LINE_AA)
    return canvas


def decorate(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    header = np.full((62, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, title, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(header, subtitle, (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA)
    return np.vstack([header, image])


def render_case(image: np.ndarray, gt_points: np.ndarray, gt_classes: np.ndarray, pred_points: np.ndarray,
                pred_classes: np.ndarray, row: dict) -> np.ndarray:
    reduced = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    ground_truth = draw_points(reduced, gt_points, gt_classes, "gt")
    prediction = draw_points(reduced, pred_points, pred_classes, "pred")
    overlay = draw_points(ground_truth, pred_points, pred_classes, "pred")
    details = "GT {} | Pred {} | Match {} | F1 {:.3f}".format(
        row["gt_count"], row["pred_count"], row["matched"], row["f1"]
    )
    return np.hstack([
        decorate(ground_truth, "Ground truth (circles)", details),
        decorate(prediction, "Prediction (crosses)", details),
        decorate(overlay, "Overlay", details),
    ])


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.source_root))
    from models.dpa_p2p.dpa_p2pnet import build_dpa_model

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mean, std = np.load(args.dataset / "mean_std.npy")
    model_args = SimpleNamespace(backbone="vit_adapter", ssl_type="", ssl_path="", hidden_dim=256,
                                 num_levels=3, num_fov=0, dropout=0.1, space=8)
    model = build_dpa_model(model_args, num_classes=len(CLASSES)).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)

    rows, saved = [], []
    images = sorted((args.dataset / "test_image").glob("*.jpg"))
    with torch.no_grad():
        for image_path in images:
            annotation_path = args.dataset / "test_point" / (image_path.stem + ".json")
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            gt_by_class = [np.asarray(annotation[name], dtype=np.float32).reshape(-1, 2) for name in CLASSES]
            gt_points = np.concatenate(gt_by_class, axis=0)
            gt_classes = np.concatenate([np.full(len(points), index) for index, points in enumerate(gt_by_class)])
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0)
            tensor = (tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
            outputs = model(tensor.unsqueeze(0).cuda())
            pred_points = outputs["pnt_coords"][0].cpu().numpy()
            scores = outputs["cls_logits"][0].softmax(-1).cpu().numpy()
            inside = ((pred_points[:, 0] >= 0) & (pred_points[:, 0] < image.shape[1]) &
                      (pred_points[:, 1] >= 0) & (pred_points[:, 1] < image.shape[0]))
            pred_points, scores = pred_points[inside], scores[inside]
            classes = np.argmax(scores, axis=1)
            keep = classes < len(CLASSES)
            pred_points, pred_classes = deduplicate(pred_points[keep], scores[keep], args.match_distance)
            metrics = crop_metrics(pred_points, gt_by_class, args.match_distance)
            row = {"crop_id": image_path.stem, **metrics}
            for index, name in enumerate(CLASSES):
                row["gt_{}".format(name)] = int(len(gt_by_class[index]))
                row["pred_{}".format(name)] = int(np.sum(pred_classes == index))
            rows.append(row)
            saved.append((row, image, gt_points, gt_classes, pred_points, pred_classes))

    fields = list(rows[0])
    with (args.output_dir / "per_crop_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(saved, key=lambda item: item[0]["f1"])
    selected = [ranked[-1], ranked[len(ranked) // 2], max(saved, key=lambda item: item[0]["gt_count"])]
    unique = []
    seen = set()
    for item in selected:
        if item[0]["crop_id"] not in seen:
            unique.append(item)
            seen.add(item[0]["crop_id"])
    panels = []
    for item in unique:
        panel = render_case(item[1], item[2], item[3], item[4], item[5], item[0])
        panels.append(panel)
        cv2.imwrite(str(args.output_dir / (item[0]["crop_id"] + "_overlay.png")), panel)

    legend = np.full((66, 1536, 3), 255, dtype=np.uint8)
    cv2.putText(legend, "Held-out test predictions: circles = annotation, crosses = DPA-P2PNet prediction", (14, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 2, cv2.LINE_AA)
    for index, label in enumerate(CLASS_LABELS):
        x = 20 + index * 270
        cv2.circle(legend, (x, 51), 7, COLORS[index], 2, cv2.LINE_AA)
        cv2.putText(legend, label, (x + 14, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[index], 2, cv2.LINE_AA)
    cv2.imwrite(str(args.output_dir / "heldout_prediction_gallery.png"), np.vstack([legend] + panels))
    (args.output_dir / "selection.json").write_text(json.dumps([item[0] for item in unique], indent=2), encoding="utf-8")
    print(json.dumps([item[0] for item in unique], indent=2))


if __name__ == "__main__":
    main()
