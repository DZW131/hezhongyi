"""Render real held-out cell images with annotation-matched correctness overlays.

Match same-class points first with the evaluation's strict <10 px maximum
bipartite matching. Match remaining points across classes only for displaying
class errors. This display decomposition is not the class-agnostic evaluation.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial.distance import cdist


CLASSES = ("endothelial_cell", "mesangial_cell", "podocyte")
NAMES = ("内皮细胞", "系膜细胞", "足细胞")
CLASS_COLORS = ("#0072B2", "#E69F00", "#CC79A7")
STATUS = {
    "correct": ("位置及类别正确", "#009E73", "o"),
    "wrong_class": ("位置匹配但类别错误", "#E69F00", "D"),
    "false_positive": ("未匹配预测点（误检）", "#D55E00", "x"),
    "missed": ("未匹配标注点（漏检）", "#0072B2", "s"),
}


def paired_points(predicted, truth, distance=10.0):
    predicted = np.asarray(predicted).reshape(-1, 2)
    truth = np.asarray(truth).reshape(-1, 2)
    if not len(predicted) or not len(truth):
        return []
    matrix = csr_matrix(cdist(predicted, truth) < distance)
    matched = maximum_bipartite_matching(matrix, perm_type="column")
    return [(i, int(j)) for i, j in enumerate(matched) if j >= 0]


def classify_matches(predicted, pred_classes, truth, true_classes, distance=10.0):
    predicted, truth = np.asarray(predicted).reshape(-1, 2), np.asarray(truth).reshape(-1, 2)
    pred_classes, true_classes = np.asarray(pred_classes), np.asarray(true_classes)
    correct = []
    for label in range(3):
        p = np.flatnonzero(pred_classes == label)
        t = np.flatnonzero(true_classes == label)
        correct.extend((int(p[i]), int(t[j])) for i, j in paired_points(predicted[p], truth[t], distance))
    used_p, used_t = {i for i, _ in correct}, {j for _, j in correct}
    remaining_p = np.array([i for i in range(len(predicted)) if i not in used_p], dtype=int)
    remaining_t = np.array([j for j in range(len(truth)) if j not in used_t], dtype=int)
    wrong = [(int(remaining_p[i]), int(remaining_t[j]))
             for i, j in paired_points(predicted[remaining_p], truth[remaining_t], distance)]
    used_p.update(i for i, _ in wrong)
    used_t.update(j for _, j in wrong)
    # Same-class maximum matching already exhausts all remaining same-class edges.
    assert all(pred_classes[i] != true_classes[j] for i, j in wrong)
    return {
        "correct": correct,
        "wrong_class": wrong,
        "false_positive": [i for i in range(len(predicted)) if i not in used_p],
        "missed": [j for j in range(len(truth)) if j not in used_t],
    }


def load_record(dataset, row):
    annotation = json.loads((dataset / "test_point" / (row["crop_id"] + ".json")).read_text(encoding="utf-8"))
    truth = np.concatenate([np.asarray(annotation[c], dtype=float).reshape(-1, 2) for c in CLASSES])
    labels = np.concatenate([np.full(len(annotation[c]), k) for k, c in enumerate(CLASSES)])
    pred = np.asarray(row["points"], dtype=float).reshape(-1, 2)
    p_labels = np.asarray(row["classes"], dtype=int)
    matching = classify_matches(pred, p_labels, truth, labels)
    for k, c in enumerate(CLASSES):
        actual = sum(p_labels[i] == k for i, _ in matching["correct"])
        if actual != row[c + "_matched"]:
            raise ValueError("Stored and visualized class matching disagree: " + row["crop_id"] + "/" + c)
    return dict(row=row, truth=truth, labels=labels, pred=pred, p_labels=p_labels,
                matches=matching, polygon=np.asarray(annotation["roi_polygon"]))


def setup_fonts():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False


def status_scatter(ax, record):
    for key, (_, color, marker) in STATUS.items():
        indexes = record["matches"][key]
        if key in ("correct", "wrong_class"):
            xy = record["pred"][[i for i, _ in indexes]]
        elif key == "false_positive":
            xy = record["pred"][indexes]
        else:
            xy = record["truth"][indexes]
        if not len(xy):
            continue
        options = dict(marker=marker, s=58 if key == "correct" else 68, linewidths=1.3, zorder=4)
        if marker == "x":
            options["color"] = color
        else:
            options.update(facecolors="none", edgecolors=color)
        ax.scatter(xy[:, 0], xy[:, 1], **options)


def overview(record, dataset, output, number):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    image = np.asarray(Image.open(dataset / "test_image" / (record["row"]["crop_id"] + ".jpg")).convert("RGB"))
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 7.0))
    fig.subplots_adjust(left=.015, right=.985, top=.82, bottom=.17, wspace=.035)
    for ax in axes:
        ax.imshow(image)
        polygon = record["polygon"]
        ax.plot(polygon[:, 0], polygon[:, 1], color="#56B4E9", lw=.75, alpha=.8)
        ax.set_xlim(0, image.shape[1]); ax.set_ylim(image.shape[0], 0); ax.set_axis_off()
    axes[0].set_title("原始肾小球图像", fontsize=15, pad=12)
    axes[1].set_title("参考点标注：三类固有细胞", fontsize=15, pad=12)
    axes[2].set_title("预测核对：绿色圆圈为位置与类别均正确", fontsize=15, pad=12)
    for k, color in enumerate(CLASS_COLORS):
        xy = record["truth"][record["labels"] == k]
        if len(xy):
            axes[1].scatter(xy[:, 0], xy[:, 1], s=40, marker="o", facecolors="none", edgecolors=color, linewidths=1.15)
    status_scatter(axes[2], record)
    m = record["matches"]
    counts = "正确 {}  |  类别错误 {}  |  误检 {}  |  漏检 {}".format(*(len(m[k]) for k in STATUS))
    fig.suptitle("细胞检测逐点核对 · 实例{}".format(number), fontsize=21, y=.97)
    fig.text(.5, .885, "参考标注 {} 个 / 预测 {} 个；{}".format(len(record["truth"]), len(record["pred"]), counts), ha="center", fontsize=12)
    handles = [Line2D([], [], marker="o", color=c, markerfacecolor="none", linestyle="none", label=n)
               for n, c in zip(NAMES, CLASS_COLORS)]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.13, .095), frameon=False, ncol=3, fontsize=11)
    handles = [Line2D([], [], marker=marker, color=color, markerfacecolor="none", linestyle="none", label=label)
               for label, color, marker in STATUS.values()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035), frameon=False, ncol=4, fontsize=11)
    fig.text(.5, .015, "正确＝与参考标注同类且一对一距离＜10像素；按同类优先匹配。仅评估已标注区域，错误点完整保留。", ha="center", fontsize=10, color="#555555")
    path = output / ("example_{}_matched.png".format(number))
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)
    return path


def detail(records, dataset, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    fig.subplots_adjust(left=.04, right=.96, top=.84, bottom=.10, hspace=.27, wspace=.12)
    chosen = []
    for k, name in enumerate(NAMES):
        options = [(r, i, j) for r in records for i, j in r["matches"]["correct"] if r["labels"][j] == k]
        options.sort(key=lambda x: (float(np.linalg.norm(x[0]["truth"][x[2]] - [512, 512])), x[0]["row"]["crop_id"], x[1]))
        picked = []
        for record, i, j in options:
            if any(record["row"]["crop_id"] == r["row"]["crop_id"] and
                   np.linalg.norm(record["truth"][j] - r["truth"][q]) < 128 for r, _, q in picked):
                continue
            picked.append((record, i, j))
            if len(picked) == 2:
                break
        for row, (record, i, j) in enumerate(picked):
            ax = axes[row, k]
            image = np.asarray(Image.open(dataset / "test_image" / (record["row"]["crop_id"] + ".jpg")).convert("RGB"))
            px, py = record["pred"][i]; tx, ty = record["truth"][j]
            x0, y0 = int(tx) - 64, int(ty) - 64
            ax.imshow(image)
            ax.set_xlim(x0, x0 + 128); ax.set_ylim(y0 + 128, y0); ax.set_axis_off()
            ax.add_patch(Circle((px, py), 9, fill=False, edgecolor="#009E73", linewidth=2.4))
            ax.plot(tx, ty, "+", color="white", markersize=8, markeredgewidth=1.6)
            ax.plot([px, tx], [py, ty], color="#009E73", linewidth=1)
            distance = float(np.linalg.norm(record["pred"][i] - record["truth"][j]))
            ax.set_title("{}：位置和类别均匹配\n点距离 {:.1f} 像素".format(name, distance), fontsize=12, pad=7)
            chosen.append(dict(cell_type=CLASSES[k], crop_id=record["row"]["crop_id"],
                               prediction_index=i, truth_index=j, distance_px=distance))
    fig.suptitle("三类细胞正确检出的局部放大", fontsize=21, y=.975)
    fig.text(.5, .915, "每幅显示128×128像素原始局部；绿色圆圈为预测位置，白色十字为参考标注", ha="center", fontsize=11)
    fig.text(.5, .055, "以上为按匹配结果选取的正确检出示例；不代表总体准确率。完整实例图同时展示误检、漏检和类别错误。", ha="center", fontsize=10, color="#555555")
    fig.savefig(output / "correct_cells_closeup.png", dpi=200, facecolor="white")
    plt.close(fig)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.results / "test_predictions.json").read_text(encoding="utf-8"))
    if not rows:
        parser.error("test_predictions.json contains no predictions")
    records = [load_record(args.dataset, row) for row in rows]
    actual = np.bincount([r["p_labels"][i] for r in records for i, _ in r["matches"]["correct"]], minlength=3)
    metrics = json.loads((args.results / "test_metrics.json").read_text(encoding="utf-8"))
    assert actual.tolist() == [metrics["per_class"][c]["matched"] for c in CLASSES]
    ranked = sorted(records, key=lambda record: abs(len(record["pred"]) - len(record["truth"])))
    examples = {record["row"]["crop_id"]: record for record in (ranked[0], ranked[len(ranked) // 2], ranked[-1])}
    selected = [examples[crop_id] for crop_id in sorted(examples)]
    setup_fonts()
    overview_paths = [overview(record, args.dataset, args.output, i + 1) for i, record in enumerate(selected)]
    closeups = detail(selected, args.dataset, args.output)
    audit = {"source": str(args.results / "test_predictions.json"), "split": "test",
             "matching": "same-class maximum one-to-one matching first, strict distance <10 pixels; remaining cross-class pairs for error display only",
             "class_agnostic_detection_decomposition": False,
             "correct_class_matches_all_75_crops": dict(zip(CLASSES, actual.tolist())),
             "all_class_counts_verified_against_saved_metrics": True,
             "selection": "minimum, median and maximum absolute total count error; ordered by crop ID",
             "examples": [{"image": p.name, "crop_id": r["row"]["crop_id"],
                           "counts": {k: len(v) for k, v in r["matches"].items()}} for r, p in zip(selected, overview_paths)],
             "correct_closeups": closeups}
    (args.output / "visualization_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "point_matches.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["crop_id", "status", "prediction_index", "truth_index", "predicted_class", "true_class", "distance_px"])
        writer.writeheader()
        for r in selected:
            for status, entries in r["matches"].items():
                pairs = entries if status in ("correct", "wrong_class") else [(i, None) if status == "false_positive" else (None, i) for i in entries]
                for i, j in pairs:
                    writer.writerow(dict(crop_id=r["row"]["crop_id"], status=status, prediction_index=i, truth_index=j,
                                         predicted_class=CLASSES[r["p_labels"][i]] if i is not None else "",
                                         true_class=CLASSES[r["labels"][j]] if j is not None else "",
                                         distance_px=float(np.linalg.norm(r["pred"][i]-r["truth"][j])) if i is not None and j is not None else ""))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
