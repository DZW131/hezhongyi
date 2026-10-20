import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_aggregate_inference_outputs_and_run_pathology_scorer(tmp_path):
    manifest = tmp_path / "crops.csv"
    write_rows(
        manifest,
        [
            {
                "slide_id": "case_01",
                "split": "val",
                "glomerulus_index": index,
                "crop_id": f"case_01_g{index:04d}_10_{index * 20}",
                "bbox_x_min": index * 10,
                "bbox_y_min": index * 20,
                "bbox_x_max": index * 10 + 50,
                "bbox_y_max": index * 20 + 50,
                "image_path": f"val/images/case_01_g{index:04d}_10_{index * 20}.jpg",
            }
            for index in range(1, 5)
        ],
    )

    proliferation = tmp_path / "proliferation.csv"
    write_rows(
        proliferation,
        [
            {"crop_id": "case_01_g0001_10_20", "prob_positive": "0.91"},
            {"crop_id": "case_01_g0002_10_40", "prob_positive": "0.82"},
        ],
    )

    crescent_detection = tmp_path / "crescent_detection.csv"
    write_rows(crescent_detection, [{"case_id": "case_01_g0001_10_20", "score": "0.88"}])

    roi_manifest = tmp_path / "roi_manifest.csv"
    write_rows(
        roi_manifest,
        [
            {
                "roi_id": "case_01_g0001_10_20_c1_r01",
                "source_stem": "case_01_g0001_10_20",
            }
        ],
    )
    subtype = tmp_path / "crescent_subtype.csv"
    write_rows(
        subtype,
        [
            {
                "roi_id": "case_01_g0001_10_20_c1_r01",
                "pred_class": "cellular_crescent",
                "prob_cellular_crescent": "0.93",
                "prob_fibrocellular_crescent": "0.04",
                "prob_fibrous_crescent": "0.03",
            }
        ],
    )

    other = tmp_path / "other.csv"
    write_rows(other, [{"crop_id": "case_01_g0003_10_60", "segmental_sclerosis_prob": "0.77"}])
    tubular_atrophy = tmp_path / "tubular_atrophy.csv"
    write_rows(
        tubular_atrophy,
        [
            {
                "slide_id": "case_01_tile_export",
                "scene_path": "/tmp/case_01.tiff",
                "tubular_atrophy_target_percent": "30%",
            }
        ],
    )

    output_csv = tmp_path / "per_glomerulus_predictions.csv"
    score_dir = tmp_path / "scores"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "aggregate_hspn_inference_for_scoring.py"),
            "--manifest-csv",
            str(manifest),
            "--proliferation-csv",
            str(proliferation),
            "--crescent-detection-csv",
            str(crescent_detection),
            "--crescent-subtype-csv",
            str(subtype),
            "--crescent-roi-manifest-csv",
            str(roi_manifest),
            "--other-lesion-csv",
            str(other),
            "--tubular-atrophy-slide-csv",
            str(tubular_atrophy),
            "--output-csv",
            str(output_csv),
            "--score-output-dir",
            str(score_dir),
            "--min-glomeruli",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8-sig")))
    assert len(rows) == 4
    first = rows[0]
    assert first["mesangial_prob"] == "0.910000"
    assert first["crescent_prob"] == "0.930000"
    assert first["cellular_crescent_prob"] == "0.930000"

    summary = json.load((tmp_path / "per_glomerulus_predictions.summary.json").open("r", encoding="utf-8"))
    assert summary["glomeruli"] == 4
    assert summary["matched_rows"]["proliferation"] == 2
    assert summary["matched_rows"]["crescent_subtype"] == 1
    assert summary["matched_rows"]["tubular_atrophy_slide"] == 4
    assert {row["ifta_percent"] for row in rows} == {"30%"}

    slide_scores = list(csv.DictReader((score_dir / "slide_scores.csv").open("r", encoding="utf-8-sig")))
    assert slide_scores[0]["M"] == "M1"
    assert slide_scores[0]["S"] == "S1"
    assert slide_scores[0]["T"] == "T1"
    assert slide_scores[0]["C"] == "C2"
    assert slide_scores[0]["any_crescent_count"] == "1"
    assert slide_scores[0]["any_crescent_percent"] == "0.2500"
    assert slide_scores[0]["cellular_crescent_count"] == "1"
    assert slide_scores[0]["cellular_crescent_percent"] == "0.2500"
    assert slide_scores[0]["fibrocellular_crescent_count"] == "0"
    assert slide_scores[0]["fibrous_crescent_count"] == "0"
    assert (score_dir / "evidence" / "case_01_evidence.png").exists()

    iskdc_scores = list(csv.DictReader((score_dir / "iskdc_scores.csv").open("r", encoding="utf-8-sig")))
    oxford_scores = list(csv.DictReader((score_dir / "oxford_scores.csv").open("r", encoding="utf-8-sig")))
    assert iskdc_scores[0]["ISKDC"] == slide_scores[0]["ISKDC"]
    assert iskdc_scores[0]["any_crescent_percent"] == "0.2500"
    assert oxford_scores[0]["Oxford_MEST_C"] == "M1E0S1T1C2"
    assert oxford_scores[0]["active_crescent_percent"] == "0.2500"
    assert (score_dir / "iskdc_scores.json").exists()
    assert (score_dir / "oxford_scores.json").exists()
    assert (score_dir / "iskdc_evidence" / "case_01_iskdc_evidence.png").exists()
    assert (score_dir / "oxford_evidence" / "case_01_oxford_evidence.png").exists()


def test_aggregate_weak_proliferation_classifiers_without_boxes(tmp_path):
    manifest = tmp_path / "crops.csv"
    write_rows(
        manifest,
        [
            {
                "slide_id": "case_02",
                "glomerulus_index": index,
                "crop_id": f"case_02_g{index:04d}_10_{index * 20}",
                "bbox_x_min": index * 10,
                "bbox_y_min": index * 20,
                "bbox_x_max": index * 10 + 50,
                "bbox_y_max": index * 20 + 50,
            }
            for index in range(1, 5)
        ],
    )

    mesangial = tmp_path / "per_crop_M.csv"
    write_rows(
        mesangial,
        [
            {"name": "case_02_g0001_10_20.jpg", "label": "yesM", "bin_pred": "1"},
            {"name": "case_02_g0002_10_40.jpg", "label": "yesM", "bin_pred": "1"},
            {"name": "case_02_g0003_10_60.jpg", "label": "noM", "bin_pred": "0"},
            {"name": "case_02_g0004_10_80.jpg", "label": "noM", "bin_pred": "0"},
        ],
    )
    endocapillary = tmp_path / "per_crop_E.csv"
    write_rows(
        endocapillary,
        [
            {"name": "case_02_g0001_10_20.jpg", "label": "noE", "bin_pred": "0"},
            {"name": "case_02_g0002_10_40.jpg", "label": "yesE", "bin_pred": "1"},
            {"name": "case_02_g0003_10_60.jpg", "label": "noE", "bin_pred": "0"},
            {"name": "case_02_g0004_10_80.jpg", "label": "noE", "bin_pred": "0"},
        ],
    )

    output_csv = tmp_path / "per_glomerulus_predictions.csv"
    score_dir = tmp_path / "scores"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "aggregate_hspn_inference_for_scoring.py"),
            "--manifest-csv",
            str(manifest),
            "--mesangial-classifier-csv",
            str(mesangial),
            "--endocapillary-classifier-csv",
            str(endocapillary),
            "--output-csv",
            str(output_csv),
            "--score-output-dir",
            str(score_dir),
            "--min-glomeruli",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8-sig")))
    assert rows[0]["mesangial_prob"] == "1.000000"
    assert rows[1]["endocapillary_prob"] == "1.000000"
    assert rows[2]["mesangial_prob"] == "0.000000"

    summary = json.load((tmp_path / "per_glomerulus_predictions.summary.json").open("r", encoding="utf-8"))
    assert summary["matched_rows"]["mesangial_classifier"] == 4
    assert summary["matched_rows"]["endocapillary_classifier"] == 4

    slide_scores = list(csv.DictReader((score_dir / "slide_scores.csv").open("r", encoding="utf-8-sig")))
    assert slide_scores[0]["M"] == "M1"
    assert slide_scores[0]["E"] == "E1"


def test_aggregate_scorer_uses_custom_me_thresholds(tmp_path):
    manifest = tmp_path / "crops.csv"
    write_rows(
        manifest,
        [
            {
                "slide_id": "case_03",
                "glomerulus_index": index,
                "crop_id": f"case_03_g{index:04d}_10_{index * 20}",
                "bbox_x_min": index * 10,
                "bbox_y_min": index * 20,
                "bbox_x_max": index * 10 + 50,
                "bbox_y_max": index * 20 + 50,
            }
            for index in range(1, 5)
        ],
    )
    proliferation = tmp_path / "proliferation_multilabel.csv"
    write_rows(
        proliferation,
        [
            {"crop_id": "case_03_g0001_10_20", "mesangial_prob": "0.70", "endocapillary_prob": "0.44"},
            {"crop_id": "case_03_g0002_10_40", "mesangial_prob": "0.70", "endocapillary_prob": "0.46"},
            {"crop_id": "case_03_g0003_10_60", "mesangial_prob": "0.20", "endocapillary_prob": "0.10"},
            {"crop_id": "case_03_g0004_10_80", "mesangial_prob": "0.10", "endocapillary_prob": "0.10"},
        ],
    )

    output_csv = tmp_path / "per_glomerulus_predictions.csv"
    score_dir = tmp_path / "scores"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "aggregate_hspn_inference_for_scoring.py"),
            "--manifest-csv",
            str(manifest),
            "--proliferation-csv",
            str(proliferation),
            "--output-csv",
            str(output_csv),
            "--score-output-dir",
            str(score_dir),
            "--min-glomeruli",
            "1",
            "--mesangial-threshold",
            "0.75",
            "--endocapillary-threshold",
            "0.45",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    summary = json.load((tmp_path / "per_glomerulus_predictions.summary.json").open("r", encoding="utf-8"))
    assert summary["thresholds"]["mesangial"] == 0.75
    assert summary["thresholds"]["endocapillary"] == 0.45
    assert summary["thresholds"]["crescent"] == 0.30
    assert summary["iskdc_crescent_mode"] == "all"

    slide_scores = list(csv.DictReader((score_dir / "slide_scores.csv").open("r", encoding="utf-8-sig")))
    assert slide_scores[0]["M"] == "M0"
    assert slide_scores[0]["E"] == "E1"
