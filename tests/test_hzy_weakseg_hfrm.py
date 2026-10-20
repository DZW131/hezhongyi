import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hzy_weakseg_hfrm import (  # noqa: E402
    HspnWeakSegDataset,
    WeakSegLossConfig,
    make_model,
    save_evidence_overlay,
    target_specs_for_task,
    weakseg_loss,
)


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_task_root(tmp_path):
    root = tmp_path / "proliferation"
    rows = []
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "masks").mkdir(parents=True)
        for index in range(2):
            stem = "{}_g{:04d}".format(split, index + 1)
            image = np.full((64, 64, 3), 235, dtype=np.uint8)
            image[12:44, 12:44] = [185, 95, 170]
            mask = np.zeros((64, 64), dtype=np.uint8)
            if index == 0:
                mask[18:34, 18:34] = 1
                class_pixels = {"1": 256, "2": 0}
            else:
                mask[28:46, 28:46] = 2
                class_pixels = {"1": 0, "2": 324}
            Image.fromarray(image).save(root / split / "images" / "{}.jpg".format(stem))
            Image.fromarray(mask).save(root / split / "masks" / "{}.png".format(stem))
            rows.append(
                {
                    "slide_id": split,
                    "split": split,
                    "glomerulus_index": index + 1,
                    "crop_id": stem,
                    "image_path": "{}/images/{}.jpg".format(split, stem),
                    "mask_path": "{}/masks/{}.png".format(split, stem),
                    "class_pixels": json.dumps(class_pixels),
                }
            )
    _write_rows(root / "manifests" / "crops.csv", rows)
    return root


def test_weakseg_dataset_parses_multilabel_targets(tmp_path):
    root = _make_task_root(tmp_path)
    dataset = HspnWeakSegDataset(root, task="proliferation", split="train", image_size=64, augmentation="off")

    assert len(dataset) == 2
    assert dataset[0]["image"].shape == (3, 64, 64)
    assert dataset[0]["target"].tolist() == [1.0, 0.0]
    assert dataset[1]["target"].tolist() == [0.0, 1.0]
    assert dataset.label_summary()["mesangial_positive"] == 1


def test_weakseg_dataset_parses_single_proliferation_targets(tmp_path):
    root = _make_task_root(tmp_path)

    mesangial_ds = HspnWeakSegDataset(root, task="mesangial", split="train", image_size=64)
    endocapillary_ds = HspnWeakSegDataset(root, task="endocapillary", split="train", image_size=64)

    assert mesangial_ds[0]["target"].tolist() == [1.0]
    assert mesangial_ds[1]["target"].tolist() == [0.0]
    assert endocapillary_ds[0]["target"].tolist() == [0.0]
    assert endocapillary_ds[1]["target"].tolist() == [1.0]


def test_hfrm_cam_model_forward_and_loss_are_finite():
    specs = target_specs_for_task("proliferation")
    model = make_model(len(specs), backbone="resnet18", pretrained=False, context_kernel=7, cam_weights=(0.2, 0.3, 0.5))
    images = torch.randn(2, 3, 64, 64)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    priors = torch.ones(2, 1, 64, 64)

    output = model(images)
    loss, parts = weakseg_loss(
        output,
        targets,
        priors,
        torch.nn.BCEWithLogitsLoss(),
        WeakSegLossConfig(cls_weights=(0.2, 0.3, 0.5)),
        image_size=(64, 64),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert output["fused_logits"].shape == (2, 2)
    assert output["image_logits"].shape == (2, 2)
    assert "loss_score_consistency" in parts


def test_hfrm_cam_model_aux_classifier_backbone_forward():
    specs = target_specs_for_task("proliferation")
    model = make_model(
        len(specs),
        backbone="resnet18",
        pretrained=False,
        context_kernel=7,
        cam_weights=(0.2, 0.3, 0.5),
        classifier_backbone="efficientnet_v2_s",
        classifier_weight=0.9,
        classifier_dropout=0.1,
        classifier_input="raw",
        cam_pool="avg_topk",
        cam_pool_topk=0.1,
    )
    model.eval()

    with torch.no_grad():
        cams, probs = model.fused_cam(torch.randn(1, 3, 96, 96), output_size=(96, 96))

    assert cams.shape == (1, 2, 96, 96)
    assert probs.shape == (1, 2)


def test_evidence_overlay_writer(tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8)).save(image_path)
    cams = np.zeros((2, 64, 64), dtype=np.float32)
    cams[0, 10:30, 10:30] = 1.0
    cams[1, 35:50, 35:50] = 0.8
    out = tmp_path / "overlay.png"

    save_evidence_overlay(
        image_path,
        cams,
        np.asarray([0.9, 0.7], dtype=np.float32),
        target_specs_for_task("proliferation"),
        {"mesangial": 0.5, "endocapillary": 0.5},
        out,
    )

    assert out.exists()


def test_aggregate_accepts_weakseg_csv(tmp_path):
    manifest = tmp_path / "crops.csv"
    _write_rows(
        manifest,
        [
            {"slide_id": "case", "split": "val", "glomerulus_index": 1, "crop_id": "g1"},
            {"slide_id": "case", "split": "val", "glomerulus_index": 2, "crop_id": "g2"},
        ],
    )
    weakseg = tmp_path / "weakseg.csv"
    _write_rows(
        weakseg,
        [
            {"crop_id": "g1", "mesangial_prob": "0.8", "endocapillary_prob": "0.1"},
            {"crop_id": "g2", "mesangial_prob": "0.2", "endocapillary_prob": "0.9"},
        ],
    )
    output = tmp_path / "per_glomerulus.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "aggregate_hspn_inference_for_scoring.py"),
            "--manifest-csv",
            str(manifest),
            "--weakseg-csv",
            str(weakseg),
            "--output-csv",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    rows = list(csv.DictReader(output.open("r", encoding="utf-8-sig")))
    assert completed.returncode == 0
    assert rows[0]["mesangial_prob"] == "0.800000"
    assert rows[1]["endocapillary_prob"] == "0.900000"


def test_aggregate_prefers_slide_glomerulus_composite_match(tmp_path):
    manifest = tmp_path / "crops.csv"
    _write_rows(
        manifest,
        [
            {"slide_id": "case_a", "split": "val", "glomerulus_index": 1, "crop_id": "case_a_crop"},
            {"slide_id": "case_b", "split": "val", "glomerulus_index": 1, "crop_id": "case_b_crop"},
        ],
    )
    weakseg = tmp_path / "weakseg.csv"
    _write_rows(
        weakseg,
        [
            {"slide_id": "case_a", "glomerulus_id": "g0001", "crop_id": "different_a", "mesangial_prob": "0.8"},
            {"slide_id": "case_b", "glomerulus_id": "g0001", "crop_id": "different_b", "mesangial_prob": "0.2"},
        ],
    )
    output = tmp_path / "per_glomerulus.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "aggregate_hspn_inference_for_scoring.py"),
            "--manifest-csv",
            str(manifest),
            "--weakseg-csv",
            str(weakseg),
            "--output-csv",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    rows = list(csv.DictReader(output.open("r", encoding="utf-8-sig")))
    assert rows[0]["mesangial_prob"] == "0.800000"
    assert rows[1]["mesangial_prob"] == "0.200000"
