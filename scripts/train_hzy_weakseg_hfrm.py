"""Train SSHR-style HFRM-CAM weak segmentation models for HSPN lesions."""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hzy_weakseg_hfrm import (  # noqa: E402
    FocalBCEWithLogitsLoss,
    HspnWeakSegDataset,
    TARGET_SPECS,
    WeakSegLossConfig,
    best_thresholds_by_f1,
    classification_metrics,
    connected_boxes,
    cam_binary_mask,
    fused_cam_with_tta,
    make_model,
    mask_metrics_from_counts,
    positive_weight,
    save_evidence_overlay,
    save_json,
    target_specs_for_task,
    threshold_grid,
    update_mask_counts,
    weakseg_loss,
    write_csv_rows,
)


def parse_cam_weights(text: str) -> Sequence[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("--cam-weights must contain exactly three comma-separated values")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an HFRM-CAM weakly supervised lesion segmentation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=sorted(TARGET_SPECS.keys()), required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Task root containing manifests/crops.csv")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", choices=("resnet18", "resnet34", "resnet50"), default="resnet18")
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--context-kernel", type=int, default=15)
    parser.add_argument("--cam-weights", type=parse_cam_weights, default=(0.2, 0.3, 0.5))
    parser.add_argument("--cam-pool", choices=("avg", "topk", "avg_topk"), default="avg_topk")
    parser.add_argument("--cam-pool-topk", type=float, default=0.08)
    parser.add_argument("--classifier-weight", type=float, default=0.45)
    parser.add_argument("--classifier-dropout", type=float, default=0.2)
    parser.add_argument("--classifier-backbone", choices=("shared", "efficientnet_v2_s", "efficientnet_v2_m", "resnet18", "resnet34"), default="shared")
    parser.add_argument("--classifier-init-model", type=Path, default=None, help="Optional classifier checkpoint used to initialize auxiliary features/head")
    parser.add_argument("--classifier-init-class-index", type=int, default=-1, help="Class row to copy from a multi-output classifier head; -1 copies only exact shape matches")
    parser.add_argument("--classifier-input", choices=("normalized", "raw"), default="normalized")
    parser.add_argument("--freeze-classifier-epochs", type=int, default=0)
    parser.add_argument("--loss", choices=("bce", "focal_bce"), default="focal_bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--class-weights", choices=("balanced", "none"), default="balanced")
    parser.add_argument("--weighted-sampler", action="store_true", default=False)
    parser.add_argument("--augmentation", choices=("off", "basic", "strong"), default="basic")
    parser.add_argument("--min-positive-pixels", type=int, default=8)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--mask-threshold", type=float, default=0.35)
    parser.add_argument("--mask-min-area", type=int, default=8)
    parser.add_argument("--mask-open-radius", type=int, default=0)
    parser.add_argument("--mask-close-radius", type=int, default=0)
    parser.add_argument("--mask-fill-holes", action="store_true", default=False)
    parser.add_argument("--mask-keep-components", type=int, default=0)
    parser.add_argument("--mask-topk-area", type=float, default=0.0)
    parser.add_argument("--mask-use-prior", action="store_true", default=False)
    parser.add_argument("--checkpoint-metric", choices=("macro_f1", "mean_auc", "mean_dice", "exact_match"), default="macro_f1")
    parser.add_argument("--cam-cls-weight", type=float, default=1.0)
    parser.add_argument("--image-cls-weight", type=float, default=0.8)
    parser.add_argument("--fused-cls-weight", type=float, default=0.4)
    parser.add_argument("--prior-weight", type=float, default=0.08)
    parser.add_argument("--negative-cam-weight", type=float, default=0.05)
    parser.add_argument("--area-weight", type=float, default=0.03)
    parser.add_argument("--score-consistency-weight", type=float, default=0.20)
    parser.add_argument("--positive-area-target", type=float, default=0.08)
    parser.add_argument("--topk-ratio", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--eval-tta", action="store_true", default=False)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--save-val-overlays", type=int, default=24, help="Number of validation overlays to save for the best checkpoint")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def sample_weights(targets: np.ndarray) -> List[float]:
    pos_counts = targets.sum(axis=0)
    neg_counts = targets.shape[0] - pos_counts
    label_weights = neg_counts / np.maximum(pos_counts, 1.0)
    weights: List[float] = []
    for row in targets:
        if row.sum() == 0:
            weights.append(1.0)
        else:
            weights.append(float(1.0 + (row * label_weights).sum()))
    return weights


def make_loader(dataset: HspnWeakSegDataset, args: argparse.Namespace, train: bool) -> DataLoader:
    if train and args.weighted_sampler:
        sampler = WeightedRandomSampler(sample_weights(dataset.targets), num_samples=len(dataset), replacement=True)
        return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=args.num_workers, pin_memory=True)


def set_classifier_branch_frozen(model: torch.nn.Module, frozen: bool) -> None:
    for name, parameter in model.named_parameters():
        if name.startswith("aux_classifier_features.") or name.startswith("image_head."):
            parameter.requires_grad = not frozen


def set_frozen_classifier_branch_eval(model: torch.nn.Module, frozen: bool) -> None:
    if not frozen:
        return
    aux_features = getattr(model, "aux_classifier_features", None)
    if aux_features is not None:
        aux_features.eval()
    image_head = getattr(model, "image_head", None)
    if image_head is not None:
        image_head.eval()


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    specs,
    thresholds: Dict[str, float],
    mask_threshold: float,
    eval_tta: bool = False,
    mask_min_area: int = 8,
    mask_open_radius: int = 0,
    mask_close_radius: int = 0,
    mask_fill_holes: bool = False,
    mask_keep_components: int = 0,
    mask_topk_area: float = 0.0,
    mask_use_prior: bool = False,
) -> Dict[str, object]:
    model.eval()
    y_true: List[np.ndarray] = []
    y_prob: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []
    mask_counts = {spec.slug: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for spec in specs}
    pointing_hits = {spec.slug: 0 for spec in specs}
    pointing_total = {spec.slug: 0 for spec in specs}
    has_masks = False

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            targets = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
            cams, probs = fused_cam_with_tta(model, images, output_size=tuple(batch["mask"].shape[-2:]), eval_tta=eval_tta)
            y_true.append(targets.cpu().numpy().astype(np.int32))
            y_prob.append(probs.cpu().numpy().astype(np.float32))
            cams_np = cams.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            masks_np = batch["mask"].cpu().numpy()
            priors_np = batch["prior"].cpu().numpy()
            targets_np = targets.cpu().numpy().astype(np.int32)

            if (masks_np >= 0).any():
                has_masks = True
            for item_idx, crop_id in enumerate(batch["crop_id"]):
                row: Dict[str, object] = {"crop_id": crop_id}
                for class_idx, spec in enumerate(specs):
                    prob = float(probs_np[item_idx, class_idx])
                    pred = int(prob >= thresholds[spec.slug])
                    row["{}_true".format(spec.slug)] = int(targets_np[item_idx, class_idx])
                    row["{}_prob".format(spec.slug)] = prob
                    row["{}_pred".format(spec.slug)] = pred
                    if (masks_np[item_idx] >= 0).any():
                        gt_mask = masks_np[item_idx] == spec.class_id
                        prior = priors_np[item_idx, 0] if mask_use_prior and priors_np.ndim == 4 else None
                        pred_mask = cam_binary_mask(
                            cams_np[item_idx, class_idx],
                            mask_threshold=mask_threshold,
                            active=bool(pred),
                            prior=prior,
                            use_prior=mask_use_prior,
                            min_area=mask_min_area,
                            open_radius=mask_open_radius,
                            close_radius=mask_close_radius,
                            fill_holes=mask_fill_holes,
                            keep_components=mask_keep_components,
                            topk_area=mask_topk_area,
                        )
                        update_mask_counts(mask_counts[spec.slug], pred_mask, gt_mask)
                        if gt_mask.any():
                            pointing_total[spec.slug] += 1
                            yx = np.unravel_index(int(np.argmax(cams_np[item_idx, class_idx])), cams_np[item_idx, class_idx].shape)
                            if gt_mask[yx]:
                                pointing_hits[spec.slug] += 1
                rows.append(row)

    yt = np.concatenate(y_true, axis=0)
    yp = np.concatenate(y_prob, axis=0)
    cls_metrics = classification_metrics(yt, yp, specs, thresholds)
    seg_metrics = {}
    if has_masks:
        dice_values = []
        iou_values = []
        for spec in specs:
            metrics = mask_metrics_from_counts(mask_counts[spec.slug])
            total = pointing_total[spec.slug]
            metrics["pointing_game"] = float(pointing_hits[spec.slug] / total) if total else None
            seg_metrics[spec.slug] = metrics
            dice_values.append(metrics["dice"])
            iou_values.append(metrics["iou"])
        seg_metrics["mean_dice"] = float(np.mean(dice_values)) if dice_values else 0.0
        seg_metrics["mean_iou"] = float(np.mean(iou_values)) if iou_values else 0.0
    return {"classification": cls_metrics, "segmentation": seg_metrics, "predictions": rows}


def save_history(history: List[Dict[str, object]], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_best_overlays(dataset: HspnWeakSegDataset, model: torch.nn.Module, device: torch.device, output_dir: Path, args, thresholds) -> None:
    if args.save_val_overlays <= 0:
        return
    specs = target_specs_for_task(args.task)
    rows = []
    model.eval()
    with torch.no_grad():
        for index in range(min(len(dataset), args.save_val_overlays)):
            sample = dataset[index]
            image = sample["image"].unsqueeze(0).to(device)
            cams, probs = fused_cam_with_tta(model, image, output_size=tuple(sample["mask"].shape[-2:]), eval_tta=args.eval_tta)
            cams_np = cams[0].detach().cpu().numpy()
            probs_np = probs[0].detach().cpu().numpy()
            out_path = output_dir / "analysis" / "val_overlays" / "{}.png".format(sample["crop_id"])
            save_evidence_overlay(
                Path(str(sample["image_path"])),
                cams_np,
                probs_np,
                specs,
                thresholds,
                out_path,
                mask_threshold=args.mask_threshold,
                min_area=args.mask_min_area,
                open_radius=args.mask_open_radius,
                close_radius=args.mask_close_radius,
                fill_holes=args.mask_fill_holes,
                keep_components=args.mask_keep_components,
                topk_area=args.mask_topk_area,
                use_prior=args.mask_use_prior,
            )
            row = {"crop_id": sample["crop_id"], "overlay_path": str(out_path)}
            for class_idx, spec in enumerate(specs):
                pred_mask = cam_binary_mask(
                    cams_np[class_idx],
                    mask_threshold=args.mask_threshold,
                    active=float(probs_np[class_idx]) >= float(thresholds[spec.slug]),
                    prior=sample["prior"][0].numpy() if args.mask_use_prior else None,
                    use_prior=args.mask_use_prior,
                    min_area=args.mask_min_area,
                    open_radius=args.mask_open_radius,
                    close_radius=args.mask_close_radius,
                    fill_holes=args.mask_fill_holes,
                    keep_components=args.mask_keep_components,
                    topk_area=args.mask_topk_area,
                )
                row["{}_area".format(spec.slug)] = float(pred_mask.mean())
                row["{}_boxes".format(spec.slug)] = len(connected_boxes(pred_mask))
            rows.append(row)
    write_csv_rows(rows, output_dir / "analysis" / "val_overlay_index.csv")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    set_seed(args.seed)
    device = resolve_device(args.device)
    specs = target_specs_for_task(args.task)
    args.num_classes = len(specs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), output_dir / "analysis" / "args.json")

    train_ds = HspnWeakSegDataset(
        args.data_root,
        task=args.task,
        split="train",
        image_size=args.image_size,
        min_positive_pixels=args.min_positive_pixels,
        manifest=args.manifest,
        augmentation=args.augmentation,
        limit=args.limit_train,
    )
    val_ds = HspnWeakSegDataset(
        args.data_root,
        task=args.task,
        split="val",
        image_size=args.image_size,
        min_positive_pixels=args.min_positive_pixels,
        manifest=args.manifest,
        augmentation="off",
        limit=args.limit_val,
    )
    save_json({"train": train_ds.label_summary(), "val": val_ds.label_summary()}, output_dir / "analysis" / "label_summary.json")
    train_loader = make_loader(train_ds, args, train=True)
    val_loader = make_loader(val_ds, args, train=False)

    model = make_model(
        num_classes=len(specs),
        backbone=args.backbone,
        pretrained=args.pretrained,
        context_kernel=args.context_kernel,
        cam_weights=args.cam_weights,
        classifier_weight=args.classifier_weight,
        classifier_dropout=args.classifier_dropout,
        classifier_backbone=args.classifier_backbone,
        classifier_init_model=args.classifier_init_model,
        classifier_init_class_index=args.classifier_init_class_index,
        classifier_input=args.classifier_input,
        cam_pool=args.cam_pool,
        cam_pool_topk=args.cam_pool_topk,
    ).to(device)
    pos_weight = positive_weight(train_ds.targets, device) if args.class_weights == "balanced" else None
    criterion = FocalBCEWithLogitsLoss(args.focal_gamma, pos_weight=pos_weight) if args.loss == "focal_bce" else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_config = WeakSegLossConfig(
        cls_weights=tuple(float(v) for v in args.cam_weights),
        cam_cls_weight=args.cam_cls_weight,
        image_cls_weight=args.image_cls_weight,
        fused_cls_weight=args.fused_cls_weight,
        prior_weight=args.prior_weight,
        negative_cam_weight=args.negative_cam_weight,
        area_weight=args.area_weight,
        score_consistency_weight=args.score_consistency_weight,
        positive_area_target=args.positive_area_target,
        topk_ratio=args.topk_ratio,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    grid = threshold_grid(args.threshold_step)
    thresholds = {spec.slug: 0.5 for spec in specs}

    logging.info("Weakseg task=%s data=%s device=%s train=%d val=%d", args.task, args.data_root, device, len(train_ds), len(val_ds))
    logging.info("Targets: %s", [spec.slug for spec in specs])
    logging.info("Label summary: %s", {"train": train_ds.label_summary(), "val": val_ds.label_summary()})

    best_score = -1.0
    best_epoch = 0
    history: List[Dict[str, object]] = []
    best_metrics: Dict[str, object] = {}

    for epoch in range(1, args.epochs + 1):
        classifier_frozen = epoch <= max(int(args.freeze_classifier_epochs), 0)
        set_classifier_branch_frozen(model, classifier_frozen)
        model.train()
        set_frozen_classifier_branch_eval(model, classifier_frozen)
        start = time.perf_counter()
        train_loss = 0.0
        train_seen = 0
        loss_parts_sum: Dict[str, float] = {}
        for batch in train_loader:
            images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            targets = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
            priors = batch["prior"].to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                output = model(images)
                loss, parts = weakseg_loss(
                    output,
                    targets,
                    priors,
                    criterion,
                    loss_config,
                    image_size=tuple(images.shape[-2:]),
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(images.shape[0])
            train_loss += float(loss.item()) * batch_size
            train_seen += batch_size
            for key, value in parts.items():
                loss_parts_sum[key] = loss_parts_sum.get(key, 0.0) + float(value) * batch_size

        raw_eval = evaluate(
            model,
            val_loader,
            device,
            specs,
            thresholds,
            args.mask_threshold,
            eval_tta=args.eval_tta,
            mask_min_area=args.mask_min_area,
            mask_open_radius=args.mask_open_radius,
            mask_close_radius=args.mask_close_radius,
            mask_fill_holes=args.mask_fill_holes,
            mask_keep_components=args.mask_keep_components,
            mask_topk_area=args.mask_topk_area,
            mask_use_prior=args.mask_use_prior,
        )
        yt = np.asarray([[row["{}_true".format(spec.slug)] for spec in specs] for row in raw_eval["predictions"]], dtype=np.int32)
        yp = np.asarray([[row["{}_prob".format(spec.slug)] for spec in specs] for row in raw_eval["predictions"]], dtype=np.float32)
        thresholds = best_thresholds_by_f1(yt, yp, specs, grid)
        eval_result = evaluate(
            model,
            val_loader,
            device,
            specs,
            thresholds,
            args.mask_threshold,
            eval_tta=args.eval_tta,
            mask_min_area=args.mask_min_area,
            mask_open_radius=args.mask_open_radius,
            mask_close_radius=args.mask_close_radius,
            mask_fill_holes=args.mask_fill_holes,
            mask_keep_components=args.mask_keep_components,
            mask_topk_area=args.mask_topk_area,
            mask_use_prior=args.mask_use_prior,
        )
        cls = eval_result["classification"]
        seg = eval_result["segmentation"]
        mean_dice = float(seg.get("mean_dice", 0.0)) if isinstance(seg, dict) else 0.0
        score_lookup = {
            "macro_f1": float(cls["macro_f1"]),
            "mean_auc": float(cls["mean_auc"]),
            "exact_match": float(cls["exact_match"]),
            "mean_dice": mean_dice,
        }
        score = score_lookup[args.checkpoint_metric]
        row: Dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_seen, 1),
            "macro_f1": cls["macro_f1"],
            "exact_match": cls["exact_match"],
            "mean_auc": cls["mean_auc"],
            "mean_dice": mean_dice,
            "mean_iou": float(seg.get("mean_iou", 0.0)) if isinstance(seg, dict) else 0.0,
            "epoch_seconds": time.perf_counter() - start,
            "classifier_frozen": int(classifier_frozen),
        }
        for key, value in loss_parts_sum.items():
            row[key] = value / max(train_seen, 1)
        for spec in specs:
            metrics = cls["per_class"][spec.slug]
            row["{}_f1".format(spec.slug)] = metrics["f1"]
            row["{}_recall".format(spec.slug)] = metrics["recall"]
            row["{}_specificity".format(spec.slug)] = metrics["specificity"]
            row["{}_auc".format(spec.slug)] = metrics.get("auc", "")
            row["{}_threshold".format(spec.slug)] = thresholds[spec.slug]
            if seg:
                row["{}_dice".format(spec.slug)] = seg[spec.slug]["dice"]
                row["{}_iou".format(spec.slug)] = seg[spec.slug]["iou"]
                row["{}_pointing".format(spec.slug)] = seg[spec.slug]["pointing_game"]
        history.append(row)
        save_history(history, output_dir / "analysis" / "history.csv")

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "target_names": [spec.slug for spec in specs],
            "thresholds": thresholds,
            "metrics": eval_result,
            "args": vars(args),
        }
        torch.save(checkpoint, output_dir / "latest.pth")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = eval_result
            torch.save(checkpoint, output_dir / "best.pth")
            save_json({"best_epoch": best_epoch, "score": best_score, "metrics": best_metrics}, output_dir / "analysis" / "best_metrics.json")
            write_csv_rows(eval_result["predictions"], output_dir / "analysis" / "best_predictions.csv")
            save_best_overlays(val_ds, model, device, output_dir, args, thresholds)

        logging.info(
            "Epoch %d/%d loss=%.4f macro_f1=%.4f mean_dice=%.4f score=%.4f best=%d",
            epoch,
            args.epochs,
            row["train_loss"],
            row["macro_f1"],
            row["mean_dice"],
            score,
            best_epoch,
        )

    save_json({"best_epoch": best_epoch, "best_score": best_score, "best_metrics": best_metrics}, output_dir / "analysis" / "run_summary.json")
    print("Best epoch:", best_epoch)
    print("Best score:", "{:.6f}".format(best_score))
    print("Output dir:", output_dir)


if __name__ == "__main__":
    main()
