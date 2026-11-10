"""Fine-tune DPA-P2PNet with ROI-aware supervision and frozen case splits."""

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from hzy_intrinsic_cell_upgrade import (CLASSES, CellDataset, ROIClassificationLoss,
                                        best_validation_postprocess, read_rows, score_cached)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "baseline", "train", "evaluate"), default="train")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recipe", choices=("data_control", "recall_mesangial"), default="recall_mesangial")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--head-only-epochs", type=int, default=2)
    parser.add_argument("--scope", choices=("late", "heads"), default="late")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--memory-fraction", type=float, default=0.26)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--max-crops", type=int, default=0, help="Smoke/debug only; full runs use all crops")
    return parser.parse_args()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def csv_write(path, rows):
    if rows:
        with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def log(args, event, **values):
    item = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **values}
    print(json.dumps(item, ensure_ascii=False), flush=True)
    with (args.output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def amp(args):
    return torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()


def build_model(args):
    sys.path.insert(0, str(args.source_root.resolve()))
    from models.dpa_p2p.dpa_p2pnet import build_dpa_model
    model_args = SimpleNamespace(backbone="vit_adapter", ssl_type="", ssl_path="", hidden_dim=256,
                                 num_levels=3, num_fov=0, dropout=0.1, space=8)
    model = build_dpa_model(model_args, num_classes=len(CLASSES))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    for module in model.modules():
        if hasattr(module, "with_cp"):
            module.with_cp = True
    return model.cuda(), checkpoint


def trainable_scope(model, scope):
    for name, parameter in model.named_parameters():
        head = not name.startswith("backbone.")
        late = name.startswith(("backbone.neck.", "backbone.backbone.blocks.10.", "backbone.backbone.blocks.11."))
        parameter.requires_grad = head or (scope == "late" and late)
    model.train()
    # Retain pretrained batch statistics in frozen and small-batch fine-tuning layers.
    model.backbone.eval()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict(model, dataset, args):
    model.eval()
    cached = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=list,
                        pin_memory=True)
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            sample = batch[0]
            with amp(args):
                outputs = model(sample["image"].unsqueeze(0).cuda(non_blocking=True))
            cached.append({"coords": outputs["pnt_coords"][0].float().cpu().numpy(),
                           "scores": outputs["cls_logits"][0].float().softmax(-1).cpu().numpy(),
                           "points": sample["points"].numpy(), "labels": sample["labels"].numpy(),
                           "roi": sample["roi"].numpy(), "row": sample["row"]})
            if (index + 1) % 25 == 0:
                log(args, "evaluation_progress", split=dataset.split, crops=index + 1)
            if args.max_crops and index + 1 >= args.max_crops:
                break
    return cached


def save_report(args, prefix, cached, metrics, rows, predictions):
    json_write(args.output_dir / (prefix + "_metrics.json"), metrics)
    csv_write(args.output_dir / (prefix + "_per_crop.csv"), rows)
    json_write(args.output_dir / (prefix + "_predictions.json"), predictions)
    grouped = {}
    for row in rows:
        case = grouped.setdefault(row["case_key"], {"case_key": row["case_key"], "crops": 0,
                                                   "disease": row["disease"]})
        case["crops"] += 1
        for key, value in row.items():
            if isinstance(value, (int, float)):
                case[key] = case.get(key, 0) + value
    csv_write(args.output_dir / (prefix + "_per_case.csv"), list(grouped.values()))
    splits = {r["case_key"]: r for r in read_rows(args.dataset / "case_splits.csv")}
    cohorts = {}
    for name, new in (("new_cases", True), ("legacy_cases", False)):
        subset = [item for item in cached if (splits[item["row"]["case_key"]]["legacy_split"] == "new") == new]
        if subset:
            cohorts[name] = score_cached(subset, metrics["foreground_ratio"], metrics["dedup_distance"])[0]
    json_write(args.output_dir / (prefix + "_cohort_metrics.json"), cohorts)
    # Show all selected examples by a fixed ranking rule, including low-performing cases.
    if not rows:
        return
    ranked = sorted(range(len(rows)), key=lambda i: abs(rows[i]["predicted_total"] - rows[i]["true_total"]))
    selected = list(dict.fromkeys((ranked[0], ranked[len(ranked) // 2], ranked[-1])))
    for index in selected:
        item, pred = cached[index], predictions[index]
        image_path = args.dataset / (args.split + "_image") / (item["row"]["crop_id"] + ".jpg")
        if not image_path.exists():
            image_path = args.dataset / "val_image" / (item["row"]["crop_id"] + ".jpg")
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        target, estimate = image.copy(), image.copy()
        colors = ((54, 67, 244), (44, 160, 44), (231, 166, 0))
        for point, label in zip(item["points"], item["labels"]):
            cv2.circle(target, tuple(np.rint(point).astype(int)), 5, colors[int(label)], 1)
        for point, label in zip(pred["points"], pred["classes"]):
            cv2.drawMarker(estimate, tuple(np.rint(point).astype(int)), colors[int(label)], cv2.MARKER_CROSS, 8, 1)
        contour = cv2.findContours(item["roi"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for panel in (target, estimate):
            cv2.drawContours(panel, contour, -1, (240, 190, 20), 2)
        panel = cv2.resize(np.hstack((target, estimate)), (1024, 512))
        cv2.imwrite(str(args.output_dir / (prefix + "_" + item["row"]["crop_id"] + "_evidence.png")), panel)


def checkpoint_save(path, model, args, epoch, metrics, optimizer=None):
    result = {"model": model.state_dict(), "epoch": epoch, "metrics": metrics,
              "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "split_sha256": json.loads((args.dataset / "dataset_summary.json").read_text())["split_sha256"],
              "normalization": np.load(args.dataset / "mean_std.npy").tolist()}
    if optimizer:
        result["optimizer"] = optimizer.state_dict()
    temporary = path.with_suffix(".partial")
    torch.save(result, temporary)
    temporary.replace(path)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for DPA-P2PNet training")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    json_write(args.output_dir / "config.json", config)
    model, previous = build_model(args)
    log(args, "loaded_checkpoint", previous_epoch=previous.get("epoch"),
        gpu_free_bytes=torch.cuda.mem_get_info()[0], memory_fraction=args.memory_fraction)
    dataset = CellDataset(args.dataset, "train", augment=True)
    validation = CellDataset(args.dataset, "val")
    if args.mode in ("baseline", "evaluate"):
        if args.split == "test" and args.calibration is None:
            raise ValueError("Test evaluation requires frozen validation calibration")
        cached = predict(model, CellDataset(args.dataset, args.split), args)
        if args.mode == "baseline":
            fixed, _, _ = score_cached(cached, ratio=1, dedup=10)
            json_write(args.output_dir / "val_legacy_postprocess_metrics.json", fixed)
            selected, grid = best_validation_postprocess(cached)
            json_write(args.output_dir / "calibration.json", {"foreground_ratio": selected["foreground_ratio"],
                                                              "dedup_distance": selected["dedup_distance"],
                                                              "source": "validation_only"})
            json_write(args.output_dir / "validation_postprocess_grid.json", grid)
        else:
            selected = json.loads(args.calibration.read_text()) if args.calibration else {"foreground_ratio": 1, "dedup_distance": 6}
        metrics, rows, preds = score_cached(cached, selected["foreground_ratio"], selected["dedup_distance"])
        save_report(args, args.split, cached, metrics, rows, preds)
        log(args, "evaluation_complete", split=args.split, metrics=metrics)
        return
    sys.path.insert(0, str(args.source_root.resolve()))
    from matcher import HungarianMatcher
    if args.recipe == "data_control":
        criterion = ROIClassificationLoss(HungarianMatcher(0.1, 1.0), focal_gamma=0,
                                          mesangial_weight=1, background_weight=1, hard_negative_ratio=0)
        sampler = None
    else:
        criterion = ROIClassificationLoss(HungarianMatcher(0.1, 1.0))
        sampler = WeightedRandomSampler(dataset.sampling_weights(), len(dataset), replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed))
    head = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    encoder = [p for name, p in model.named_parameters() if name.startswith("backbone.")]
    optimizer = torch.optim.AdamW([{"params": head, "lr": args.learning_rate},
                                   {"params": encoder, "lr": args.learning_rate * args.encoder_lr_scale}],
                                  weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.workers, collate_fn=list, pin_memory=True)
    if args.mode == "smoke":
        for scope in ("heads", args.scope):
            params = trainable_scope(model, scope)
            sample = next(iter(loader))[0]
            optimizer.zero_grad(set_to_none=True)
            with amp(args):
                outputs = model(sample["image"].unsqueeze(0).cuda())
                loss, details = criterion(outputs, sample)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite smoke loss")
            loss.backward()
            optimizer.step()
            log(args, "smoke_forward_backward_ok", scope=scope, parameters=params,
                loss=float(loss.detach()), details=details, peak_memory=torch.cuda.max_memory_allocated())
            torch.cuda.empty_cache()
        return
    # Initialize best with the unchanged checkpoint; a degraded fine-tune is never promoted.
    cached = predict(model, validation, args)
    best_metrics, _, _ = score_cached(cached)
    best_score = best_metrics["selection_score"]
    checkpoint_save(args.output_dir / "best.pth", model, args, -1, best_metrics)
    log(args, "initial_validation", metrics=best_metrics)
    history = []
    stale = 0
    for epoch in range(args.epochs):
        scope = "heads" if epoch < args.head_only_epochs else args.scope
        params = trainable_scope(model, scope)
        decay = 0.5 * (1 + math.cos(math.pi * epoch / max(args.epochs, 1)))
        optimizer.param_groups[0]["lr"] = args.learning_rate * max(decay, 0.1)
        optimizer.param_groups[1]["lr"] = args.learning_rate * args.encoder_lr_scale * max(decay, 0.1)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        start = time.monotonic()
        for step, batch in enumerate(loader):
            sample = batch[0]
            with amp(args):
                outputs = model(sample["image"].unsqueeze(0).cuda(non_blocking=True))
                loss, details = criterion(outputs, sample)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss at epoch {} step {}".format(epoch, step))
            (loss / args.accumulate).backward()
            if (step + 1) % args.accumulate == 0 or step + 1 == len(loader):
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.1)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if (step + 1) % 50 == 0:
                log(args, "training_progress", epoch=epoch, step=step + 1, loss=float(np.mean(losses)), scope=scope)
            if args.max_crops and step + 1 >= args.max_crops:
                break
        cached = predict(model, validation, args)
        metrics, _, _ = score_cached(cached)
        improved = metrics["selection_score"] > best_score + 1e-4
        stale = 0 if improved else stale + 1
        if improved:
            best_score = metrics["selection_score"]
            best_metrics = metrics
            checkpoint_save(args.output_dir / "best.pth", model, args, epoch, metrics)
        checkpoint_save(args.output_dir / "recent.pth", model, args, epoch, metrics, optimizer)
        history.append({"epoch": epoch, "scope": scope, "trainable_parameters": params,
                        "loss": float(np.mean(losses)), "seconds": time.monotonic() - start,
                        "detection_f1": metrics["detection"]["f1"], "recall": metrics["detection"]["recall"],
                        "mesangial_f1": metrics["per_class"]["mesangial_cell"]["f1"],
                        "macro_f1": metrics["macro_f1"], "selection_score": metrics["selection_score"]})
        csv_write(args.output_dir / "history.csv", history)
        log(args, "epoch_complete", **history[-1], improved=improved, stale_epochs=stale,
            peak_memory=torch.cuda.max_memory_allocated())
        if stale >= args.patience and epoch >= args.head_only_epochs:
            log(args, "early_stopping", epoch=epoch)
            break
    best = torch.load(args.output_dir / "best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    del best
    cached = predict(model, validation, args)
    selected, grid = best_validation_postprocess(cached)
    json_write(args.output_dir / "calibration.json", {"foreground_ratio": selected["foreground_ratio"],
                                                      "dedup_distance": selected["dedup_distance"], "source": "validation_only"})
    json_write(args.output_dir / "validation_postprocess_grid.json", grid)
    metrics, rows, preds = score_cached(cached, selected["foreground_ratio"], selected["dedup_distance"])
    save_report(args, "val", cached, metrics, rows, preds)
    log(args, "training_complete", best_fixed_metrics=best_metrics, best_calibrated_metrics=metrics)


if __name__ == "__main__":
    main()
