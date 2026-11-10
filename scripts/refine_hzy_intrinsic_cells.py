"""Train a local/context cell classifier after frozen DPA point proposals."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

from hzy_intrinsic_cell_upgrade import CLASSES, CellDataset, score_cached, select_points
from train_hzy_intrinsic_cell_upgrade import build_model, predict, save_report
from calibrate_hzy_intrinsic_cells import validation_objective


def cell_patches(image, points, sides=(64, 192), size=64):
    patches = []
    for x, y in points:
        views = []
        for side in sides:
            view = cv2.getRectSubPix(image, (side, side), (float(x), float(y)))
            views.append(cv2.resize(view, (size, size), interpolation=cv2.INTER_AREA))
        patches.append(np.concatenate(views, axis=2).transpose(2, 0, 1))
    return np.stack(patches) if patches else np.zeros((0, 6, size, size), dtype=np.uint8)


def metadata_features(points, probabilities, roi):
    yy, xx = np.where(roi)
    if not len(xx):
        raise ValueError("Empty annotation ROI")
    center = np.array([(xx.min() + xx.max()) / 2, (yy.min() + yy.max()) / 2])
    scale = max(xx.max() - xx.min(), yy.max() - yy.min(), 1)
    offsets = (np.asarray(points) - center) / scale
    radius = np.sqrt((offsets * offsets).sum(1, keepdims=True))
    return np.concatenate((probabilities, offsets, radius), axis=1).astype(np.float32)


class ContextClassifier(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.encoder = resnet18(weights=None)
        if pretrained:
            self.encoder.load_state_dict(torch.load(pretrained, map_location="cpu", weights_only=True))
        weight = self.encoder.conv1.weight.detach().clone()
        self.encoder.conv1 = nn.Conv2d(6, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.encoder.conv1.weight.copy_(torch.cat((weight, weight), dim=1) * 0.5)
        self.encoder.fc = nn.Identity()
        self.head = nn.Sequential(nn.Linear(519, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 4))
        self.register_buffer("mean", torch.tensor([.485, .456, .406] * 2).view(1, 6, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225] * 2).view(1, 6, 1, 1))

    def forward(self, patches, extra):
        patches = patches.float() / 255
        features = self.encoder((patches - self.mean) / self.std)
        return self.head(torch.cat((features, extra.float()), dim=1))


class PatchDataset(Dataset):
    def __init__(self, patches, metadata, labels):
        self.patches, self.metadata, self.labels = patches, metadata, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (torch.from_numpy(np.array(self.patches[index], copy=True)),
                torch.from_numpy(self.metadata[index]), int(self.labels[index]))


def get_raw(args, split):
    if split == "val":
        existing = args.proposal_cache / "upgraded_val_raw.pt"
        if existing.exists():
            return torch.load(existing, map_location="cpu", weights_only=False)
    path = args.output_dir / (split + "_dpa_raw.pt")
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    args.split = split
    args.checkpoint = args.dpa_checkpoint
    model, old = build_model(args)
    del old
    cached = predict(model, CellDataset(args.dataset, split), args)
    del model
    torch.cuda.empty_cache()
    torch.save(cached, path)
    return cached


def proposals(cached):
    selected = []
    for item in cached:
        xy, labels = select_points(item["coords"], item["scores"], item["roi"], 0.4, 4.0)
        # Query scores follow the selected proposal coordinates; nearest query is deterministic.
        _, indexes = cKDTree(item["coords"]).query(xy, k=1)
        selected.append({**item, "coords": xy.astype(np.float32), "scores": item["scores"][indexes]})
    return selected


def image_for(args, item, split):
    path = args.dataset / (split + "_image") / (item["row"]["crop_id"] + ".jpg")
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_training(args, raw):
    meta_path = args.output_dir / "training_patches_metadata.pt"
    patch_path = args.output_dir / "training_patches.npy"
    if meta_path.exists() and patch_path.exists():
        meta = torch.load(meta_path, weights_only=False)
        return np.load(patch_path, mmap_mode="r"), meta["metadata"], meta["labels"]
    rng = np.random.default_rng(args.seed)
    records = []
    for item in proposals(raw):
        truth, labels = item["points"], item["labels"]
        if not len(truth):
            continue
        candidates = item["coords"]
        distances, closest = cKDTree(truth).query(candidates, k=1)
        positive = np.flatnonzero(distances <= 5)
        negative = np.flatnonzero(distances >= 12)
        if len(positive) > len(truth) // 2:
            positive = rng.choice(positive, max(1, len(truth) // 2), replace=False)
        if len(negative) > max(10, len(truth) // 2):
            ranking = item["scores"][negative, :3].max(1).argsort()[::-1]
            negative = negative[ranking[:max(10, len(truth) // 2)]]
        coordinates = np.concatenate((truth, candidates[positive], candidates[negative]))
        targets = np.concatenate((labels, labels[closest[positive]], np.full(len(negative), 3)))
        _, score_indexes = cKDTree(candidates).query(coordinates)
        probabilities = item["scores"][score_indexes]
        records.append((item, coordinates, targets.astype(np.int64),
                        metadata_features(coordinates, probabilities, item["roi"])))
    total = sum(len(r[1]) for r in records)
    patches = np.lib.format.open_memmap(patch_path, mode="w+", dtype=np.uint8, shape=(total, 6, 64, 64))
    metadata, labels = [], []
    cursor = 0
    for index, (item, coordinates, targets, extra) in enumerate(records):
        image = image_for(args, item, "train")
        patches[cursor:cursor + len(coordinates)] = cell_patches(image, coordinates)
        cursor += len(coordinates)
        metadata.append(extra)
        labels.append(targets)
        if (index + 1) % 50 == 0:
            print(json.dumps({"event": "prepare_patches", "crops": index + 1, "patches": cursor}), flush=True)
    patches.flush()
    metadata, labels = np.concatenate(metadata), np.concatenate(labels)
    torch.save({"metadata": metadata, "labels": labels}, meta_path)
    print(json.dumps({"event": "training_patch_counts", "counts": dict(Counter(labels.tolist()))}), flush=True)
    return np.load(patch_path, mmap_mode="r"), metadata, labels


def prepare_eval(args, cached, split):
    prepared = []
    for item in proposals(cached):
        image = image_for(args, item, split)
        patches = cell_patches(image, item["coords"])
        metadata = metadata_features(item["coords"], item["scores"], item["roi"])
        prepared.append((item, patches, metadata))
    return prepared


def refine(model, prepared, args):
    model.eval()
    cached = []
    with torch.inference_mode():
        for item, patches, extra in prepared:
            scores = []
            for offset in range(0, len(patches), args.batch_size):
                x = torch.from_numpy(patches[offset:offset + args.batch_size]).cuda()
                m = torch.from_numpy(extra[offset:offset + args.batch_size]).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x, m)
                scores.append(logits.float().softmax(-1).cpu().numpy())
            output = np.concatenate(scores) if scores else np.empty((0, 4), dtype=np.float32)
            cached.append({**item, "cnn_scores": output})
    return cached


def combine(cached, beta):
    return [{**item, "scores": beta * item["cnn_scores"] + (1 - beta) * item["scores"]} for item in cached]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fit", "evaluate"), default="fit")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dpa-checkpoint", type=Path, required=True)
    parser.add_argument("--imagenet-checkpoint", type=Path, required=True)
    parser.add_argument("--proposal-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    args.max_crops, args.precision = 0, "bf16"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.25)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.mode == "evaluate":
        frozen = json.loads((args.output_dir / "frozen_calibration.json").read_text())
        if frozen["source_split"] != "val":
            raise ValueError("Calibration was not frozen on validation")
        model = ContextClassifier().cuda()
        checkpoint = torch.load(args.output_dir / "best.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        cached = refine(model, prepare_eval(args, get_raw(args, "test"), "test"), args)
        args.split = "test"
        combined = combine(cached, frozen["beta"])
        metrics, rows, preds = score_cached(combined, frozen["ratio"], frozen["dedup"])
        save_report(args, "test", combined, metrics, rows, preds)
        print(json.dumps({"event": "retrospective_test_complete", "metrics": metrics}), flush=True)
        return
    raw_train = get_raw(args, "train")
    patches, metadata, labels = prepare_training(args, raw_train)
    del raw_train
    raw_val = get_raw(args, "val")
    reference = score_cached(raw_val, 1.2, 10)[0]
    prepared_val = prepare_eval(args, raw_val, "val")
    train = PatchDataset(patches, metadata, labels)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    model = ContextClassifier(args.imagenet_checkpoint).cuda()
    counts = np.bincount(labels, minlength=4)
    weights = torch.from_numpy(np.sqrt(counts.sum() / np.maximum(counts * 4, 1))).float().cuda()
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 1e-5},
                                   {"params": model.head.parameters(), "lr": 3e-4}], weight_decay=1e-3)
    best_score, stale, history = -1.0, 0, []
    for epoch in range(args.epochs):
        model.train()
        for parameter in model.encoder.parameters():
            parameter.requires_grad = epoch > 0
        if epoch == 0:
            model.encoder.eval()
        losses = []
        for x, extra, y in loader:
            x, extra, y = x.cuda(non_blocking=True), extra.cuda(non_blocking=True), y.cuda(non_blocking=True)
            if random.random() < .5:
                x = x.flip(-1)
            if random.random() < .5:
                x = x.flip(-2)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, extra)
                loss = nn.functional.cross_entropy(logits.float(), y, weight=weights, label_smoothing=0.03)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite refinement loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        cached = refine(model, prepared_val, args)
        metrics = score_cached(combine(cached, .75), 1.0, 8)[0]
        objective, nmae = validation_objective(metrics, reference)
        # Early epochs may not meet the recall guard; retain the best available
        # validation checkpoint, while final promotion still applies all guards.
        rank = (0.35 * metrics["detection"]["f1"] + .45 * metrics["macro_f1"] +
                .20 * metrics["per_class"]["mesangial_cell"]["f1"] - .05 * nmae)
        improved = rank > best_score
        stale = 0 if improved else stale + 1
        if improved:
            best_score = rank
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation": metrics,
                        "training_only_class_counts": counts.tolist()}, args.output_dir / "best.pth")
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "rank": rank,
               "guarded_objective": objective, "metrics": metrics, "improved": improved}
        history.append(row)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps({"event": "epoch_complete", **row}), flush=True)
        if stale >= 3 and epoch >= 3:
            break
    checkpoint = torch.load(args.output_dir / "best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    cached = refine(model, prepared_val, args)
    torch.save(cached, args.output_dir / "val_refined_raw.pt")
    trials = []
    for beta in (.5, .75, 1.0):
        for ratio in (.7, 1.0, 1.4):
            for dedup in (6.0, 8.0, 10.0):
                metrics = score_cached(combine(cached, beta), ratio, dedup)[0]
                objective, nmae = validation_objective(metrics, reference)
                trials.append({"beta": beta, "ratio": ratio, "dedup": dedup,
                               "objective": objective, "normalized_count_mae": nmae, "metrics": metrics})
    best = max(trials, key=lambda row: row["objective"])
    chosen = {key: best[key] for key in ("beta", "ratio", "dedup", "objective")}
    chosen.update({"source_split": "val", "test_used_for_selection": False,
                   "training_cases": 29, "validation_cases": 7})
    (args.output_dir / "frozen_calibration.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    (args.output_dir / "validation_search.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")
    combined = combine(cached, best["beta"])
    metrics, rows, preds = score_cached(combined, best["ratio"], best["dedup"])
    args.split = "val"
    save_report(args, "val", combined, metrics, rows, preds)
    print(json.dumps({"event": "refinement_fit_complete", "chosen": chosen, "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
