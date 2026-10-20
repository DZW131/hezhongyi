"""Evaluate Grad-CAM localization quality against doctor-annotated masks.

This script answers the question: "Is the heatmap pointing at the region the
doctor actually annotated as proliferation?"

It runs on POSITIVE crops only (where a doctor mask exists), computes three
localization metrics, and saves a montage for visual sanity check:

1. Pointing Game: does the peak activation pixel fall inside the doctor mask?
2. Energy Percentage: what fraction of total heatmap energy lies inside the mask?
3. IoU: overlap between thresholded heatmap and doctor mask.

A high pointing-hit-rate / energy-percentage means the classifier learned real
proliferation visual features (not shortcuts), which indirectly validates that
the hard-negative heatmaps are also pointing at meaningful regions.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

try:
    import torchvision
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torchvision is required: {}".format(exc))


CLASS_NAMES = ["background", "proliferation"]
IMAGE_SIZE = 224
NORMALIZE = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Grad-CAM localization against doctor masks on positive crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", required=True,
                        help="Crop dataset root containing <split>/{images,masks}")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cam-method", default="layercam",
                        choices=("gradcam", "gradcampp", "xgradcam", "layercam"))
    parser.add_argument("--target-layer", default="layer3",
                        choices=("layer4", "layer3", "layer2"))
    parser.add_argument("--sharpen", type=float, default=0.3)
    parser.add_argument("--target-class", type=int, default=1)
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="Heatmap binarization threshold for IoU computation.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--montage-max", type=int, default=24,
                        help="Max positive crops in the visual sanity-check montage.")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "resnet18")
    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights=None)
    else:
        model = torchvision.models.resnet34(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(CLASS_NAMES))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


class GradCAM:
    def __init__(self, model, target_layer, method="layercam"):
        self.model = model
        self.target_layer = target_layer
        self.method = method
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _m, _i, output):
        self.activations = output.detach()

    def _save_gradient(self, _m, _gi, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()

    def _compute_weights(self):
        grads = self.gradients
        acts = self.activations
        if self.method == "gradcam":
            return grads.mean(dim=(2, 3), keepdim=True)
        if self.method == "gradcampp":
            g2 = grads.pow(2)
            g3 = g2 * grads
            denom = 2.0 * g2 + (acts * g3).sum(dim=(2, 3), keepdim=True)
            denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
            alpha = g2 / denom
            alpha = F.relu(grads) * alpha
            return alpha.sum(dim=(2, 3), keepdim=True) / max(1, alpha.shape[1])
        if self.method == "xgradcam":
            product = grads * acts
            w = product.sum(dim=(2, 3), keepdim=True)
            d = grads.sum(dim=(2, 3), keepdim=True).abs() + 1e-8
            return w / d
        if self.method == "layercam":
            return F.relu(grads)
        return grads.mean(dim=(2, 3), keepdim=True)

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        logits[:, target_class].backward()
        if self.activations is None or self.gradients is None:
            return np.zeros((1, 1), dtype=np.float32)
        w = self._compute_weights()
        cam = (w * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bicubic", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cmin, cmax = cam.min(), cam.max()
        if cmax - cmin > 1e-8:
            cam = (cam - cmin) / (cmax - cmin)
        else:
            cam = np.zeros_like(cam)
        return cam


def _sharpen_cam(cam, amount):
    if amount <= 0:
        return cam
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return cam
    blurred = gaussian_filter(cam.astype(np.float32), sigma=1.0)
    s = cam.astype(np.float32) + amount * (cam.astype(np.float32) - blurred)
    s = np.clip(s, 0.0, 1.0)
    smin, smax = s.min(), s.max()
    if smax - smin > 1e-8:
        s = (s - smin) / (smax - smin)
    return s


def load_image(path, device):
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = transforms.ToTensor()(image)
    tensor = NORMALIZE(tensor).unsqueeze(0).to(device=device, dtype=torch.float32)
    return tensor, image


def load_mask(path, size):
    """Load doctor mask, resize to target size with nearest-neighbor, return bool array."""
    mask = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return np.array(mask) > 0


def pointing_game(cam, mask_bool):
    """1 if the argmax pixel of cam falls inside mask, else 0."""
    if cam.size == 0:
        return 0
    ys, xs = np.unravel_index(np.argmax(cam), cam.shape)
    return int(mask_bool[ys, xs])


def energy_percentage(cam, mask_bool):
    """Fraction of total heatmap energy inside the mask."""
    total = cam.sum()
    if total <= 1e-8:
        return 0.0
    inside = cam[mask_bool].sum()
    return float(inside / total)


def cam_iou(cam, mask_bool, threshold):
    """IoU between binarized cam (>= threshold) and the doctor mask."""
    cam_bin = cam >= threshold
    inter = np.logical_and(cam_bin, mask_bool).sum()
    union = np.logical_or(cam_bin, mask_bool).sum()
    if union <= 0:
        return 0.0
    return float(inter / union)


def _apply_jet(cam_flat):
    v = np.clip(cam_flat, 0.0, 1.0)
    r = np.zeros_like(v); g = np.zeros_like(v); b = np.zeros_like(v)
    m1 = v < 0.125; r[m1] = 0; g[m1] = 0; b[m1] = 0.5 + v[m1] * 4.0
    m2 = (v >= 0.125) & (v < 0.375); r[m2] = 0; g[m2] = (v[m2] - 0.125) * 4.0; b[m2] = 1.0
    m3 = (v >= 0.375) & (v < 0.625); r[m3] = (v[m3] - 0.375) * 4.0; g[m3] = 1.0; b[m3] = 1.0 - (v[m3] - 0.375) * 4.0
    m4 = (v >= 0.625) & (v < 0.875); r[m4] = 1.0; g[m4] = 1.0 - (v[m4] - 0.625) * 4.0; b[m4] = 0.0
    m5 = v >= 0.875; r[m5] = 1.0 - (v[m5] - 0.875) * 4.0; g[m5] = 0.0; b[m5] = 0.0
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(image, cam, alpha=0.5):
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    ).astype(np.float32) / 255.0
    cam_flat = cam_resized.reshape(-1)
    jet_rgb = _apply_jet(cam_flat).reshape(*cam_resized.shape, 3)
    base = np.array(image, dtype=np.float32)
    local_alpha = (cam_resized[..., None] * alpha).clip(0, alpha)
    overlay = base * (1.0 - local_alpha) + jet_rgb * local_alpha
    return Image.fromarray(overlay.astype(np.uint8))


def draw_mask_contour(image, mask_bool, color=(0, 255, 0), width=2):
    """Draw a green contour around the doctor mask region on the image."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    h, w = mask_bool.shape
    # find boundary pixels (mask pixel with a non-mask neighbor)
    padded = np.pad(mask_bool, 1, constant_values=False)
    boundary = np.zeros_like(mask_bool)
    boundary = (mask_bool & (~padded[1:-1, 2:] | ~padded[1:-1, :-2] | ~padded[2:, 1:-1] | ~padded[:-2, 1:-1]))
    ys, xs = np.where(boundary)
    for x, y in zip(xs, ys):
        for d in range(width):
            draw.rectangle([x - d, y - d, x + d, y + d], outline=color)
    return out


def build_montage(records, output_path, cell_size=224, cols=4):
    """records: list of (overlay_with_contour, caption, hit)"""
    if not records:
        return
    rows = int(np.ceil(len(records) / cols))
    label_h = 28
    canvas = Image.new("RGB", (cols * cell_size, rows * (cell_size + label_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (img, cap, hit) in enumerate(records):
        x = (idx % cols) * cell_size
        y = (idx // cols) * (cell_size + label_h)
        canvas.paste(img.resize((cell_size, cell_size)), (x, y))
        bg = (0, 128, 0) if hit else (200, 0, 0)
        draw.rectangle((x, y + cell_size, x + cell_size, y + cell_size + label_h), fill=bg)
        draw.text((x + 4, y + cell_size + 6), cap, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    images_dir = data_root / args.split / "images"
    masks_dir = data_root / args.split / "masks"
    if not images_dir.exists():
        raise FileNotFoundError("Missing images dir: {}".format(images_dir))

    model = load_model(Path(args.model), device)
    layer_module = getattr(model, args.target_layer)
    cam_extractor = GradCAM(model, layer_module[-1], method=args.cam_method)
    logging.info("CAM method=%s target_layer=%s", args.cam_method, args.target_layer)

    image_paths = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rows = []
    montage_records = []

    for path in image_paths:
        mask_path = masks_dir / (path.stem + ".png")
        if not mask_path.exists():
            continue
        mask_full = np.array(Image.open(mask_path).convert("L"))
        if mask_full.max() == 0:
            continue  # skip negatives — we evaluate localization only on positives

        tensor, pil_image = load_image(path, device)
        mask_bool = load_mask(mask_path, IMAGE_SIZE)
        with torch.enable_grad():
            cam = cam_extractor.generate(tensor, args.target_class)
        cam = _sharpen_cam(cam, args.sharpen)

        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
        prob_pos = float(probs[1])

        hit = pointing_game(cam, mask_bool)
        energy = energy_percentage(cam, mask_bool)
        iou = cam_iou(cam, mask_bool, args.iou_threshold)
        mask_area_frac = float(mask_bool.mean())

        rows.append({
            "name": path.name,
            "prob_positive": round(prob_pos, 4),
            "pointing_hit": hit,
            "energy_percentage": round(energy, 4),
            "iou": round(iou, 4),
            "mask_area_fraction": round(mask_area_frac, 4),
            "pred_label": int(prob_pos > 0.5),
        })

        if len(montage_records) < args.montage_max:
            overlay = overlay_heatmap(pil_image, cam)
            overlay_with_contour = draw_mask_contour(overlay, mask_bool, color=(0, 255, 0), width=2)
            cap = "{} hit={} e={:.2f} iou={:.2f}".format(path.stem[:20], hit, energy, iou)
            montage_records.append((overlay_with_contour, cap, bool(hit)))

    cam_extractor.remove_hooks()

    # aggregate metrics
    n = len(rows)
    if n == 0:
        logging.warning("No positive crops found for evaluation.")
        return
    pointing_acc = sum(r["pointing_hit"] for r in rows) / n
    mean_energy = np.mean([r["energy_percentage"] for r in rows])
    mean_iou = np.mean([r["iou"] for r in rows])
    # energy ratio relative to mask area (a naive baseline): if cam were uniform, energy ~= mask_area
    mean_mask_area = np.mean([r["mask_area_fraction"] for r in rows])
    energy_lift = mean_energy / max(mean_mask_area, 1e-6)

    summary = {
        "num_positive_crops": n,
        "pointing_accuracy": round(pointing_acc, 4),
        "mean_energy_percentage": round(float(mean_energy), 4),
        "mean_iou": round(float(mean_iou), 4),
        "mean_mask_area_fraction": round(float(mean_mask_area), 4),
        "energy_lift_over_uniform": round(float(energy_lift), 4),
        "cam_method": args.cam_method,
        "target_layer": args.target_layer,
        "iou_threshold": args.iou_threshold,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as h:
        json.dump(summary, h, ensure_ascii=False, indent=2)
    with (output_dir / "per_crop.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if montage_records:
        build_montage(montage_records, output_dir / "localization_montage.jpg")

    logging.info("Pointing accuracy: %.4f (%d/%d)", pointing_acc, sum(r["pointing_hit"] for r in rows), n)
    logging.info("Mean energy%%: %.4f  (uniform baseline %.4f, lift %.2fx)", mean_energy, mean_mask_area, energy_lift)
    logging.info("Mean IoU: %.4f", mean_iou)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()