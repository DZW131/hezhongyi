"""Grad-CAM heatmap generation and hard-negative mining for the proliferation classifier.

Given a trained ResNet18 binary classifier, this script:

1. Runs the classifier on a chosen split (default: val) of glomerulus crops.
2. Computes Grad-CAM heatmaps for the "proliferation" class (class 1) on every crop.
3. Identifies **hard negatives**: true-negative crops that the classifier falsely
   predicts as positive (prob_positive > threshold). These are exactly the crops
   where the model "sees something that looks like proliferation but isn't".
4. Saves per-crop overlay images (original + heatmap) and a CSV report sorted by
   prediction confidence, so the doctor can review *where* the model is looking.

The Grad-CAM target layer defaults to ``layer4`` (the last conv block of ResNet),
which gives the coarsest but most semantically meaningful localization.
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
        description="Grad-CAM hard-negative mining for the proliferation classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to best.pth from the classifier")
    parser.add_argument("--data-root", required=True,
                        help="Crop dataset root containing <split>/images and <split>/masks")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--output-dir", required=True, help="Directory for heatmap outputs")
    parser.add_argument("--target-class", type=int, default=1,
                        help="Class index to compute Grad-CAM for (1 = proliferation)")
    parser.add_argument("--positive-threshold", type=float, default=0.5,
                        help="Probability threshold to consider a crop predicted positive")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Maximum number of hard-negative overlays to save")
    parser.add_argument("--save-all-overlays", action="store_true", default=False,
                        help="Save overlays for all crops, not just hard negatives")
    parser.add_argument("--montage-max", type=int, default=24,
                        help="Max items in the summary montage")
    parser.add_argument("--cam-method", default="layercam",
                        choices=("gradcam", "gradcampp", "xgradcam", "layercam"),
                        help="CAM variant. layercam is sharpest, gradcam is the classic coarse baseline.")
    parser.add_argument("--target-layer", default="layer4",
                        choices=("layer4", "layer3", "layer2"),
                        help="ResNet block to hook. layer3/layer2 are higher-resolution (14x14/28x28) "
                             "and give finer localization at the cost of less semantic abstraction.")
    parser.add_argument("--sharpen", type=float, default=0.3,
                        help="Unsharp-mask strength applied after upsampling. 0 disables. "
                             "Typical 0.2-0.5 sharpens blurry layer4 maps.")
    parser.add_argument("--bbox-threshold", type=float, default=0.4,
                        help="Heatmap threshold for bbox extraction. Lower = bigger box, "
                             "higher = tighter box around peak activation. Range (0,1).")
    parser.add_argument("--bbox-min-area", type=int, default=16,
                        help="Minimum connected-component area (pixels) to keep a bbox.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "resnet18")
    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights=None)
    else:
        model = torchvision.models.resnet34(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, NUM_CLASSES := len(CLASS_NAMES))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


class GradCAM:
    """Grad-CAM family for a ResNet-style model, supporting sharper variants.

    Methods (selectable via ``--cam-method``):
      - gradcam   : classic Grad-CAM, weights = mean of gradients. Coarse, may diffuse.
      - gradcampp : Grad-CAM++, uses second-order gradients for sharper, better
                    localization on multiple targets. Good default when gradcam is blurry.
      - xgradcam  : XGrad-CAM, weights from element-wise grad*act products then
                    normalized. More robust to noise, often sharper than gradcam.
      - layercam  : LayerCAM, weights = ReLU(gradient) element-wise * activation,
                    then sum. Produces the sharpest, most pixel-precise maps because it
                    keeps per-location gradient sign instead of averaging.

    For ResNet18 at 224 input, layer4 outputs 7x7. All methods upsample to the input
    size with bicubic interpolation and optional gaussian sharpening to recover detail.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module,
                 method: str = "gradcam"):
        self.model = model
        self.target_layer = target_layer
        self.method = method
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self._fwd.remove()
        self._bwd.remove()

    def _compute_weights(self) -> torch.Tensor:
        """Compute per-channel weights according to the selected method."""
        grads = self.gradients
        acts = self.activations
        if self.method == "gradcam":
            return grads.mean(dim=(2, 3), keepdim=True)
        if self.method == "gradcampp":
            # Grad-CAM++: alpha_k approximated by softmax of normalized positive grads
            grads_pow2 = grads.pow(2)
            grads_pow3 = grads_pow2 * grads
            denom = 2.0 * grads_pow2 + (acts * grads_pow3).sum(dim=(2, 3), keepdim=True)
            denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
            alpha = grads_pow2 / denom
            alpha = F.relu(grads) * alpha
            return alpha.sum(dim=(2, 3), keepdim=True) / max(1, alpha.shape[1])
        if self.method == "xgradcam":
            product = grads * acts
            weights = product.sum(dim=(2, 3), keepdim=True)
            denom = grads.sum(dim=(2, 3), keepdim=True).abs() + 1e-8
            return weights / denom
        if self.method == "layercam":
            # weights are per-location: relu(grad) * act, summed over channels later
            return F.relu(grads)
        return grads.mean(dim=(2, 3), keepdim=True)

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(input_tensor)
        target = logits[:, target_class]
        target.backward()

        if self.activations is None or self.gradients is None:
            return np.zeros((1, 1), dtype=np.float32)

        weights = self._compute_weights()
        if self.method == "layercam":
            # element-wise weighting, then channel sum
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
        else:
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bicubic", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam


def _sharpen_cam(cam: np.ndarray, amount: float) -> np.ndarray:
    """Apply unsharp masking to a normalized [0,1] heatmap to recover edge detail
    lost during low-resolution upsampling. amount in [0,1] controls the strength."""
    if amount <= 0:
        return cam
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return cam
    blurred = gaussian_filter(cam.astype(np.float32), sigma=1.0)
    sharpened = cam.astype(np.float32) + amount * (cam.astype(np.float32) - blurred)
    sharpened = np.clip(sharpened, 0.0, 1.0)
    s_min, s_max = sharpened.min(), sharpened.max()
    if s_max - s_min > 1e-8:
        sharpened = (sharpened - s_min) / (s_max - s_min)
    return sharpened


def compute_cam_bbox(cam: np.ndarray, threshold: float = 0.4,
                      min_area: int = 16) -> Optional[Tuple[int, int, int, int]]:
    """Derive a localization bbox from the heatmap by thresholding + largest connected
    component. Returns (x0, y0, x1, y1) in image-space pixel coords, or None if no
    region exceeds the threshold/area. The bbox is drawn on top of the heatmap so the
    doctor sees both the precise activation shape and a clean enclosing rectangle.
    """
    if cam.size == 0 or cam.max() < threshold:
        return None
    try:
        from scipy.ndimage import label, find_objects
    except ImportError:
        return None
    binary = (cam >= threshold).astype(np.uint8)
    labeled, num = label(binary)
    if num == 0:
        return None
    # pick the largest connected component
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    largest = int(sizes.argmax())
    if sizes[largest] < min_area:
        return None
    slices = find_objects(labeled)
    y_slice, x_slice = slices[largest - 1]
    x0, x1 = int(x_slice.start), int(x_slice.stop)
    y0, y1 = int(y_slice.start), int(y_slice.stop)
    return x0, y0, x1, y1


def draw_bbox_on_image(image: Image.Image, bbox: Optional[Tuple[int, int, int, int]],
                        color: Tuple[int, int, int] = (255, 60, 0),
                        width: int = 3, label: Optional[str] = None) -> Image.Image:
    """Draw a bbox rectangle on a copy of the image. If label is given, draw it above the box."""
    out = image.copy()
    if bbox is None:
        return out
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = bbox
    for w in range(width):
        draw.rectangle([x0 - w, y0 - w, x1 + w, y1 + w], outline=color)
    if label:
        # label background bar
        tw, th = draw.textbbox((0, 0), label)[2:]
        draw.rectangle([x0, max(0, y0 - th - 4), x0 + tw + 6, y0], fill=color)
        draw.text((x0 + 3, max(0, y0 - th - 3)), label, fill=(255, 255, 255))
    return out


def load_image(path: Path, device: torch.device) -> Tuple[torch.Tensor, Image.Image]:
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = transforms.ToTensor()(image)
    tensor = NORMALIZE(tensor).unsqueeze(0).to(device=device, dtype=torch.float32)
    return tensor, image


def _jet_colormap(value: float) -> Tuple[int, int, int]:
    """Map a scalar in [0, 1] to a jet-color RGB tuple."""
    value = max(0.0, min(1.0, float(value)))
    if value < 0.125:
        r, g, b = 0.0, 0.0, 0.5 + value * 4.0
    elif value < 0.375:
        r, g, b = 0.0, (value - 0.125) * 4.0, 1.0
    elif value < 0.625:
        r, g, b = (value - 0.375) * 4.0, 1.0, 1.0 - (value - 0.375) * 4.0
    elif value < 0.875:
        r, g, b = 1.0, 1.0 - (value - 0.625) * 4.0, 0.0
    else:
        r, g, b = 1.0 - (value - 0.875) * 4.0, 0.0, 0.0
    return int(r * 255), int(g * 255), int(b * 255)


def _apply_jet(cam_flat: np.ndarray) -> np.ndarray:
    """Vectorized jet colormap for a flat [N] array in [0,1] -> [N,3] uint8."""
    v = np.clip(cam_flat, 0.0, 1.0)
    r = np.zeros_like(v); g = np.zeros_like(v); b = np.zeros_like(v)
    m1 = v < 0.125; r[m1] = 0; g[m1] = 0; b[m1] = 0.5 + v[m1] * 4.0
    m2 = (v >= 0.125) & (v < 0.375); r[m2] = 0; g[m2] = (v[m2] - 0.125) * 4.0; b[m2] = 1.0
    m3 = (v >= 0.375) & (v < 0.625); r[m3] = (v[m3] - 0.375) * 4.0; g[m3] = 1.0; b[m3] = 1.0 - (v[m3] - 0.375) * 4.0
    m4 = (v >= 0.625) & (v < 0.875); r[m4] = 1.0; g[m4] = 1.0 - (v[m4] - 0.625) * 4.0; b[m4] = 0.0
    m5 = v >= 0.875; r[m5] = 1.0 - (v[m5] - 0.875) * 4.0; g[m5] = 0.0; b[m5] = 0.0
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay a jet-colored Grad-CAM heatmap onto the image.

    The blend alpha is intensity-weighted: high-activation pixels are opaque red/yellow,
    low-activation pixels stay close to the original image. This makes the region the
    model is looking at clearly visible while preserving anatomical context elsewhere.
    """
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    ).astype(np.float32) / 255.0
    cam_flat = cam_resized.reshape(-1)
    jet_rgb = _apply_jet(cam_flat).reshape(*cam_resized.shape, 3)
    base = np.array(image, dtype=np.float32)
    # intensity-weighted alpha: low activation ~0, high activation ~alpha
    local_alpha = (cam_resized[..., None] * alpha).clip(0, alpha)
    overlay = base * (1.0 - local_alpha) + jet_rgb * local_alpha
    return Image.fromarray(overlay.astype(np.uint8))


def label_from_mask(data_root: Path, split: str, image_name: str) -> int:
    mask_path = data_root / split / "masks" / (Path(image_name).stem + ".png")
    if not mask_path.exists():
        return 0
    mask = np.array(Image.open(mask_path))
    return int(mask.max() > 0)


def build_montage(records: List[Tuple[Image.Image, Image.Image, Image.Image, str]],
                   output_path: Path, cell_size: int = 224, cols: int = 4) -> None:
    """Build a 3-column-per-item montage: original | heatmap | overlay.

    Each record is (original, heatmap_only, overlay, caption).
    """
    if not records:
        return
    triplets = cols
    rows = int(np.ceil(len(records) / triplets))
    label_h = 32
    cell_w = cell_size
    cell_h = cell_size + label_h
    canvas = Image.new("RGB", (triplets * 3 * cell_w, rows * cell_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (orig, heat, overlay, cap) in enumerate(records):
        col = idx % triplets
        row = idx // triplets
        x0 = col * 3 * cell_w
        y0 = row * cell_h
        for j, img in enumerate((orig, heat, overlay)):
            canvas.paste(img.resize((cell_w, cell_size)), (x0 + j * cell_w, y0))
        draw.rectangle((x0, y0 + cell_size, x0 + 3 * cell_w, y0 + cell_size + label_h), fill=(30, 30, 30))
        draw.text((x0 + 4, y0 + cell_size + 8), cap, fill=(255, 220, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    images_dir = data_root / args.split / "images"
    if not images_dir.exists():
        raise FileNotFoundError("Missing images dir: {}".format(images_dir))
    image_paths = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise RuntimeError("No images in {}".format(images_dir))

    model = load_model(Path(args.model), device)
    layer_module = getattr(model, args.target_layer)
    target_layer = layer_module[-1]
    cam_extractor = GradCAM(model, target_layer, method=args.cam_method)
    logging.info("CAM method=%s target_layer=%s sharpen=%.2f",
                 args.cam_method, args.target_layer, args.sharpen)

    rows: List[dict] = []
    hard_neg_records: List[Tuple[Image.Image, Image.Image, Image.Image, str, float]] = []

    logging.info("Running Grad-CAM on %s crops from %s/%s ...", len(image_paths), data_root, args.split)

    for path in image_paths:
        tensor, pil_image = load_image(path, device)
        with torch.enable_grad():
            cam = cam_extractor.generate(tensor, args.target_class)
        cam = _sharpen_cam(cam, args.sharpen)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        prob_pos = float(probs[1])
        pred = 1 if prob_pos > args.positive_threshold else 0
        true_label = label_from_mask(data_root, args.split, path.name)
        is_hard_neg = (true_label == 0 and pred == 1)

        cam_center = None
        if cam.shape[0] > 0 and cam.shape[1] > 0:
            ys, xs = np.unravel_index(np.argmax(cam), cam.shape)
            cam_center = [int(xs), int(ys)]

        cam_bbox = compute_cam_bbox(cam, threshold=args.bbox_threshold, min_area=args.bbox_min_area)
        bbox_x0, bbox_y0, bbox_x1, bbox_y1 = (-1, -1, -1, -1)
        bbox_w, bbox_h = 0, 0
        if cam_bbox is not None:
            bbox_x0, bbox_y0, bbox_x1, bbox_y1 = cam_bbox
            bbox_w, bbox_h = bbox_x1 - bbox_x0, bbox_y1 - bbox_y0

        rows.append({
            "name": path.name,
            "true_label": true_label,
            "pred_label": pred,
            "prob_positive": round(prob_pos, 4),
            "prob_background": round(float(probs[0]), 4),
            "is_hard_negative": int(is_hard_neg),
            "cam_max": round(float(cam.max()), 4),
            "cam_center_x": cam_center[0] if cam_center else "",
            "cam_center_y": cam_center[1] if cam_center else "",
            "bbox_x0": bbox_x0,
            "bbox_y0": bbox_y0,
            "bbox_x1": bbox_x1,
            "bbox_y1": bbox_y1,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
        })

        if is_hard_neg:
            overlay = overlay_heatmap(pil_image, cam)
            heatmap_only = Image.fromarray(
                _apply_jet(cam.reshape(-1)).reshape(*cam.shape, 3)
            ).resize(pil_image.size, Image.BILINEAR)
            # draw bbox on original and overlay (same coords), label with prob
            box_label = "p={:.2f}".format(prob_pos)
            orig_boxed = draw_bbox_on_image(pil_image, cam_bbox, color=(255, 60, 0), label=box_label)
            overlay_boxed = draw_bbox_on_image(overlay, cam_bbox, color=(255, 60, 0), label=box_label)
            gt_str = "neg" if true_label == 0 else "pos"
            pred_str = "pos" if pred == 1 else "neg"
            cap = "{} | GT={} pred={} p={:.2f}".format(path.stem, gt_str, pred_str, prob_pos)
            hard_neg_records.append((orig_boxed, heatmap_only, overlay_boxed, cap, prob_pos))

    cam_extractor.remove_hooks()

    hard_neg_records.sort(key=lambda x: x[4], reverse=True)
    rows.sort(key=lambda r: (r["is_hard_negative"] == 0, -r["prob_positive"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "hard_negatives.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0].keys()) if rows else
                                ["name", "true_label", "pred_label", "prob_positive"])
        writer.writeheader()
        writer.writerows(rows)

    hard_neg_count = sum(1 for r in rows if r["is_hard_negative"])
    logging.info("Found %s hard negatives (true=neg, pred=pos) out of %s crops", hard_neg_count, len(rows))

    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    for orig_boxed, heat, overlay_boxed, cap, prob in hard_neg_records[:args.top_k]:
        name = cap.split(" ")[0]
        # save side-by-side: original+box | heatmap | overlay+box
        side_by_side = Image.new("RGB", (pil_image.size[0] * 3, pil_image.size[1]), color=(255, 255, 255))
        side_by_side.paste(orig_boxed, (0, 0))
        side_by_side.paste(heat, (pil_image.size[0], 0))
        side_by_side.paste(overlay_boxed, (pil_image.size[0] * 2, 0))
        side_by_side.save(overlays_dir / "{}_compare.jpg".format(name), quality=95)

    if hard_neg_records:
        build_montage(
            [(o, he, ov, c) for o, he, ov, c, _ in hard_neg_records[:args.montage_max]],
            output_dir / "hard_negatives_montage.jpg",
        )
        logging.info("Saved 3-column montage to %s", output_dir / "hard_negatives_montage.jpg")

    summary = {
        "total_crops": len(rows),
        "hard_negatives": hard_neg_count,
        "positive_threshold": args.positive_threshold,
        "model": str(args.model),
        "data_root": str(data_root),
        "split": args.split,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as h:
        json.dump(summary, h, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()