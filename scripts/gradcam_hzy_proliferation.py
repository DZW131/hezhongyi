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
    """Minimal Grad-CAM for a ResNet-style model.

    Registers forward and backward hooks on the target layer to capture activations
    and gradients, then produces a coarse localization map for the target class.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
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

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(input_tensor)
        target = logits[:, target_class]
        target.backward()

        if self.activations is None or self.gradients is None:
            return np.zeros((1, 1), dtype=np.float32)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam


def load_image(path: Path, device: torch.device) -> Tuple[torch.Tensor, Image.Image]:
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = transforms.ToTensor()(image)
    tensor = NORMALIZE(tensor).unsqueeze(0).to(device=device, dtype=torch.float32)
    return tensor, image


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    cam_uint8 = (cam * 255).astype(np.uint8)
    cam_img = Image.fromarray(cam_uint8)
    cam_img = cam_img.resize(image.size, Image.BILINEAR)
    cam_array = np.array(cam_img)
    heatmap = np.zeros((cam_array.shape[0], cam_array.shape[1], 3), dtype=np.uint8)
    heatmap[..., 0] = cam_array
    heatmap[..., 2] = (cam_array * 0.3).astype(np.uint8)
    overlay = Image.blend(image, Image.fromarray(heatmap), alpha)
    return overlay


def label_from_mask(data_root: Path, split: str, image_name: str) -> int:
    mask_path = data_root / split / "masks" / (Path(image_name).stem + ".png")
    if not mask_path.exists():
        return 0
    mask = np.array(Image.open(mask_path))
    return int(mask.max() > 0)


def build_montage(overlays: List[Image.Image], captions: List[str], output_path: Path,
                  cell_size: int = 224, cols: int = 4) -> None:
    if not overlays:
        return
    rows = int(np.ceil(len(overlays) / cols))
    canvas = Image.new("RGB", (cols * cell_size, rows * (cell_size + 28)), color=(255, 255, 255))
    for idx, (img, cap) in enumerate(zip(overlays, captions)):
        x = (idx % cols) * cell_size
        y = (idx // cols) * (cell_size + 28)
        thumb = img.resize((cell_size, cell_size))
        canvas.paste(thumb, (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((x, y + cell_size, x + cell_size, y + cell_size + 28), fill=(40, 40, 40))
        draw.text((x + 4, y + cell_size + 6), cap, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


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
    target_layer = model.layer4[-1]
    cam_extractor = GradCAM(model, target_layer)

    rows: List[dict] = []
    hard_neg_overlays: List[Tuple[Image.Image, str, float]] = []
    all_overlays: List[Tuple[Image.Image, str]] = []

    logging.info("Running Grad-CAM on %s crops from %s/%s ...", len(image_paths), data_root, args.split)

    for path in image_paths:
        tensor, pil_image = load_image(path, device)
        with torch.enable_grad():
            cam = cam_extractor.generate(tensor, args.target_class)
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
        })

        if is_hard_neg:
            overlay = overlay_heatmap(pil_image, cam)
            cap = "{} p={:.2f}".format(path.stem, prob_pos)
            hard_neg_overlays.append((overlay, cap, prob_pos))
            if args.save_all_overlays:
                all_overlays.append((overlay, cap))

    cam_extractor.remove_hooks()

    hard_neg_overlays.sort(key=lambda x: x[2], reverse=True)
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
    for overlay, cap, prob in hard_neg_overlays[:args.top_k]:
        name = cap.split(" ")[0]
        overlay.save(overlays_dir / "{}_gradcam.jpg".format(name))

    if hard_neg_overlays:
        montage = build_montage(
            [o for o, _, _ in hard_neg_overlays[:args.montage_max]],
            [c for _, c, _ in hard_neg_overlays[:args.montage_max]],
            output_dir / "hard_negatives_montage.jpg",
        )
        logging.info("Saved montage to %s", output_dir / "hard_negatives_montage.jpg")

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