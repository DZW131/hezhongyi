"""Visualize MESCnn M/E/S/C classifiers with Grad-CAM on HSPN crops.

For each crop in the dataset, computes Grad-CAM heatmaps from all four MESCnn
classifiers (M/E/S/C, IgA-trained) and saves a 4-row comparison montage:
  row M: original | heatmap | overlay  (target = yesM, class 2)
  row E: original | heatmap | overlay  (target = yesE logit)
  row S: original | heatmap | overlay  (target = SGS, class 2)
  row C: original | heatmap | overlay  (target = yesC logit)

This shows WHERE each IgA-trained classifier looks on HSPN crops, helping
diagnose whether cross-center domain shift affects localization too.
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
except ImportError as exc:
    raise SystemExit("torchvision is required: {}".format(exc))


IMAGE_SIZE = 224
NORMALIZE = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grad-CAM visualization for MESCnn M/E/S/C on HSPN crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models-dir", required=True,
                        help="Directory containing efficientnetv2-m_M_V3.pth etc.")
    parser.add_argument("--data-root", required=True,
                        help="Crop dataset root with <split>/images and <split>/masks")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-crops", type=int, default=24,
                        help="Max crops to visualize (sorted by M yesM-prob descending)")
    parser.add_argument("--cam-method", default="gradcam",
                        choices=("gradcam", "gradcampp", "xgradcam", "layercam"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_classifier(path: Path, device: torch.device):
    return torch.load(str(path), map_location=device, weights_only=False)


def find_last_conv_block(model: torch.nn.Module, arch: str) -> torch.nn.Module:
    """Find the last conv block for Grad-CAM target layer."""
    if arch == "efficientnet":
        # EfficientNet: features.8 is the last conv (1x1 conv)
        return model.features[8][1] if hasattr(model, "features") else list(model.modules())[-5]
    if arch == "densenet":
        # DenseNet161 wrapped in Sequential: [0]=densenet, ..., [4]=Linear
        # use norm5 (last BN before classifier)
        if hasattr(model, "0") and hasattr(model[0], "norm5"):
            return model[0].norm5
        return list(model.modules())[-10]
    if arch == "mobilenet":
        # MobileNetV2: features[-1][1] is last BN
        return model.features[-1][1] if hasattr(model, "features") else list(model.modules())[-5]
    return list(model.modules())[-5]


class GradCAM:
    """Grad-CAM using forward hook for activations + tensor hook for gradients.

    Uses register_forward_hook + output.register_hook instead of
    register_full_backward_hook, to avoid the inplace-SiLU view conflict
    that breaks backward hooks on EfficientNet.
    """

    def __init__(self, model, target_layer, method="gradcam"):
        self.model = model
        self.target_layer = target_layer
        self.method = method
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, _m, _input, output):
        # clone so we can register a hook even if output doesn't require grad by default
        self.activations = output
        self.gradients = None
        # register hook on the output tensor; works when called under enable_grad
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def remove_hooks(self):
        self._fwd.remove()

    def _compute_weights(self):
        g = self.gradients
        a = self.activations
        if self.method == "gradcam":
            return g.mean(dim=(2, 3), keepdim=True)
        if self.method == "gradcampp":
            g2 = g.pow(2); g3 = g2 * g
            denom = 2.0 * g2 + (a * g3).sum(dim=(2, 3), keepdim=True)
            denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
            alpha = g2 / denom
            alpha = F.relu(g) * alpha
            return alpha.sum(dim=(2, 3), keepdim=True) / max(1, alpha.shape[1])
        if self.method == "xgradcam":
            w = (g * a).sum(dim=(2, 3), keepdim=True)
            d = g.sum(dim=(2, 3), keepdim=True).abs() + 1e-8
            return w / d
        if self.method == "layercam":
            return F.relu(g)
        return g.mean(dim=(2, 3), keepdim=True)

    def generate(self, input_tensor, target_class: int):
        """Run forward+backward and return CAM. target_class: index into logits."""
        self.model.zero_grad()
        with torch.enable_grad():
            # ensure input has grad to propagate through the network
            inp = input_tensor.detach().clone().requires_grad_(False)
            # we rely on parameter grads, not input grad; enable_grad lets hooks fire
            logits = self.model(inp)
            target_logit = logits[0, target_class]
            target_logit.backward()
        if self.activations is None or self.gradients is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        w = self._compute_weights()
        cam = (w * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cmin, cmax = cam.min(), cam.max()
        if cmax - cmin > 1e-8:
            cam = (cam - cmin) / (cmax - cmin)
        else:
            cam = np.zeros_like(cam)
        return cam


def load_image(path: Path, device: torch.device):
    """Load with MESCnn transform (only ToTensor, no normalize)."""
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device=device, dtype=torch.float32)
    return tensor, image


def _apply_jet(cam_flat):
    v = np.clip(cam_flat, 0.0, 1.0)
    r = np.zeros_like(v); g = np.zeros_like(v); b = np.zeros_like(v)
    m1 = v < 0.125; r[m1] = 0; g[m1] = 0; b[m1] = 0.5 + v[m1] * 4.0
    m2 = (v >= 0.125) & (v < 0.375); r[m2] = 0; g[m2] = (v[m2] - 0.125) * 4.0; b[m2] = 1.0
    m3 = (v >= 0.375) & (v < 0.625); r[m3] = (v[m3] - 0.375) * 4.0; g[m3] = 1.0; b[m3] = 1.0 - (v[m3] - 0.375) * 4.0
    m4 = (v >= 0.625) & (v < 0.875); r[m4] = 1.0; g[m4] = 1.0 - (v[m4] - 0.625) * 4.0; b[m4] = 0.0
    m5 = v >= 0.875; r[m5] = 1.0 - (v[m5] - 0.875) * 4.0; g[m5] = 0.0; b[m5] = 0.0
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_heatmap(image, cam, alpha=0.5):
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    ).astype(np.float32) / 255.0
    jet = _apply_jet(cam_resized.reshape(-1)).reshape(*cam_resized.shape, 3)
    base = np.array(image, dtype=np.float32)
    local_alpha = (cam_resized[..., None] * alpha).clip(0, alpha)
    return Image.fromarray((base * (1 - local_alpha) + jet * local_alpha).astype(np.uint8))


def heatmap_only_image(cam, size):
    jet = _apply_jet(cam.reshape(-1)).reshape(*cam.shape, 3)
    return Image.fromarray(jet).resize(size, Image.BILINEAR)


def build_montage(records, output_path, cell_size=224, cols=1):
    """records: list of (crop_name, rows_data) where rows_data is dict of lesion->(orig, heat, overlay, prob_str).
    Each crop produces 4 rows (M/E/S/C), each row = 3 cells (orig | heat | overlay)."""
    if not records:
        return
    lesions = ["M", "E", "S", "C"]
    lesion_names = {"M": "Mesangial", "E": "Endocapillary", "S": "SegmentalGS", "C": "Crescent"}
    rows_per_crop = len(lesions) * 3  # 4 lesions * (orig+heat+overlay) but we stack vertically
    label_h = 24
    # layout: per crop, 4 lesion-rows, each lesion-row has 3 columns (orig/heat/overlay)
    total_rows = len(records) * len(lesions)
    canvas_w = 3 * cell_size + 120  # extra for label column
    canvas_h = total_rows * (cell_size + label_h)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y = 0
    for crop_name, rows_data in records:
        for li, lesion in enumerate(lesions):
            if lesion not in rows_data:
                continue
            orig, heat, overlay, prob_str = rows_data[lesion]
            # label column
            draw.rectangle([0, y, 120, y + cell_size], fill=(240, 240, 240))
            short_name = crop_name[:18] if li == 0 else ""
            draw.text((4, y + 4), short_name, fill=(0, 0, 0))
            draw.text((4, y + cell_size // 2 - 6), lesion_names[lesion], fill=(60, 60, 60))
            # 3 cells
            for ci, img in enumerate([orig, heat, overlay]):
                x = 120 + ci * cell_size
                canvas.paste(img.resize((cell_size, cell_size)), (x, y))
            # prob label below overlay
            draw.rectangle([120 + 2 * cell_size, y + cell_size, 120 + 3 * cell_size, y + cell_size + label_h],
                           fill=(30, 30, 30))
            draw.text((120 + 2 * cell_size + 4, y + cell_size + 6), prob_str, fill=(255, 220, 0))
            y += cell_size + label_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    models_dir = Path(args.models_dir)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = data_root / args.split / "images"
    masks_dir = data_root / args.split / "masks"
    image_paths = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    # classifier configs: (lesion, filename, arch, target_class_index)
    # M: 3-class, target=2 (yesM); E: 2-class, target=1 (yesE); S: 3-class, target=2 (SGS); C: 2-class, target=1 (yesC)
    configs = [
        ("M", "efficientnetv2-m_M_V3.pth", "efficientnet", 2),
        ("E", "efficientnetv2-m_E_V3.pth", "efficientnet", 1),
        ("S", "densenet161_S_V3.pth", "densenet", 2),
        ("C", "mobilenetv2_C_V3.pth", "mobilenet", 1),
    ]

    # load all classifiers and set up GradCAM
    loaded = {}
    for lesion, fname, arch, tgt_cls in configs:
        path = models_dir / fname
        if not path.exists():
            logging.warning("%s weight missing, skip", lesion)
            continue
        logging.info("Loading %s classifier (%s)", lesion, fname)
        model = load_classifier(path, device)
        model.eval().to(device)
        target_layer = find_last_conv_block(model, arch)
        cam = GradCAM(model, target_layer, method=args.cam_method)
        loaded[lesion] = (model, cam, arch, tgt_cls)

    # first pass: get M yesM-prob for sorting
    m_probs = []
    if "M" in loaded:
        model, cam, _, tgt = loaded["M"]
        with torch.no_grad():
            for p in image_paths:
                tensor, _ = load_image(p, device)
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                # mask label
                mp = masks_dir / (p.stem + ".png")
                lab = -1
                if mp.exists():
                    m = np.array(Image.open(mp).convert("L"))
                    lab = int(m.max() > 0)
                m_probs.append((p, float(probs[tgt]), lab))
        m_probs.sort(key=lambda x: -x[1])
    else:
        m_probs = [(p, 0.0, -1) for p in image_paths]

    selected = m_probs[:args.max_crops]
    logging.info("Visualizing %d crops (sorted by M yesM-prob)", len(selected))

    records = []
    for path, m_prob, true_label in selected:
        tensor, pil_image = load_image(path, device)
        gt_str = "pos" if true_label == 1 else ("neg" if true_label == 0 else "?")
        rows_data = {}
        for lesion, _fname, _arch, tgt_cls in configs:
            if lesion not in loaded:
                continue
            model, cam, _arch, _ = loaded[lesion]
            # recompute forward to get fresh logits (needed for backward)
            tensor.grad = None
            cam_map = cam.generate(tensor, tgt_cls)
            # separate forward (no_grad) just to get probs for display
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            prob_str = "{} p={:.2f}".format(lesion, float(probs[tgt_cls]))
            heat = heatmap_only_image(cam_map, pil_image.size)
            overlay = overlay_heatmap(pil_image, cam_map)
            rows_data[lesion] = (pil_image, heat, overlay, prob_str)
        # add GT info to first lesion's prob_str
        if "M" in rows_data:
            o, h, ov, ps = rows_data["M"]
            rows_data["M"] = (o, h, ov, "{} | GT={}".format(ps, gt_str))
        records.append((path.name, rows_data))

    # cleanup
    for lesion, (model, cam, _, _) in loaded.items():
        cam.remove_hooks()

    montage_path = output_dir / "mescnn_gradcam_montage.jpg"
    build_montage(records, montage_path)
    logging.info("Saved montage to %s", montage_path)

    # also save per-crop summary csv
    with (output_dir / "per_crop_summary.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.writer(h)
        writer.writerow(["name", "true_label", "M_yesM_prob", "E_yesE_prob", "S_SGS_prob", "C_yesC_prob"])
        for path, m_p, tl in selected:
            row = [path.name, tl]
            for lesion in ["M", "E", "S", "C"]:
                if lesion in loaded:
                    model = loaded[lesion][0]
                    with torch.no_grad():
                        t, _ = load_image(path, device)
                        lg = model(t)
                        pr = F.softmax(lg, dim=1)[0].cpu().numpy()
                        tgt = loaded[lesion][3]
                        row.append(round(float(pr[tgt]), 4))
                else:
                    row.append("")
            writer.writerow(row)

    print(json.dumps({"montage": str(montage_path), "num_crops": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()