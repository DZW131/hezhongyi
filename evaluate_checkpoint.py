import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluate import evaluate
from train import create_dataset
from unet import UNet
from utils.checkpoint_io import load_torch_state
from utils.segmentation_metrics import format_metrics
from utils.visualization import save_segmentation_preview


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained checkpoint on an image/mask dataset')
    parser.add_argument('--model', '-m', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--images-dir', type=str, required=True, help='Directory containing evaluation images')
    parser.add_argument('--masks-dir', type=str, required=True, help='Directory containing evaluation masks')
    parser.add_argument('--output-dir', type=str, default='evaluation',
                        help='Directory used to save evaluation results')
    parser.add_argument('--batch-size', '-b', type=int, default=1, help='Evaluation batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Scale factor for evaluation images')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision during evaluation')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Threshold for binary segmentation')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = create_dataset(Path(args.images_dir), Path(args.masks_dir), args.scale)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
    )

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    state_dict = load_torch_state(args.model, map_location=device)
    state_dict.pop('mask_values', None)
    model.load_state_dict(state_dict)
    model.to(device=device)

    metrics, preview = evaluate(model, dataloader, device=device, amp=args.amp, threshold=args.threshold)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / 'metrics.json'
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    if preview is not None:
        save_segmentation_preview(
            image_tensor=preview['image'],
            true_mask_tensor=preview['true_mask'],
            pred_mask_tensor=preview['pred_mask'],
            output_path=output_dir / 'preview.png',
            metrics=metrics,
        )

    logging.info('Evaluation complete: %s', format_metrics(metrics))
    logging.info('Saved metrics to %s', metrics_path)
