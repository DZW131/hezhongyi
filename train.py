import argparse
import csv
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from utils.segmentation_metrics import format_metrics
from utils.visualization import save_segmentation_preview, save_training_curves

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')
DEFAULT_CHECKPOINT_METRIC = 'dice'
AVAILABLE_CHECKPOINT_METRICS = ('dice', 'iou', 'precision', 'recall', 'specificity', 'accuracy')


class NullExperiment:
    def log(self, *_args, **_kwargs):
        return None

    def finish(self):
        return None


def compute_segmentation_loss(logits: torch.Tensor, true_masks: torch.Tensor, n_classes: int) -> torch.Tensor:
    if n_classes == 1:
        loss = nn.BCEWithLogitsLoss()(logits.squeeze(1), true_masks.float())
        loss += dice_loss(torch.sigmoid(logits.squeeze(1)), true_masks.float(), multiclass=False)
        return loss

    loss = nn.CrossEntropyLoss()(logits, true_masks)
    loss += dice_loss(
        F.softmax(logits, dim=1).float(),
        F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2).float(),
        multiclass=True
    )
    return loss


def create_dataset(images_dir: Path, masks_dir: Path, img_scale: float):
    try:
        return CarvanaDataset(images_dir, masks_dir, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        return BasicDataset(images_dir, masks_dir, img_scale)


def init_experiment(config: Dict[str, object], mode: str):
    if wandb is None:
        logging.warning('wandb is not installed. Continuing without W&B logging.')
        return NullExperiment()

    init_kwargs = dict(project='U-Net', resume='allow', anonymous='must', config=config, mode=mode)

    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        if mode == 'online':
            logging.warning('W&B online init failed (%s). Falling back to offline mode.', exc)
            try:
                return wandb.init(project='U-Net', resume='allow', anonymous='must', config=config, mode='offline')
            except Exception as offline_exc:
                logging.warning('W&B offline init failed (%s). Disabling W&B logging.', offline_exc)
        return wandb.init(project='U-Net', resume='allow', anonymous='must', config=config, mode='disabled')


def save_model_weights(model: torch.nn.Module, mask_values, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    state_dict['mask_values'] = mask_values
    torch.save(state_dict, str(output_path))


def write_history_csv(history_rows: List[Dict[str, float]], output_path: Path) -> None:
    if not history_rows:
        return

    fieldnames = []
    for row in history_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history_rows:
            writer.writerow(row)


def get_dataset_mask_values(dataset) -> List[int]:
    if isinstance(dataset, Subset):
        return dataset.dataset.mask_values
    return dataset.mask_values


def determine_split_sizes(dataset_size: int, val_percent: float) -> List[int]:
    if dataset_size < 2:
        raise ValueError('Dataset must contain at least 2 samples to create train/validation splits.')

    n_val = max(1, int(dataset_size * val_percent))
    n_val = min(n_val, dataset_size - 1)
    n_train = dataset_size - n_val
    return [n_train, n_val]


def train_model(
    model,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    save_every_epoch: bool = False,
    checkpoint_metric: str = DEFAULT_CHECKPOINT_METRIC,
    img_scale: float = 0.5,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
    num_workers: int = 0,
    checkpoint_dir: Path = dir_checkpoint,
    wandb_mode: str = 'online',
    train_images_dir: Path = dir_img,
    train_masks_dir: Path = dir_mask,
    val_images_dir: Path = None,
    val_masks_dir: Path = None,
):
    if (val_images_dir is None) != (val_masks_dir is None):
        raise ValueError('Validation image and mask directories must be provided together.')

    if val_images_dir is not None and val_masks_dir is not None:
        train_set = create_dataset(train_images_dir, train_masks_dir, img_scale)
        val_set = create_dataset(val_images_dir, val_masks_dir, img_scale)
        n_train = len(train_set)
        n_val = len(val_set)
    else:
        dataset = create_dataset(train_images_dir, train_masks_dir, img_scale)
        n_train, n_val = determine_split_sizes(len(dataset), val_percent)
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = checkpoint_dir / 'analysis'
    preview_dir = analysis_dir / 'val_previews'
    history_path = analysis_dir / 'history.csv'
    curves_path = analysis_dir / 'training_curves.png'
    best_metrics_path = analysis_dir / 'best_metrics.json'
    best_preview_path = analysis_dir / 'best_preview.png'

    experiment = init_experiment(
        config=dict(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_percent=val_percent,
            train_images_dir=str(train_images_dir),
            train_masks_dir=str(train_masks_dir),
            val_images_dir=str(val_images_dir) if val_images_dir else '',
            val_masks_dir=str(val_masks_dir) if val_masks_dir else '',
            save_checkpoint=save_checkpoint,
            save_every_epoch=save_every_epoch,
            checkpoint_metric=checkpoint_metric,
            img_scale=img_scale,
            amp=amp,
            num_workers=num_workers,
        ),
        mode=wandb_mode,
    )

    logging.info(
        'Starting training:\n'
        '    Epochs:          %s\n'
        '    Batch size:      %s\n'
        '    Learning rate:   %s\n'
        '    Training size:   %s\n'
        '    Validation size: %s\n'
        '    Checkpoints:     %s\n'
        '    Checkpoint rule: best %s\n'
        '    Device:          %s\n'
        '    Images scaling:  %s\n'
        '    Mixed Precision: %s',
        epochs,
        batch_size,
        learning_rate,
        n_train,
        n_val,
        save_checkpoint,
        checkpoint_metric,
        device.type,
        img_scale,
        amp,
    )

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == 'cuda')

    best_metric_value = float('-inf')
    best_epoch = 0
    history_rows = []
    mask_values = get_dataset_mask_values(train_set)

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            seen_images = 0

            with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch, epochs), unit='img') as pbar:
                for batch in train_loader:
                    images, true_masks = batch['image'], batch['mask']

                    assert images.shape[1] == model.n_channels, (
                        'Network has been defined with {} input channels, but loaded images have {} channels. '
                        'Please check that the images are loaded correctly.'
                    ).format(model.n_channels, images.shape[1])

                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                        masks_pred = model(images)
                        loss = compute_segmentation_loss(masks_pred, true_masks, model.n_classes)

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    batch_size_current = images.shape[0]
                    epoch_loss += loss.item() * batch_size_current
                    seen_images += batch_size_current
                    pbar.update(batch_size_current)
                    pbar.set_postfix(loss='{:.4f}'.format(loss.item()))

            train_loss = epoch_loss / max(seen_images, 1)
            val_metrics, preview = evaluate(model, val_loader, device, amp)
            scheduler.step(val_metrics[checkpoint_metric])

            learning_rate_current = optimizer.param_groups[0]['lr']
            preview_path = None
            if preview is not None:
                preview_path = preview_dir / 'epoch_{:03d}.png'.format(epoch)
                save_segmentation_preview(
                    image_tensor=preview['image'],
                    true_mask_tensor=preview['true_mask'],
                    pred_mask_tensor=preview['pred_mask'],
                    output_path=preview_path,
                    metrics=val_metrics,
                )

            history_row = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'learning_rate': learning_rate_current,
                'checkpoint_metric': val_metrics[checkpoint_metric],
            }

            for key, value in val_metrics.items():
                history_row[key] = float(value)

            is_best = val_metrics[checkpoint_metric] > best_metric_value
            history_row['is_best'] = int(is_best)
            history_rows.append(history_row)
            write_history_csv(history_rows, history_path)
            save_training_curves(history_rows, curves_path)

            if save_checkpoint:
                save_model_weights(model, mask_values, checkpoint_dir / 'latest.pth')
                if save_every_epoch:
                    save_model_weights(model, mask_values, checkpoint_dir / 'epoch_{:03d}.pth'.format(epoch))

                if is_best:
                    best_metric_value = val_metrics[checkpoint_metric]
                    best_epoch = epoch
                    save_model_weights(model, mask_values, checkpoint_dir / 'best.pth')

                    if preview_path is not None:
                        shutil.copyfile(str(preview_path), str(best_preview_path))

                    with best_metrics_path.open('w', encoding='utf-8') as metrics_file:
                        json.dump(
                            {
                                'best_epoch': best_epoch,
                                'selection_metric': checkpoint_metric,
                                'metrics': history_row,
                            },
                            metrics_file,
                            indent=2,
                        )

            logging.info(
                'Epoch %s finished. train_loss=%.4f, %s%s',
                epoch,
                train_loss,
                format_metrics(val_metrics),
                ' [best]' if is_best else '',
            )

            log_payload = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/learning_rate': learning_rate_current,
            }
            for key, value in val_metrics.items():
                log_payload['val/{}'.format(key)] = value
            log_payload['val/is_best'] = int(is_best)

            if preview_path is not None:
                try:
                    if wandb is not None:
                        log_payload['val/preview'] = wandb.Image(str(preview_path))
                except Exception:
                    pass

            experiment.log(log_payload)

    finally:
        experiment.finish()

    if save_checkpoint and best_epoch > 0:
        logging.info(
            'Training complete. Best %s=%.4f at epoch %s. Best weights saved to %s',
            checkpoint_metric,
            best_metric_value,
            best_epoch,
            checkpoint_dir / 'best.pth',
        )


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--images-dir', type=str, default=str(dir_img),
                        help='Directory containing training images')
    parser.add_argument('--masks-dir', type=str, default=str(dir_mask),
                        help='Directory containing training masks')
    parser.add_argument('--val-images-dir', type=str, default='',
                        help='Optional directory containing validation images')
    parser.add_argument('--val-masks-dir', type=str, default='',
                        help='Optional directory containing validation masks')
    parser.add_argument('--num-workers', type=int, default=min(8, os.cpu_count() or 1),
                        help='Number of dataloader workers')
    parser.add_argument('--save-every-epoch', action='store_true', default=False,
                        help='Also save a dedicated checkpoint file for every epoch')
    parser.add_argument('--checkpoint-metric', choices=AVAILABLE_CHECKPOINT_METRICS,
                        default=DEFAULT_CHECKPOINT_METRIC,
                        help='Metric used to decide which checkpoint is the best one')
    parser.add_argument('--checkpoint-dir', type=str, default=str(dir_checkpoint),
                        help='Directory used to store checkpoints and analysis artifacts')
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online',
                        help='Weights & Biases logging mode')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(
        'Network:\n'
        '\t%s input channels\n'
        '\t%s output channels (classes)\n'
        '\t%s upscaling',
        model.n_channels,
        model.n_classes,
        'Bilinear' if model.bilinear else 'Transposed conv',
    )

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        state_dict.pop('mask_values', None)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            num_workers=args.num_workers,
            save_every_epoch=args.save_every_epoch,
            checkpoint_metric=args.checkpoint_metric,
            checkpoint_dir=Path(args.checkpoint_dir),
            wandb_mode=args.wandb_mode,
            train_images_dir=Path(args.images_dir),
            train_masks_dir=Path(args.masks_dir),
            val_images_dir=Path(args.val_images_dir) if args.val_images_dir else None,
            val_masks_dir=Path(args.val_masks_dir) if args.val_masks_dir else None,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error(
            'Detected OutOfMemoryError! Enabling checkpointing to reduce memory usage, but this slows down training. '
            'Consider enabling AMP (--amp) for faster and more memory efficient training.'
        )
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            num_workers=args.num_workers,
            save_every_epoch=args.save_every_epoch,
            checkpoint_metric=args.checkpoint_metric,
            checkpoint_dir=Path(args.checkpoint_dir),
            wandb_mode=args.wandb_mode,
            train_images_dir=Path(args.images_dir),
            train_masks_dir=Path(args.masks_dir),
            val_images_dir=Path(args.val_images_dir) if args.val_images_dir else None,
            val_masks_dir=Path(args.val_masks_dir) if args.val_masks_dir else None,
        )
