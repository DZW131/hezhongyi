import argparse
import csv
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
AVAILABLE_OPTIMIZERS = ('rmsprop', 'adamw')
AVAILABLE_COMPILE_MODES = ('auto', 'off', 'default', 'reduce-overhead', 'max-autotune')
AVAILABLE_MATMUL_PRECISIONS = ('highest', 'high', 'medium')


class NullExperiment:
    def log(self, *_args, **_kwargs):
        return None

    def finish(self):
        return None


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


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
    model_to_save = unwrap_model(model)
    state_dict = model_to_save.state_dict()
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


def configure_runtime(device: torch.device, enable_tf32: bool, cudnn_benchmark: bool, matmul_precision: str) -> None:
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision(matmul_precision)

    if device.type == 'cuda':
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = enable_tf32

        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.allow_tf32 = enable_tf32
            torch.backends.cudnn.benchmark = cudnn_benchmark

        logging.info(
            'CUDA performance options: tf32=%s, cudnn_benchmark=%s, matmul_precision=%s',
            enable_tf32,
            cudnn_benchmark,
            matmul_precision,
        )


def maybe_compile_model(model: torch.nn.Module, device: torch.device, compile_mode: str) -> Tuple[torch.nn.Module, bool]:
    if compile_mode == 'off':
        return model, False

    if not hasattr(torch, 'compile'):
        logging.warning('torch.compile is unavailable in this PyTorch build. Continuing without compilation.')
        return model, False

    if compile_mode == 'auto':
        if device.type != 'cuda':
            return model, False
        selected_mode = 'reduce-overhead'
    else:
        selected_mode = compile_mode

    try:
        compiled_model = torch.compile(model, mode=selected_mode)
        logging.info('Enabled torch.compile with mode=%s', selected_mode)
        return compiled_model, True
    except Exception as exc:
        logging.warning('torch.compile failed (%s). Continuing without compilation.', exc)
        return model, False


def create_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    device: torch.device,
    use_fused_optimizer: bool,
):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'adamw':
        optimizer_kwargs = dict(lr=learning_rate, weight_decay=weight_decay)
        if use_fused_optimizer and device.type == 'cuda':
            try:
                optimizer = optim.AdamW(model.parameters(), fused=True, **optimizer_kwargs)
                logging.info('Using fused AdamW optimizer')
                return optimizer
            except TypeError:
                logging.warning('Fused AdamW is unavailable in this PyTorch version. Falling back to standard AdamW.')

        return optim.AdamW(model.parameters(), **optimizer_kwargs)

    return optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=device.type == 'cuda',
    )


def build_dataloader_args(
    batch_size: int,
    num_workers: int,
    device: torch.device,
    persistent_workers: bool,
    prefetch_factor: int,
) -> Dict[str, object]:
    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
    )

    if num_workers > 0:
        loader_args['persistent_workers'] = persistent_workers
        if prefetch_factor > 0:
            loader_args['prefetch_factor'] = prefetch_factor

    return loader_args


def should_run_epoch_task(epoch: int, total_epochs: int, frequency: int) -> bool:
    if frequency <= 0:
        return epoch == total_epochs
    return epoch == total_epochs or epoch % frequency == 0


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
    val_images_dir: Optional[Path] = None,
    val_masks_dir: Optional[Path] = None,
    optimizer_name: str = 'rmsprop',
    use_fused_optimizer: bool = True,
    compile_mode: str = 'auto',
    enable_tf32: bool = True,
    cudnn_benchmark: bool = True,
    matmul_precision: str = 'high',
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    val_frequency: int = 1,
    analysis_frequency: int = 5,
    preview_frequency: int = 5,
):
    if (val_images_dir is None) != (val_masks_dir is None):
        raise ValueError('Validation image and mask directories must be provided together.')

    configure_runtime(device, enable_tf32=enable_tf32, cudnn_benchmark=cudnn_benchmark, matmul_precision=matmul_precision)

    if val_images_dir is not None and val_masks_dir is not None:
        train_set = create_dataset(train_images_dir, train_masks_dir, img_scale)
        val_set = create_dataset(val_images_dir, val_masks_dir, img_scale)
        n_train = len(train_set)
        n_val = len(val_set)
    else:
        dataset = create_dataset(train_images_dir, train_masks_dir, img_scale)
        n_train, n_val = determine_split_sizes(len(dataset), val_percent)
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = build_dataloader_args(
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    optimizer = create_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        device=device,
        use_fused_optimizer=use_fused_optimizer,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == 'cuda')
    train_model_for_forward, is_compiled = maybe_compile_model(model, device=device, compile_mode=compile_mode)

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
            optimizer=optimizer_name,
            fused_optimizer=use_fused_optimizer,
            compile_mode=compile_mode,
            compiled=is_compiled,
            enable_tf32=enable_tf32,
            cudnn_benchmark=cudnn_benchmark,
            matmul_precision=matmul_precision,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            val_frequency=val_frequency,
            analysis_frequency=analysis_frequency,
            preview_frequency=preview_frequency,
        ),
        mode=wandb_mode,
    )

    logging.info(
        'Starting training:\n'
        '    Epochs:              %s\n'
        '    Batch size:          %s\n'
        '    Learning rate:       %s\n'
        '    Training size:       %s\n'
        '    Validation size:     %s\n'
        '    Checkpoints:         %s\n'
        '    Checkpoint rule:     best %s\n'
        '    Device:              %s\n'
        '    Images scaling:      %s\n'
        '    Mixed Precision:     %s\n'
        '    Optimizer:           %s\n'
        '    torch.compile:       %s\n'
        '    num_workers:         %s\n'
        '    persistent_workers:  %s\n'
        '    prefetch_factor:     %s\n'
        '    val_frequency:       %s',
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
        optimizer_name,
        is_compiled,
        num_workers,
        persistent_workers and num_workers > 0,
        prefetch_factor if num_workers > 0 else 0,
        val_frequency,
    )

    best_metric_value = float('-inf')
    best_epoch = 0
    history_rows = []
    mask_values = get_dataset_mask_values(train_set)
    non_blocking = device.type == 'cuda'

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            train_model_for_forward.train()
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            seen_images = 0

            with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch, epochs), unit='img') as pbar:
                for batch in train_loader:
                    images, true_masks = batch['image'], batch['mask']

                    assert images.shape[1] == model.n_channels, (
                        'Network has been defined with {} input channels, but loaded images have {} channels. '
                        'Please check that the images are loaded correctly.'
                    ).format(model.n_channels, images.shape[1])

                    images = images.to(
                        device=device,
                        dtype=torch.float32,
                        memory_format=torch.channels_last,
                        non_blocking=non_blocking,
                    )
                    true_masks = true_masks.to(device=device, dtype=torch.long, non_blocking=non_blocking)

                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                        masks_pred = train_model_for_forward(images)
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

            epoch_seconds = time.perf_counter() - epoch_start
            train_loss = epoch_loss / max(seen_images, 1)
            train_images_per_second = seen_images / max(epoch_seconds, 1e-6)
            learning_rate_current = optimizer.param_groups[0]['lr']

            run_validation = should_run_epoch_task(epoch, epochs, val_frequency)
            val_metrics = {}
            preview = None
            validation_seconds = 0.0

            if run_validation:
                validation_start = time.perf_counter()
                val_metrics, preview = evaluate(train_model_for_forward, val_loader, device, amp)
                validation_seconds = time.perf_counter() - validation_start
                scheduler.step(val_metrics[checkpoint_metric])

            history_row = {
                'epoch': epoch,
                'train_loss': train_loss,
                'learning_rate': learning_rate_current,
                'epoch_seconds': epoch_seconds,
                'train_images_per_second': train_images_per_second,
                'validation_seconds': validation_seconds,
                'validated': int(run_validation),
            }

            if run_validation:
                history_row['val_loss'] = float(val_metrics['loss'])
                history_row['checkpoint_metric'] = float(val_metrics[checkpoint_metric])
                for key, value in val_metrics.items():
                    history_row[key] = float(value)
            else:
                history_row['val_loss'] = float('nan')
                history_row['checkpoint_metric'] = float('nan')
                for key in AVAILABLE_CHECKPOINT_METRICS:
                    history_row[key] = float('nan')

            is_best = run_validation and val_metrics[checkpoint_metric] > best_metric_value
            history_row['is_best'] = int(is_best)
            history_rows.append(history_row)
            write_history_csv(history_rows, history_path)

            should_update_analysis = should_run_epoch_task(epoch, epochs, analysis_frequency)
            if should_update_analysis:
                save_training_curves(history_rows, curves_path)

            should_save_preview = (
                run_validation and preview is not None and (
                    is_best or should_run_epoch_task(epoch, epochs, preview_frequency)
                )
            )
            preview_path = None
            if should_save_preview:
                preview_path = preview_dir / 'epoch_{:03d}.png'.format(epoch)
                save_segmentation_preview(
                    image_tensor=preview['image'],
                    true_mask_tensor=preview['true_mask'],
                    pred_mask_tensor=preview['pred_mask'],
                    output_path=preview_path,
                    metrics=val_metrics,
                )

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
                    elif preview is not None:
                        save_segmentation_preview(
                            image_tensor=preview['image'],
                            true_mask_tensor=preview['true_mask'],
                            pred_mask_tensor=preview['pred_mask'],
                            output_path=best_preview_path,
                            metrics=val_metrics,
                        )

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

            if run_validation:
                logging.info(
                    'Epoch %s finished in %.1fs (train %.1f img/s, val %.1fs). train_loss=%.4f, %s%s',
                    epoch,
                    epoch_seconds,
                    train_images_per_second,
                    validation_seconds,
                    train_loss,
                    format_metrics(val_metrics),
                    ' [best]' if is_best else '',
                )
            else:
                logging.info(
                    'Epoch %s finished in %.1fs (train %.1f img/s). train_loss=%.4f [validation skipped]',
                    epoch,
                    epoch_seconds,
                    train_images_per_second,
                    train_loss,
                )

            log_payload = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/learning_rate': learning_rate_current,
                'train/epoch_seconds': epoch_seconds,
                'train/images_per_second': train_images_per_second,
                'val/validated': int(run_validation),
            }

            if run_validation:
                log_payload['val/seconds'] = validation_seconds
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

    save_training_curves(history_rows, curves_path)

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
    parser.add_argument('--num-workers', type=int, default=min(16, os.cpu_count() or 1),
                        help='Number of dataloader workers')
    parser.add_argument('--prefetch-factor', type=int, default=4,
                        help='Number of batches preloaded by each dataloader worker')
    parser.add_argument('--disable-persistent-workers', action='store_true', default=False,
                        help='Disable dataloader persistent workers')
    parser.add_argument('--save-every-epoch', action='store_true', default=False,
                        help='Also save a dedicated checkpoint file for every epoch')
    parser.add_argument('--checkpoint-metric', choices=AVAILABLE_CHECKPOINT_METRICS,
                        default=DEFAULT_CHECKPOINT_METRIC,
                        help='Metric used to decide which checkpoint is the best one')
    parser.add_argument('--checkpoint-dir', type=str, default=str(dir_checkpoint),
                        help='Directory used to store checkpoints and analysis artifacts')
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online',
                        help='Weights & Biases logging mode')
    parser.add_argument('--optimizer', choices=AVAILABLE_OPTIMIZERS, default='rmsprop',
                        help='Optimizer used for training')
    parser.add_argument('--disable-fused-optimizer', action='store_true', default=False,
                        help='Disable fused optimizer kernels when supported')
    parser.add_argument('--compile', dest='compile_mode', choices=AVAILABLE_COMPILE_MODES, default='auto',
                        help='Enable torch.compile for faster training on supported setups')
    parser.add_argument('--disable-tf32', action='store_true', default=False,
                        help='Disable TF32 matmul / cuDNN acceleration on Ampere+ GPUs')
    parser.add_argument('--disable-cudnn-benchmark', action='store_true', default=False,
                        help='Disable cuDNN benchmark autotuning')
    parser.add_argument('--matmul-precision', choices=AVAILABLE_MATMUL_PRECISIONS, default='high',
                        help='torch.set_float32_matmul_precision setting')
    parser.add_argument('--val-frequency', type=int, default=1,
                        help='Run validation every N epochs (final epoch always validates)')
    parser.add_argument('--analysis-frequency', type=int, default=5,
                        help='Refresh training curves every N epochs (final epoch always refreshes)')
    parser.add_argument('--preview-frequency', type=int, default=5,
                        help='Save validation preview images every N epochs (best and final still save)')

    return parser.parse_args()


def run_training(args, model, device):
    return train_model(
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
        optimizer_name=args.optimizer,
        use_fused_optimizer=not args.disable_fused_optimizer,
        compile_mode=args.compile_mode,
        enable_tf32=not args.disable_tf32,
        cudnn_benchmark=not args.disable_cudnn_benchmark,
        matmul_precision=args.matmul_precision,
        persistent_workers=not args.disable_persistent_workers,
        prefetch_factor=args.prefetch_factor,
        val_frequency=args.val_frequency,
        analysis_frequency=args.analysis_frequency,
        preview_frequency=args.preview_frequency,
    )


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
        run_training(args, model, device)
    except torch.cuda.OutOfMemoryError:
        logging.error(
            'Detected OutOfMemoryError! Enabling checkpointing to reduce memory usage, but this slows down training. '
            'Consider lowering --batch-size if you want to preserve maximum throughput.'
        )
        torch.cuda.empty_cache()
        model.use_checkpointing()
        run_training(args, model, device)
