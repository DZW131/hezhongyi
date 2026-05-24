import argparse
import json
import subprocess
import sys
from pathlib import Path

from hzy_lesion_config import resolve_lesion_tasks


DEFAULT_TILES_ROOT = "/root/datasets/HZY_HSPN_lesion_tasks"
DEFAULT_CHECKPOINT_ROOT = "/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_lesion_tasks"
DEFAULT_BASE_CHECKPOINT = "/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_finetune_glom_scene_ds025_resume/best.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one multi-class U-Net model per grouped HZY lesion task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tiles-root", default=DEFAULT_TILES_ROOT, help="Root directory produced by prepare_hzy_lesion_tiles.py")
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT, help="Root directory for lesion task checkpoints")
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT,
                        help="Checkpoint used to initialize matching model tensors")
    parser.add_argument("--tasks", nargs="*", default=[],
                        help="Optional task slugs/names to train: proliferation, crescent, other_lesions")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per lesion task model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--scale", type=float, default=1.0, help="Training image scale")
    parser.add_argument("--optimizer", choices=("rmsprop", "adamw"), default="adamw", help="Optimizer")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled",
                        help="Weights & Biases mode")
    parser.add_argument("--amp", action="store_true", default=False, help="Use AMP")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="Skip a task if checkpoint best.pth already exists")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Print commands without running them")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def dataset_dirs(tiles_root: Path, task_slug: str):
    dataset_root = tiles_root / task_slug
    return {
        "train_images": dataset_root / "train" / "images",
        "train_masks": dataset_root / "train" / "masks",
        "val_images": dataset_root / "val" / "images",
        "val_masks": dataset_root / "val" / "masks",
    }


def dataset_summary(tiles_root: Path, task_slug: str):
    summary_path = tiles_root / task_slug / "manifests" / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_stems(directory: Path, suffix: str):
    return {path.stem for path in directory.glob("*{}".format(suffix)) if not path.name.startswith(".")}


def validate_split(task_slug: str, split: str, image_dir: Path, mask_dir: Path):
    image_ids = file_stems(image_dir, ".jpg")
    mask_ids = file_stems(mask_dir, ".png")

    if not image_ids:
        raise RuntimeError("Task '{}' has no {} images in {}".format(task_slug, split, image_dir))
    if not mask_ids:
        raise RuntimeError("Task '{}' has no {} masks in {}".format(task_slug, split, mask_dir))

    missing_masks = sorted(image_ids - mask_ids)
    missing_images = sorted(mask_ids - image_ids)
    if missing_masks or missing_images:
        raise RuntimeError(
            "Task '{}' {} split has unmatched image/mask ids. Missing masks: {}. Missing images: {}.".format(
                task_slug,
                split,
                ", ".join(missing_masks[:5]) or "none",
                ", ".join(missing_images[:5]) or "none",
            )
        )


def validate_dataset(tiles_root: Path, task):
    dirs = dataset_dirs(tiles_root, task.slug)
    missing = [str(path) for path in dirs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing tile directories for task '{}': {}. Run prepare_hzy_lesion_tiles.py first.".format(
                task.slug,
                ", ".join(missing),
            )
        )
    validate_split(task.slug, "train", dirs["train_images"], dirs["train_masks"])
    validate_split(task.slug, "val", dirs["val_images"], dirs["val_masks"])

    summary = dataset_summary(tiles_root, task.slug)
    summary_num_classes = summary.get("num_classes") if summary else None
    if summary_num_classes is not None and int(summary_num_classes) != task.num_classes:
        raise ValueError(
            "Task '{}' dataset summary says num_classes={}, but config expects {}.".format(
                task.slug,
                summary_num_classes,
                task.num_classes,
            )
        )


def build_train_command(args, task) -> list:
    dirs = dataset_dirs(Path(args.tiles_root), task.slug)
    checkpoint_dir = Path(args.checkpoint_root) / task.slug
    command = [
        sys.executable,
        str(repo_root() / "train.py"),
        "--images-dir",
        str(dirs["train_images"]),
        "--masks-dir",
        str(dirs["train_masks"]),
        "--val-images-dir",
        str(dirs["val_images"]),
        "--val-masks-dir",
        str(dirs["val_masks"]),
        "--load",
        args.base_checkpoint,
        "--load-partial",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--scale",
        str(args.scale),
        "--classes",
        str(task.num_classes),
        "--optimizer",
        args.optimizer,
        "--wandb-mode",
        args.wandb_mode,
        "--num-workers",
        str(args.num_workers),
    ]
    if args.amp:
        command.append("--amp")
    return command


def main():
    args = parse_args()
    tiles_root = Path(args.tiles_root)
    checkpoint_root = Path(args.checkpoint_root)
    tasks = resolve_lesion_tasks(args.tasks)

    for task in tasks:
        checkpoint_dir = checkpoint_root / task.slug
        if args.skip_existing and (checkpoint_dir / "best.pth").exists():
            print("Skipping {} because best.pth already exists".format(task.slug))
            continue

        if not args.dry_run:
            validate_dataset(tiles_root, task)

        summary = dataset_summary(tiles_root, task.slug)
        if summary:
            print(
                "\n==> {} ({}) classes={}, train={}, val={}, positive={}".format(
                    task.name,
                    task.slug,
                    task.num_classes,
                    summary.get("train_tiles"),
                    summary.get("val_tiles"),
                    summary.get("positive_tiles"),
                )
            )
        else:
            print("\n==> {} ({}) classes={}".format(task.name, task.slug, task.num_classes))

        command = build_train_command(args, task)
        print(" ".join('"{}"'.format(item) if " " in item else item for item in command))
        if args.dry_run:
            continue

        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SystemExit("Training failed for task {} with return code {}".format(task.slug, completed.returncode))


if __name__ == "__main__":
    main()
