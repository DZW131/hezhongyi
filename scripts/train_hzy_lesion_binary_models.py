import argparse
import json
import subprocess
import sys
from pathlib import Path

from hzy_lesion_config import resolve_lesion_labels


DEFAULT_TILES_ROOT = "/root/datasets/HZY_HSPN_lesion_tiles"
DEFAULT_CHECKPOINT_ROOT = "/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_lesion_binary"
DEFAULT_BASE_CHECKPOINT = "/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_finetune_glom_scene_ds025_resume/best.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one binary U-Net model per HZY lesion tile dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tiles-root", default=DEFAULT_TILES_ROOT, help="Root directory produced by prepare_hzy_lesion_tiles.py")
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT, help="Root directory for lesion checkpoints")
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT,
                        help="Checkpoint used to initialize each lesion model")
    parser.add_argument("--preset", choices=("trainable", "all", "rare"), default="trainable",
                        help="Default lesion label set when --labels is omitted")
    parser.add_argument("--labels", nargs="*", default=[],
                        help="Optional lesion labels or slugs to train")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per lesion model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--scale", type=float, default=1.0, help="Training image scale")
    parser.add_argument("--classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--optimizer", choices=("rmsprop", "adamw"), default="adamw", help="Optimizer")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled",
                        help="Weights & Biases mode")
    parser.add_argument("--amp", action="store_true", default=False, help="Use AMP")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="Skip a lesion if checkpoint best.pth already exists")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Print commands without running them")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def dataset_dirs(tiles_root: Path, slug: str):
    dataset_root = tiles_root / slug
    return {
        "train_images": dataset_root / "train" / "images",
        "train_masks": dataset_root / "train" / "masks",
        "val_images": dataset_root / "val" / "images",
        "val_masks": dataset_root / "val" / "masks",
    }


def dataset_summary(tiles_root: Path, slug: str):
    summary_path = tiles_root / slug / "manifests" / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_train_command(args, lesion) -> list:
    dirs = dataset_dirs(Path(args.tiles_root), lesion.slug)
    checkpoint_dir = Path(args.checkpoint_root) / lesion.slug
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
        str(args.classes),
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


def validate_dataset(tiles_root: Path, lesion):
    dirs = dataset_dirs(tiles_root, lesion.slug)
    missing = [str(path) for path in dirs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing tile directories for '{}': {}. Run prepare_hzy_lesion_tiles.py first.".format(
                lesion.label,
                ", ".join(missing),
            )
        )


def main():
    args = parse_args()
    tiles_root = Path(args.tiles_root)
    checkpoint_root = Path(args.checkpoint_root)
    lesions = resolve_lesion_labels(args.preset, args.labels)

    for lesion in lesions:
        if not args.dry_run:
            validate_dataset(tiles_root, lesion)
        checkpoint_dir = checkpoint_root / lesion.slug
        if args.skip_existing and (checkpoint_dir / "best.pth").exists():
            print("Skipping {} because best.pth already exists".format(lesion.slug))
            continue

        summary = dataset_summary(tiles_root, lesion.slug)
        if summary:
            print(
                "\n==> {} ({}) tiles: train={}, val={}, positive={}".format(
                    lesion.label,
                    lesion.slug,
                    summary.get("train_tiles"),
                    summary.get("val_tiles"),
                    summary.get("positive_tiles"),
                )
            )
        else:
            print("\n==> {} ({})".format(lesion.label, lesion.slug))

        command = build_train_command(args, lesion)
        print(" ".join('"{}"'.format(item) if " " in item else item for item in command))
        if args.dry_run:
            continue

        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SystemExit("Training failed for {} with return code {}".format(lesion.slug, completed.returncode))


if __name__ == "__main__":
    main()
