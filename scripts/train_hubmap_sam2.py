import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG = "configs/sam2.1_training/sam2.1_hiera_b+_hubmap_glomerulus.yaml"
PROFILE_FAST_4090 = "fast_4090"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convenience launcher for HuBMAP glomerulus SAM2 training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Hydra config path inside the sam2 config module.")
    parser.add_argument("--profile", type=str, default="default", choices=("default", PROFILE_FAST_4090), help="Convenience preset for common training scenarios.")
    parser.add_argument("--dataset-root", type=str, default="", help="Prepared dataset root created by prepare_hubmap_sam2_from_tiles.py.")
    parser.add_argument("--init-checkpoint", type=str, default="", help="SAM2 initialization checkpoint, for example sam2.1_hiera_base_plus.pt.")
    parser.add_argument("--output-dir", type=str, required=True, help="Experiment directory used for logs and checkpoints.")
    parser.add_argument("--resume-from", type=str, default="", help="Optional checkpoint to resume from when starting a fresh output directory.")
    parser.add_argument("--num-gpus", type=int, default=1, help="GPUs per node.")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes.")
    parser.add_argument("--train-batch-size", type=int, default=0, help="Optional override for scratch.train_batch_size.")
    parser.add_argument("--val-batch-size", type=int, default=0, help="Optional override for scratch.val_batch_size.")
    parser.add_argument("--resolution", type=int, default=0, help="Optional override for scratch.resolution.")
    parser.add_argument("--num-workers", type=int, default=0, help="Optional override for scratch.num_train_workers.")
    parser.add_argument("--prefetch-factor", type=int, default=0, help="Optional override for scratch.prefetch_factor.")
    parser.add_argument("--disable-persistent-workers", action="store_true", help="Disable DataLoader persistent workers.")
    parser.add_argument("--num-epochs", type=int, default=0, help="Optional override for scratch.num_epochs.")
    parser.add_argument("--val-epoch-freq", type=int, default=0, help="Optional override for trainer.val_epoch_freq.")
    parser.add_argument("--compile-image-encoder", action="store_true", help="Enable torch.compile on the image encoder for faster long runs after compile warmup.")
    parser.add_argument("--use-cluster", type=int, default=0, help="0 for local run, 1 for SLURM/submitit.")
    parser.add_argument("--partition", type=str, default="", help="Optional SLURM partition.")
    parser.add_argument("--account", type=str, default="", help="Optional SLURM account.")
    parser.add_argument("--qos", type=str, default="", help="Optional SLURM qos.")
    parser.add_argument("--hydra-override", action="append", default=[], help="Additional Hydra override, can be passed multiple times.")
    parser.add_argument("--summary-only", action="store_true", help="Skip training and only generate analysis artifacts from an existing run directory.")
    parser.add_argument("--skip-summary", action="store_true", help="Skip post-training history summary generation.")
    return parser.parse_args()


def apply_profile_defaults(args):
    if args.profile != PROFILE_FAST_4090:
        return args

    if args.resolution <= 0:
        args.resolution = 768
    if args.num_workers <= 0:
        args.num_workers = 8
    if args.prefetch_factor <= 0:
        args.prefetch_factor = 4
    if args.num_epochs <= 0:
        args.num_epochs = 20
    if args.val_epoch_freq <= 0:
        args.val_epoch_freq = 4
    if not args.compile_image_encoder:
        args.compile_image_encoder = True

    return args


def main():
    args = apply_profile_defaults(parse_args())
    repo_root = Path(__file__).resolve().parents[1]

    if args.summary_only:
        subprocess.run(
            [sys.executable, "scripts/summarize_hubmap_sam2_run.py", "--run-dir", args.output_dir],
            cwd=repo_root,
            check=True,
        )
        return
    if not args.dataset_root:
        raise ValueError("--dataset-root is required unless --summary-only is set.")
    if not args.init_checkpoint:
        raise ValueError("--init-checkpoint is required unless --summary-only is set.")

    command = [
        sys.executable,
        "training/train.py",
        "-c",
        args.config,
        "--use-cluster",
        str(args.use_cluster),
        "--num-gpus",
        str(args.num_gpus),
        "--num-nodes",
        str(args.num_nodes),
        "--hydra-override",
        f"paths.dataset_root={args.dataset_root}",
        "--hydra-override",
        f"paths.init_checkpoint={args.init_checkpoint}",
        "--hydra-override",
        f"paths.experiment_dir={args.output_dir}",
    ]
    if args.resume_from:
        # `resume_from` is an optional dataclass field on the trainer checkpoint config.
        # Hydra treats it as an appended key in this composed config tree, so we add it
        # with the `+` prefix instead of overriding a pre-existing YAML entry.
        command.extend(["--hydra-override", f"+trainer.checkpoint.resume_from={args.resume_from}"])
    if args.train_batch_size > 0:
        command.extend(["--hydra-override", f"scratch.train_batch_size={args.train_batch_size}"])
    if args.val_batch_size > 0:
        command.extend(["--hydra-override", f"scratch.val_batch_size={args.val_batch_size}"])
    if args.resolution > 0:
        command.extend(["--hydra-override", f"scratch.resolution={args.resolution}"])
    if args.num_workers > 0:
        command.extend(["--hydra-override", f"scratch.num_train_workers={args.num_workers}"])
    if args.prefetch_factor > 0:
        command.extend(["--hydra-override", f"scratch.prefetch_factor={args.prefetch_factor}"])
    if args.disable_persistent_workers:
        command.extend(["--hydra-override", "scratch.persistent_workers=False"])
    if args.num_epochs > 0:
        command.extend(["--hydra-override", f"scratch.num_epochs={args.num_epochs}"])
    if args.val_epoch_freq > 0:
        command.extend(["--hydra-override", f"trainer.val_epoch_freq={args.val_epoch_freq}"])
    if args.compile_image_encoder:
        command.extend(["--hydra-override", "trainer.model.compile_image_encoder=True"])

    if args.partition:
        command.extend(["--partition", args.partition])
    if args.account:
        command.extend(["--account", args.account])
    if args.qos:
        command.extend(["--qos", args.qos])
    for override in args.hydra_override:
        command.extend(["--hydra-override", override])

    subprocess.run(command, cwd=repo_root, check=True)

    if not args.skip_summary:
        subprocess.run(
            [sys.executable, "scripts/summarize_hubmap_sam2_run.py", "--run-dir", args.output_dir],
            cwd=repo_root,
            check=True,
        )


if __name__ == "__main__":
    main()
