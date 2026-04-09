import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hubmap_sam2.training_summary import summarize_training_run


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize a HuBMAP SAM2 training run.")
    parser.add_argument("--run-dir", type=str, required=True, help="Experiment directory containing logs/ and checkpoints/.")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = summarize_training_run(Path(args.run_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
