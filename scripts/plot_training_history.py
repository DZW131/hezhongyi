import argparse
import csv
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.visualization import save_training_curves


def parse_value(value: str):
    if value is None:
        return value

    value = value.strip()
    if value == '':
        return value

    lowered = value.lower()
    if lowered == 'nan':
        return float('nan')

    try:
        if any(token in value for token in ('.', 'e', 'E')):
            return float(value)
        return int(value)
    except ValueError:
        return value


def get_args():
    parser = argparse.ArgumentParser(description='Regenerate training curves from analysis/history.csv')
    parser.add_argument('--history-csv', type=str, required=True, help='Path to the history.csv file')
    parser.add_argument('--output', '-o', type=str, default='training_curves.png',
                        help='Output image path')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    history_path = Path(args.history_csv)
    with history_path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        rows = [{key: parse_value(value) for key, value in row.items()} for row in reader]

    output_path = Path(args.output)
    save_training_curves(rows, output_path)
    logging.info('Saved training curves to %s', output_path)
