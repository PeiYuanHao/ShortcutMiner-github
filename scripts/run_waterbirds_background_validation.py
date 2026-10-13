"""Verify official masks and measure paired background responses; no model fitting."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shortcutminer.experiment.waterbirds_background_validation import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/waterbirds-paper-v1.json")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--assets", default="E:/ShortcutMiner-Experiments/waterbirds-validation-assets")
    args = parser.parse_args()
    run(args.config, args.matrix, args.assets)
