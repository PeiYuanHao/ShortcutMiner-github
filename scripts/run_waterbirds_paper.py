"""Actual CUDA model capacity/training entry point; never routes to old metadata audit."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shortcutminer.experiment.waterbirds_paper import capacity, load_config, run_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["capacity", "train"])
    parser.add_argument("--config", default="configs/experiments/waterbirds-paper-v1.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--acceptance", help="Matching capacity.json required for train")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "capacity":
        capacity(config, args.output)
    else:
        if not args.acceptance:
            parser.error("train requires --acceptance")
        run_matrix(config, args.output, args.acceptance)


if __name__ == "__main__":
    main()
