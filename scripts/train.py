"""Train a model on a MAD task.

Usage:
    uv run scripts/train.py configs/minimal.yaml [section.key=value ...]

Example:
    uv run scripts/train.py configs/minimal.yaml task.name=memorization train.epochs=50
"""
import argparse
from pathlib import Path

from forgetting_study.config import load_config
from forgetting_study.train import make_run_dir, train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="path to a YAML config")
    parser.add_argument("overrides", nargs="*", help="section.key=value overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    run_dir = make_run_dir(cfg, Path(args.config).stem)
    print(f"run directory: {run_dir}")
    train(cfg, run_dir)


if __name__ == "__main__":
    main()
