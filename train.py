from __future__ import annotations

import argparse

from fcn_training.runner import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one FCN task from an offline synthetic dataset."
    )
    parser.add_argument("--config", required=True, help="Path to training YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
