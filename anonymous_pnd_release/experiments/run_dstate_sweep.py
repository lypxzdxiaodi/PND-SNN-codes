"""Launch the linear-expansion dimension ablation sequentially."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--states", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    for state in args.states:
        command = [
            sys.executable,
            "-m",
            "experiments.train",
            "--task", args.task,
            "--data-dir", args.data_dir,
            "--output", str(Path(args.output) / f"d_state_{state}"),
            "--d-state", str(state),
            "--epochs", str(args.epochs),
            *args.extra,
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
