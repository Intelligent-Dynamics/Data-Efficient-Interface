"""Usage: python -m baseline prepare | run --shots 5 --seed 11 --output PATH."""

import argparse
import json
from pathlib import Path

from .data import SEEDS, SHOTS, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--raw", type=Path, default=Path("data/raw/banking77"))
        command.add_argument("--manifest", type=Path,
                             default=Path("data/processed/banking77/manifest.json"))
        if name == "run":
            command.add_argument("--shots", type=int, choices=SHOTS, required=True)
            command.add_argument("--seed", type=int, choices=SEEDS, required=True)
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(args.raw, args.manifest)
        print(json.dumps({k: v for k, v in manifest["audit"].items()
                          if k != "available_training_per_class"}, indent=2))
    else:
        from .experiment import run

        metrics = run(args.raw, args.manifest, args.output, args.shots, args.seed)
        result = metrics["tfidf_logistic_regression"]
        print(json.dumps({"output": str(args.output), "evaluation_split": "validation",
                          "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
