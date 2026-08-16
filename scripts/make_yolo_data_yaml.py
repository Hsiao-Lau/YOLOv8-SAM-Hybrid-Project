#!/usr/bin/env python
"""Create a minimal Ultralytics YOLO data.yaml."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create YOLO data.yaml.")
    parser.add_argument("--train", required=True, help="Path to train images directory or txt list.")
    parser.add_argument("--val", required=True, help="Path to val images directory or txt list.")
    parser.add_argument("--test", default=None, help="Optional test images directory or txt list.")
    parser.add_argument("--names", nargs="+", required=True, help="Class names, e.g. lesion or CG PZ PCa.")
    parser.add_argument("--output", required=True, help="Output data.yaml path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = {"train": args.train, "val": args.val, "nc": len(args.names), "names": args.names}
    if args.test:
        data["test"] = args.test
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"Saved YOLO data yaml to {out}")


if __name__ == "__main__":
    main()
