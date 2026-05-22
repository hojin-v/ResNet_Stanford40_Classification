from __future__ import annotations

"""Stanford40 데이터셋 탐색이 정상 동작하는지 확인하는 진단 스크립트입니다."""

import argparse
from pathlib import Path

from src.stanford40_classification.dataset import build_datasets


def parse_args() -> argparse.Namespace:
    """데이터셋 확인에 필요한 CLI 인자를 정의합니다."""
    parser = argparse.ArgumentParser(description="Check Stanford40 dataset discovery.")
    parser.add_argument("--root", default="data/stanford40")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Dataset builder를 호출하고 class/sample 수를 출력합니다."""
    args = parse_args()
    train, val, test = build_datasets(Path(args.root), args.image_size, args.val_ratio, args.seed)
    print(f"classes={len(train.class_names)}")
    print(f"train={len(train)} val={len(val)} test={len(test) if test else 0}")
    print("first_classes=" + ", ".join(train.class_names[:10]))


if __name__ == "__main__":
    main()
