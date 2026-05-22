from __future__ import annotations

"""저장된 ResNet checkpoint를 평가하는 스크립트입니다."""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.common.config import load_config
from src.common.train_utils import get_device
from src.stanford40_classification.dataset import build_datasets
from src.stanford40_classification.model_resnet import build_resnet
from src.stanford40_classification.train import evaluate


def parse_args() -> argparse.Namespace:
    """평가에 필요한 checkpoint/config/data root 인자를 정의합니다."""
    parser = argparse.ArgumentParser(description="Evaluate a trained ResNet checkpoint.")
    parser.add_argument("--config", default="configs/resnet_stanford40.yaml")
    parser.add_argument("--checkpoint", default="outputs/resnet_stanford40/checkpoints/best.pt")
    parser.add_argument("--data-root", default=None)
    return parser.parse_args()


def main() -> None:
    """checkpoint를 불러와 test 또는 validation split에서 loss/accuracy를 출력합니다."""
    args = parse_args()
    # map_location="cpu"로 먼저 읽으면 GPU가 없는 환경에서도 checkpoint를 열 수 있습니다.
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    # 학습 때 저장된 config가 있으면 그것을 우선 사용합니다.
    config = checkpoint.get("config") or load_config(args.config)
    if args.data_root is not None:
        config["data"]["root"] = args.data_root

    device = get_device(config.get("device", "auto"))
    # class_to_idx를 학습 때와 동일하게 맞추기 위해 train_dataset도 함께 생성합니다.
    train_dataset, val_dataset, test_dataset = build_datasets(
        root=config["data"]["root"],
        image_size=config["data"]["image_size"],
        val_ratio=config["data"]["val_ratio"],
        seed=config.get("seed", 42),
    )
    # 명시적 test split이 있으면 test를 쓰고, 없으면 validation으로 평가합니다.
    dataset = test_dataset or val_dataset
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    # checkpoint의 FC layer 크기와 맞도록 학습 데이터의 class 수로 모델을 만듭니다.
    model = build_resnet(config["model"]["name"], num_classes=len(train_dataset.class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loss, acc = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    split_name = "test" if test_dataset is not None else "validation"
    print(f"{split_name}_loss={loss:.4f} {split_name}_accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
