from __future__ import annotations

"""Stanford40 행동 분류 모델 학습 스크립트입니다.

이 파일은 과제의 Training & Evaluation Loop 요구사항을 담당합니다.
한 epoch마다 train loss/accuracy와 validation loss/accuracy를 기록하고,
validation accuracy가 가장 좋은 모델을 checkpoint로 저장합니다.
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.config import load_config
from src.common.metrics import multiclass_accuracy
from src.common.seed import set_seed
from src.common.train_utils import AverageMeter, append_history_csv, ensure_dir, get_device, save_checkpoint
from src.common.visualization import plot_history
from src.stanford40_classification.dataset import build_datasets
from src.stanford40_classification.model_resnet import build_resnet


def parse_args() -> argparse.Namespace:
    """CLI 인자를 정의합니다.

    YAML 설정을 기본으로 사용하되, 자주 바꾸는 data root/epoch/batch size는
    명령줄에서 바로 덮어쓸 수 있게 했습니다.
    """
    parser = argparse.ArgumentParser(description="Train ResNet on Stanford40.")
    parser.add_argument("--config", default="configs/resnet_stanford40.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def make_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    """설정값에 따라 optimizer를 생성합니다."""
    name = config["training"].get("optimizer", "adamw").lower()
    lr = config["training"]["lr"]
    weight_decay = config["training"].get("weight_decay", 0.0)
    if name == "sgd":
        # CNN 분류에서 고전적으로 많이 쓰는 SGD + momentum 조합입니다.
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adamw":
        # AdamW는 learning rate 튜닝이 비교적 수월해 입문 실험에 편합니다.
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """학습 데이터셋을 한 epoch 학습합니다."""
    # BatchNorm/Dropout이 학습 모드로 동작하도록 설정합니다.
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels, _ in tqdm(loader, desc="train", leave=False):
        # non_blocking=True는 pin_memory=True DataLoader와 함께 GPU 전송을 조금 더 효율적으로 만듭니다.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 이전 batch의 gradient가 남아 있지 않도록 초기화합니다.
        optimizer.zero_grad(set_to_none=True)
        # logits shape: `[B, num_classes]`
        logits = model(images)
        # CrossEntropyLoss는 softmax 전 logits와 class index label을 입력으로 받습니다.
        loss = criterion(logits, labels)
        # loss를 기준으로 모든 parameter의 gradient를 계산합니다.
        loss.backward()
        # optimizer가 gradient를 사용해 parameter를 갱신합니다.
        optimizer.step()

        batch_size = images.size(0)
        # batch 크기를 가중치로 주어 epoch 평균을 정확히 계산합니다.
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(multiclass_accuracy(logits.detach(), labels), batch_size)

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """검증 또는 테스트 데이터셋을 평가합니다."""
    # 평가 모드에서는 BatchNorm running statistics를 사용하고 Dropout을 끕니다.
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels, _ in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(multiclass_accuracy(logits, labels), batch_size)

    return loss_meter.avg, acc_meter.avg


def main() -> None:
    """설정 로드부터 학습 종료 후 curve 저장까지 전체 학습 절차를 실행합니다."""
    args = parse_args()
    config = load_config(args.config)
    # CLI 인자가 주어지면 YAML보다 우선합니다. 실험을 빠르게 반복하기 위한 장치입니다.
    if args.data_root is not None:
        config["data"]["root"] = args.data_root
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))
    # 결과물은 output_dir 아래에 history, curve, checkpoint로 나뉘어 저장됩니다.
    output_dir = ensure_dir(config["training"]["output_dir"])
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")

    # Dataset 생성 단계에서 train/val split과 class_to_idx가 결정됩니다.
    train_dataset, val_dataset, _ = build_datasets(
        root=config["data"]["root"],
        image_size=config["data"]["image_size"],
        val_ratio=config["data"]["val_ratio"],
        seed=config.get("seed", 42),
    )

    # 학습 DataLoader는 매 epoch 샘플 순서를 섞습니다.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )
    # 검증 DataLoader는 metric 재현성을 위해 shuffle=False입니다.
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    # class 개수는 데이터셋에서 자동으로 얻습니다. Stanford40이면 보통 40입니다.
    model = build_resnet(config["model"]["name"], num_classes=len(train_dataset.class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(config, model)
    # CosineAnnealingLR은 epoch이 진행될수록 learning rate를 부드럽게 낮춥니다.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    # 그래프를 그리기 위해 epoch별 값을 메모리에도 보관합니다.
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_acc = -1.0

    for epoch in range(1, config["training"]["epochs"] + 1):
        # 1. train split으로 parameter를 업데이트합니다.
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # 2. validation split으로 현재 모델의 일반화 성능을 측정합니다.
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # CSV에 저장할 한 epoch의 요약 row입니다.
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        append_history_csv(output_dir / "history.csv", row)
        for key in history:
            history[key].append(row[key])

        # checkpoint에는 모델 가중치뿐 아니라 optimizer와 config도 같이 저장합니다.
        # 나중에 같은 설정으로 평가/시각화하기 위해 class_names도 포함합니다.
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "class_names": train_dataset.class_names,
            "config": config,
            "val_acc": val_acc,
        }
        save_checkpoint(state, checkpoint_dir / "last.pt")
        if val_acc > best_val_acc:
            # overfitting을 피하기 위해 마지막 모델이 아니라 validation 기준 best 모델을 따로 보관합니다.
            best_val_acc = val_acc
            save_checkpoint(state, checkpoint_dir / "best.pt")

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    # 학습이 끝나면 loss/metric curve를 PNG로 저장합니다.
    plot_history(history, output_dir / "curves.png")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Artifacts saved to: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
