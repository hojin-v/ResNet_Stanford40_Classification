from __future__ import annotations

"""Stanford40 분류 모델의 예측 결과를 이미지 grid로 저장하는 스크립트입니다."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.common.config import load_config
from src.common.train_utils import ensure_dir, get_device
from src.common.visualization import denormalize
from src.stanford40_classification.dataset import build_datasets
from src.stanford40_classification.model_resnet import build_resnet


def parse_args() -> argparse.Namespace:
    """시각화에 필요한 checkpoint와 출력 경로 인자를 정의합니다."""
    parser = argparse.ArgumentParser(description="Visualize Stanford40 predictions.")
    parser.add_argument("--config", default="configs/resnet_stanford40.yaml")
    parser.add_argument("--checkpoint", default="outputs/resnet_stanford40/checkpoints/best.pt")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--out", default="outputs/resnet_stanford40/predictions.png")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """검증/테스트 이미지 일부를 뽑아 GT와 예측 class를 함께 그립니다."""
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config") or load_config(args.config)
    if args.data_root is not None:
        config["data"]["root"] = args.data_root

    device = get_device(config.get("device", "auto"))
    # test split이 있으면 test에서, 없으면 validation에서 이미지를 가져옵니다.
    train_dataset, val_dataset, test_dataset = build_datasets(
        root=config["data"]["root"],
        image_size=config["data"]["image_size"],
        val_ratio=config["data"]["val_ratio"],
        seed=config.get("seed", 42),
    )
    dataset = test_dataset or val_dataset
    # 시각화는 무작위 샘플을 보는 편이 실패 사례 탐색에 유용하므로 shuffle=True입니다.
    loader = DataLoader(dataset, batch_size=args.num_images, shuffle=True, num_workers=0)

    model = build_resnet(config["model"]["name"], num_classes=len(train_dataset.class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 한 batch만 뽑아 예측 grid를 만듭니다.
    images, labels, _ = next(iter(loader))
    logits = model(images.to(device))
    preds = logits.argmax(dim=1).cpu()
    # normalize된 tensor는 바로 보면 색이 이상하므로 denormalize 후 `[H, W, C]`로 바꿉니다.
    images = denormalize(images).permute(0, 2, 3, 1).cpu().numpy()

    n = min(args.num_images, len(images))
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = [axes] if n == 1 else axes.reshape(-1)

    class_names = train_dataset.class_names
    for i in range(n):
        axes[i].imshow(images[i])
        gt = class_names[int(labels[i])]
        pred = class_names[int(preds[i])]
        # 맞춘 샘플은 초록색, 틀린 샘플은 빨간색 title로 표시해 실패 사례를 빨리 찾게 합니다.
        color = "green" if gt == pred else "red"
        axes[i].set_title(f"GT: {gt}\nPred: {pred}", color=color, fontsize=9)
        axes[i].axis("off")
    for ax in axes[n:]:
        # grid 칸이 남는 경우 빈 subplot을 숨깁니다.
        ax.axis("off")

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved prediction grid to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
