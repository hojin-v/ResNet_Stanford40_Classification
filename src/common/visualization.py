from __future__ import annotations

"""학습 결과를 이미지로 저장하기 위한 시각화 helper입니다."""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch

from src.common.train_utils import ensure_dir


def denormalize(
    images: torch.Tensor,
    mean: Iterable[float] = (0.485, 0.456, 0.406),
    std: Iterable[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """정규화된 image tensor를 다시 화면에 보기 좋은 범위로 되돌립니다.

    Dataset에서는 ImageNet mean/std로 normalize합니다. matplotlib으로 시각화하려면
    `image = image * std + mean`을 적용해 다시 `[0, 1]` 범위로 돌려야 합니다.
    """
    device = images.device
    # `[3]` 형태의 mean/std를 `[1, 3, 1, 1]`로 바꿔 batch 전체에 broadcasting합니다.
    mean_t = torch.tensor(tuple(mean), device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(tuple(std), device=device).view(1, -1, 1, 1)
    return (images * std_t + mean_t).clamp(0, 1)


def plot_history(history: dict[str, list[float]], out_path: str | Path) -> None:
    """학습 history를 loss 그래프와 metric 그래프로 저장합니다.

    Args:
        history: `epoch`, `train_loss`, `val_loss`, `val_acc` 같은 리스트를 담은 dict입니다.
        out_path: 저장할 PNG 경로입니다.
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    epochs = history.get("epoch") or list(range(1, len(next(iter(history.values()))) + 1))
    # key 이름에 "loss"가 들어 있으면 왼쪽 loss subplot에 그립니다.
    loss_keys = [key for key in history if "loss" in key]
    # loss와 epoch이 아닌 값은 accuracy 같은 metric으로 간주합니다.
    metric_keys = [key for key in history if key not in set(loss_keys + ["epoch"])]

    ncols = 2 if metric_keys else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    for key in loss_keys:
        axes[0].plot(epochs, history[key], marker="o", label=key)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if metric_keys:
        # 분류 accuracy curve가 여기에 그려집니다.
        for key in metric_keys:
            axes[1].plot(epochs, history[key], marker="o", label=key)
        axes[1].set_title("Metric")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
