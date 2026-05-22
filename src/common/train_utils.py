from __future__ import annotations

"""Stanford40 분류 학습 루프에서 반복적으로 쓰이는 helper 모음입니다."""

import csv
from pathlib import Path
from typing import Any

import torch


class AverageMeter:
    """batch별 값을 sample 수로 가중 평균 내기 위한 누적기입니다.

    loss나 accuracy를 단순히 batch 개수로 평균 내면 마지막 batch 크기가 작을 때
    약간의 오차가 생길 수 있습니다. 그래서 `n=batch_size`를 함께 받아 sample 기준 평균을 냅니다.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """누적합과 sample 개수를 초기화합니다."""
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        """새 값을 누적합니다.

        Args:
            value: 현재 batch의 평균 loss 또는 metric입니다.
            n: 현재 batch에 들어 있는 sample 수입니다.
        """
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        """지금까지 누적한 값의 sample 기준 평균입니다."""
        return self.total / max(self.count, 1)


def ensure_dir(path: str | Path) -> Path:
    """폴더가 없으면 만들고, Path 객체로 반환합니다."""
    out = Path(path)
    # parents=True라서 `outputs/resnet_stanford40/checkpoints`처럼 중첩 경로도 한 번에 생성됩니다.
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_device(prefer: str = "auto") -> torch.device:
    """학습에 사용할 torch device를 결정합니다.

    Args:
        prefer: `"auto"`이면 CUDA가 있을 때 GPU를, 없으면 CPU를 선택합니다.
                `"cpu"` 또는 `"cuda:0"`처럼 명시 device도 받을 수 있습니다.
    """
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """모델/옵티마이저/설정 등을 checkpoint 파일로 저장합니다."""
    path = Path(path)
    ensure_dir(path.parent)
    # torch.save는 tensor가 포함된 dict를 PyTorch가 다시 읽을 수 있는 형태로 직렬화합니다.
    torch.save(state, path)


def append_history_csv(path: str | Path, row: dict[str, Any]) -> None:
    """epoch별 학습 로그를 CSV에 한 줄씩 추가합니다.

    파일이 처음 만들어지는 경우에는 header도 함께 작성합니다.
    `history.csv`는 나중에 그래프를 다시 그리거나 보고서 표를 만들 때 사용합니다.
    """
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            # 첫 row를 쓰기 전 column 이름을 저장합니다.
            writer.writeheader()
        writer.writerow(row)
