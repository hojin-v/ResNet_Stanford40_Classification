from __future__ import annotations

"""Stanford40 분류 모델 평가에 사용하는 metric"""

import torch


@torch.no_grad()
def multiclass_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """다중 클래스 분류 정확도(Accuracy)를 계산

    Args:
        logits: 모델의 softmax 전 출력. shape는 `[B, num_classes]`
        targets: 정답 class index. shape는 `[B]`

    Returns:
        batch 단위 accuracy를 Python float로 반환
    """
    # CrossEntropyLoss와 마찬가지로 softmax를 별도로 적용하지 않아도 argmax 결과는 같다.
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()
