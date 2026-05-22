from __future__ import annotations

"""실험 재현성을 높이기 위한 seed 고정 유틸입니다.

완전한 bit-level 재현을 보장하려면 deterministic 설정을 더 강하게 걸어야 하지만,
과제 수준에서는 Python, NumPy, PyTorch의 난수를 같은 seed로 맞추는 것만으로도
데이터 split과 초기화가 크게 흔들리는 문제를 줄일 수 있습니다.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Python/NumPy/PyTorch 난수 seed를 한 번에 고정합니다."""
    # Python 표준 라이브러리 random은 데이터 split과 augmentation 분기에 사용됩니다.
    random.seed(seed)
    # NumPy는 외부 전처리나 배열 연산에서 난수를 사용할 때 영향을 줍니다.
    np.random.seed(seed)
    # hash seed를 고정하면 set/dict 순서가 실험마다 흔들리는 상황을 줄일 수 있습니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    # CPU 텐서 연산의 PyTorch 난수를 고정합니다.
    torch.manual_seed(seed)
    # 여러 GPU를 쓰는 경우까지 포함해 CUDA 난수를 고정합니다.
    torch.cuda.manual_seed_all(seed)
    # 입력 크기가 대체로 고정된 CNN 학습에서는 benchmark=True가 더 빠를 수 있습니다.
    # 완전 재현성이 최우선이라면 deterministic=True와 benchmark=False를 고려합니다.
    torch.backends.cudnn.benchmark = True
