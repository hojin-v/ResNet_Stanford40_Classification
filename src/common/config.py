from __future__ import annotations

"""YAML 설정 파일을 읽는 공통 유틸입니다."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML 파일을 Python dict로 읽어 옵니다.

    Args:
        path: `configs/*.yaml`처럼 실험 설정이 저장된 파일 경로입니다.

    Returns:
        YAML의 key/value 구조를 그대로 담은 dict입니다.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
