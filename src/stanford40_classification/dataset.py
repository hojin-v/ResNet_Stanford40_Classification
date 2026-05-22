from __future__ import annotations

"""Stanford40 행동 분류 데이터셋을 PyTorch Dataset으로 감싸는 코드입니다.

Kaggle에 올라온 데이터는 압축을 푸는 방식에 따라 폴더 구조가 조금 달라질 수 있습니다.
그래서 이 파일은 다음 두 가지 구조를 모두 지원하도록 작성했습니다.

1. ImageFolder 구조: `train_FUll/<class_name>/*.jpg`
2. Stanford40 원본 구조: `JPEGImages/`와 `ImageSplits/`

최종적으로는 `(image_tensor, label_tensor, image_path)`를 반환해 학습과 시각화에 모두 사용할 수 있게 합니다.
"""

import random
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ClassificationSample:
    """이미지 1장의 경로와 class 이름을 함께 보관하는 작은 자료구조입니다."""

    path: Path
    label_name: str


def _norm_name(name: str) -> str:
    """폴더명/파일명을 비교하기 쉽게 소문자 snake_case 형태로 정규화합니다."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _iter_images(root: Path) -> list[Path]:
    """root 아래의 이미지 파일을 재귀적으로 모두 찾습니다."""
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def _is_imagefolder(root: Path) -> bool:
    """root가 `class_name/images...` 형태의 ImageFolder 구조인지 확인합니다."""
    # class 폴더가 하나라도 있고, 그 안에 이미지가 있으면 ImageFolder로 판단합니다.
    return any(child.is_dir() and _iter_images(child) for child in root.iterdir())


def _collect_imagefolder(root: Path) -> list[ClassificationSample]:
    """ImageFolder 구조에서 이미지 경로와 class 이름을 수집합니다."""
    samples: list[ClassificationSample] = []
    for class_dir in sorted(child for child in root.iterdir() if child.is_dir()):
        # ImageFolder에서는 폴더명이 곧 label 이름입니다.
        for image_path in _iter_images(class_dir):
            samples.append(ClassificationSample(image_path, class_dir.name))
    return samples


def _find_split_imagefolder(root: Path, split: str) -> Path | None:
    """train/val/test에 해당하는 ImageFolder split 폴더를 찾습니다.

    Kaggle 데이터셋은 `train_FUll`처럼 대소문자나 underscore가 들쑥날쑥할 수 있으므로,
    폴더명을 정규화한 뒤 여러 alias와 비교합니다.
    """
    split_aliases = {
        "train": {"train", "training", "train_full", "trainset"},
        "val": {"val", "valid", "validation", "dev"},
        "test": {"test", "testing", "test_full", "testset"},
    }
    aliases = split_aliases[split]
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        # 예: root/train_FUll/action/*.jpg 같은 구조를 찾습니다.
        if _norm_name(child.name) in aliases and _is_imagefolder(child):
            return child
    has_named_split = any(
        _norm_name(child.name) in set().union(*split_aliases.values())
        for child in root.iterdir()
        if child.is_dir()
    )
    # 사용자가 root로 이미 class 폴더들의 부모를 넘긴 경우를 지원합니다.
    # 단, val/test에 이 fallback을 허용하면 train 데이터를 검증으로 중복 사용할 수 있어 train에만 허용합니다.
    if split == "train" and not has_named_split:
        if _is_imagefolder(root):
            return root
    return None


def _collect_stanford_original(root: Path, split: str) -> list[ClassificationSample]:
    """Stanford40 원본 `JPEGImages`/`ImageSplits` 구조에서 샘플을 수집합니다."""
    image_root = root / "JPEGImages"
    split_root = root / "ImageSplits"
    if not image_root.exists() or not split_root.exists():
        return []

    samples: list[ClassificationSample] = []
    # 원본 Stanford40에는 보통 train/test split만 있으므로 val은 이후 train에서 따로 분리합니다.
    if split == "val":
        return samples
    suffix = "train" if split == "train" else "test"
    for split_file in sorted(split_root.glob(f"*_{suffix}.txt")):
        # 파일명이 `applauding_train.txt`이면 class 이름은 `applauding`입니다.
        label_name = split_file.stem[: -len(f"_{suffix}")]
        for line in split_file.read_text(encoding="utf-8").splitlines():
            image_name = line.strip()
            if not image_name:
                continue
            image_path = image_root / image_name
            if image_path.exists():
                samples.append(ClassificationSample(image_path, label_name))
    return samples


def discover_samples(root: str | Path, split: str) -> list[ClassificationSample]:
    """지정한 split에 해당하는 classification sample 목록을 찾습니다."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}. Put the Kaggle files there or pass --data-root."
        )

    # 먼저 Stanford40 원본 구조를 확인합니다.
    original_samples = _collect_stanford_original(root, split)
    if original_samples:
        return original_samples

    # 원본 구조가 아니면 ImageFolder 스타일 split 폴더를 탐색합니다.
    imagefolder = _find_split_imagefolder(root, split)
    if imagefolder is not None:
        return _collect_imagefolder(imagefolder)

    return []


def split_train_val(
    samples: list[ClassificationSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[ClassificationSample], list[ClassificationSample]]:
    """train sample을 class별로 train/validation으로 나눕니다.

    단순 랜덤 split은 데이터가 작을 때 validation에 특정 class가 빠질 수 있습니다.
    그래서 class별로 섞고 나눈 뒤 합칩니다.
    """
    by_class: dict[str, list[ClassificationSample]] = {}
    for sample in samples:
        by_class.setdefault(sample.label_name, []).append(sample)

    rng = random.Random(seed)
    train_samples: list[ClassificationSample] = []
    val_samples: list[ClassificationSample] = []
    for label_name, class_samples in by_class.items():
        shuffled = class_samples[:]
        rng.shuffle(shuffled)
        # class당 이미지가 2장 이상이면 최소 1장은 validation으로 보내도록 합니다.
        val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
        val_samples.extend(shuffled[:val_count])
        train_samples.extend(shuffled[val_count:])

    return sorted(train_samples, key=lambda s: str(s.path)), sorted(val_samples, key=lambda s: str(s.path))


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """분류 모델 입력용 image transform을 만듭니다."""
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if train:
        # 학습에서는 crop/flip/color jitter로 과적합을 줄이고 다양한 입력을 보게 합니다.
        return transforms.Compose(
            [
                transforms.Resize(image_size + 32),
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )
    # 평가에서는 랜덤성이 없어야 하므로 resize 후 center crop만 사용합니다.
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


class Stanford40Dataset(Dataset):
    """Stanford40 sample 목록을 PyTorch Dataset 인터페이스로 제공합니다."""

    def __init__(
        self,
        samples: list[ClassificationSample],
        class_to_idx: dict[str, int],
        image_size: int,
        train: bool,
    ) -> None:
        if not samples:
            raise ValueError("No classification samples were found.")
        self.samples = samples
        self.class_to_idx = class_to_idx
        # index -> class 이름 변환은 예측 시각화에서 사용합니다.
        self.class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
        self.transform = build_transforms(image_size, train)

    def __len__(self) -> int:
        """Dataset 전체 sample 수를 반환합니다."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """index번째 이미지를 읽어 tensor와 label로 변환합니다.

        Returns:
            image tensor: `[3, image_size, image_size]`
            label tensor: scalar long tensor
            path string: 시각화나 디버깅용 원본 이미지 경로
        """
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            # 흑백/팔레트 이미지가 섞여 있어도 모델 입력 channel을 3개로 통일합니다.
            image = image.convert("RGB")
            tensor = self.transform(image)
        label = torch.tensor(self.class_to_idx[sample.label_name], dtype=torch.long)
        return tensor, label, str(sample.path)


def build_datasets(
    root: str | Path,
    image_size: int,
    val_ratio: float,
    seed: int,
) -> tuple[Stanford40Dataset, Stanford40Dataset, Stanford40Dataset | None]:
    """train/val/test Dataset 객체를 한 번에 생성합니다."""
    train_all = discover_samples(root, "train")
    explicit_val = discover_samples(root, "val")
    test_samples = discover_samples(root, "test")

    if not train_all:
        raise RuntimeError(
            "Could not find Stanford40 training images. Expected either an ImageFolder layout "
            "such as train_FUll/<class>/*.jpg or the original JPEGImages/ImageSplits layout."
        )

    if explicit_val:
        # validation 폴더가 따로 있으면 그것을 그대로 사용합니다.
        train_samples, val_samples = train_all, explicit_val
    else:
        # validation 폴더가 없으면 train에서 class별 stratified split을 만듭니다.
        train_samples, val_samples = split_train_val(train_all, val_ratio, seed)

    # train과 validation에 등장한 class만 학습 대상으로 삼습니다.
    class_names = sorted({sample.label_name for sample in train_samples + val_samples})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_dataset = Stanford40Dataset(train_samples, class_to_idx, image_size, train=True)
    val_dataset = Stanford40Dataset(val_samples, class_to_idx, image_size, train=False)
    test_dataset = None
    if test_samples:
        # test에 학습 때 보지 않은 class 폴더가 섞여 있으면 평가할 수 없으므로 제외합니다.
        known_test_samples = [sample for sample in test_samples if sample.label_name in class_to_idx]
        test_dataset = Stanford40Dataset(known_test_samples, class_to_idx, image_size, train=False)
    return train_dataset, val_dataset, test_dataset
