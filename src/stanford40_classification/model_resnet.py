from __future__ import annotations

"""ResNet을 torchvision 호출 없이 직접 구현한 파일입니다.

과제의 핵심 학습 포인트가 residual block과 skip connection이므로,
`torchvision.models.resnet18`을 바로 쓰지 않고 기본 block부터 작성했습니다.
"""

from collections.abc import Callable

import torch
from torch import nn


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """ResNet basic block에서 반복적으로 사용하는 3x3 convolution입니다."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """ResNet-18/34에서 사용하는 basic residual block입니다.

    block 내부의 main branch는 3x3 conv 두 개이고, shortcut branch는 대부분 identity입니다.
    feature map 크기나 channel 수가 바뀌는 경우에만 downsample projection을 사용합니다.
    """

    # Bottleneck block에서는 expansion이 4지만, BasicBlock은 출력 channel이 planes와 같으므로 1입니다.
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # 첫 convolution에서 stride=2가 들어가면 feature map 해상도가 절반으로 줄어듭니다.
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 두 번째 convolution은 block의 residual F(x)를 완성합니다.
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # identity와 residual의 shape가 다를 때 shortcut을 맞춰 주는 projection입니다.
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 x에 대해 `F(x) + x` 형태의 residual 연산을 수행합니다."""
        # shortcut branch입니다. 기본값은 입력을 그대로 더하는 identity mapping입니다.
        identity = x

        # main branch: Conv-BN-ReLU-Conv-BN
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            # stride나 channel 수가 달라졌다면 identity도 같은 shape로 변환해야 덧셈이 가능합니다.
            identity = self.downsample(x)

        # residual learning의 핵심입니다. main branch가 학습한 변화량 F(x)를 원래 입력에 더합니다.
        out = out + identity
        return self.relu(out)


class ResNet(nn.Module):
    """ImageNet 스타일 stem과 4개 stage를 가진 ResNet 본체입니다."""

    def __init__(
        self,
        block: type[BasicBlock],
        layers: list[int],
        num_classes: int,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        # 현재 stage에 들어가는 channel 수를 추적합니다. _make_layer가 이 값을 갱신합니다.
        self.inplanes = 64
        # 큰 이미지 입력을 빠르게 줄이기 위한 초기 7x7 convolution입니다.
        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ResNet-18 기준 block 개수는 [2, 2, 2, 2]입니다.
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # AdaptiveAvgPool을 쓰면 입력 image size가 조금 달라도 `[B, C, 1, 1]`로 정리됩니다.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._init_weights()

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """하나의 ResNet stage를 구성합니다.

        첫 block만 stride=2로 downsampling할 수 있고, 나머지 block은 같은 해상도를 유지합니다.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # main branch 출력과 shortcut 출력의 shape를 맞추기 위한 1x1 projection입니다.
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        # stage의 첫 block은 필요하면 해상도/channel을 바꿉니다.
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # 같은 stage 안의 나머지 block은 shape를 유지합니다.
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """CNN 학습에 적합한 Kaiming 초기화를 적용합니다."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """이미지 batch를 class logits로 변환합니다.

        Args:
            x: `[B, 3, H, W]` image tensor입니다.

        Returns:
            `[B, num_classes]` logits입니다. CrossEntropyLoss에 바로 넣을 수 있습니다.
        """
        # Stem: 큰 해상도를 먼저 줄이고 low-level edge/texture feature를 추출합니다.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 4개의 residual stage를 통과하며 channel은 늘고 spatial size는 줄어듭니다.
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # `[B, 512, h, w] -> [B, 512, 1, 1] -> [B, 512]`
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet18(num_classes: int, in_channels: int = 3) -> ResNet:
    """ResNet-18 모델을 생성합니다."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)


def resnet34(num_classes: int, in_channels: int = 3) -> ResNet:
    """ResNet-34 모델을 생성합니다."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, in_channels)


# 문자열 설정값을 실제 모델 생성 함수로 연결하는 registry입니다.
MODEL_FACTORY: dict[str, Callable[[int], ResNet]] = {
    "resnet18": resnet18,
    "resnet34": resnet34,
}


def build_resnet(name: str, num_classes: int) -> ResNet:
    """YAML 설정의 모델 이름으로 ResNet 객체를 생성합니다."""
    try:
        return MODEL_FACTORY[name](num_classes)
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_FACTORY))
        raise ValueError(f"Unknown ResNet model '{name}'. Choose one of: {choices}") from exc
