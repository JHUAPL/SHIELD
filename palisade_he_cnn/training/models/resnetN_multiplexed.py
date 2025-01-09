# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

from pathlib import Path
from typing import Any, List, Optional, Type

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# Low-Complexity deep convolutional neural networks on
# FHE using multiplexed parallel convolutions
#
# https://eprint.iacr.org/2021/1688.pdf
#
# Our implementation is not an exact 1-to-1

__all__ = [
    "resnet_test",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
]

POOL = nn.AvgPool2d(2, 2)
BN_MOMENTUM = 0.1


class Debug(nn.Module):
    def __init__(self, filename="temp.txt", debug=False):
        super().__init__()
        self.filename = Path("debug") / filename
        self.debug = debug
        # print(self.debug, filename)

    def forward(self, x) -> torch.Tensor:
        if self.debug:
            data = x.detach().cpu().numpy().ravel()
            np.savetxt(self.filename, data, fmt="%0.04f")
        return x


class Scale(nn.Module):
    def __init__(self, scale: float = 0.125):
        super().__init__()
        self.scale = scale

    def forward(self, x) -> torch.Tensor:
        return self.scale * x


def conv_bn(inchan: int,
            outchan: int,
            kernel: int = 3,
            stride: int = 1,
            padding: str = "same",
            filenames: list = ["temp.txt"],
            debug: bool = False) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            inchan,
            outchan,
            kernel_size=kernel,
            stride=1,
            padding=padding
        ),
        nn.BatchNorm2d(outchan, momentum=BN_MOMENTUM),
        Debug(filenames[0], debug)
    )


def conv_bn_down(inchan: int,
                 outchan: int,
                 kernel: int = 3,
                 stride: int = 1,
                 padding: str = "same",
                 filenames: list = ["temp.txt", "temp.txt"],
                 debug: bool = False, ) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            inchan,
            outchan,
            kernel_size=kernel,
            stride=1,
            padding=padding
        ),
        nn.BatchNorm2d(outchan, momentum=BN_MOMENTUM),
        Debug(filenames[0], debug),
        POOL,
        Debug(filenames[1], debug)
    )


class BasicBlock(nn.Module):
    def __init__(
            self,
            inchan: int,
            outchan: int,
            kernel: int = 3,
            stride: int = 1,
            padding: str = "same",
            activation: nn.Module = nn.GELU(),
            downsample: Optional[nn.Module] = None,
            prefix: str = "l1",
            debug: bool = False
    ) -> None:
        super().__init__()

        # If a skip module is defined (defined as downsample), then our first block
        # in series needs to also include a downsampling operation, aka pooling.
        if downsample is not None:
            self.conv_bn_1 = conv_bn_down(
                inchan=inchan,
                outchan=outchan,
                kernel=3,
                stride=1,
                padding=padding,
                filenames=["%s_bn1.txt" % prefix,
                           "%s_pool.txt" % prefix],
                debug=debug
            )

        else:
            self.conv_bn_1 = conv_bn(
                inchan=outchan,
                outchan=outchan,
                kernel=3,
                stride=1,
                padding=padding,
                filenames=["%s_bn1.txt" % prefix],
                debug=debug
            )

        self.conv_bn_2 = conv_bn(
            inchan=outchan,
            outchan=outchan,
            kernel=3,
            stride=1,
            padding=padding,
            filenames=["%s_bn2.txt" % prefix],
            debug=debug
        )

        self.gelu = activation
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv_bn_1(x)
        out = self.gelu(out)

        out = self.conv_bn_2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.downsample is not None:
            out = self.gelu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[BasicBlock],
            layers: List[int],
            num_classes: int = 10,
            debug: bool = False
    ):
        super().__init__()
        self.debug = debug

        self.conv_bn_1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(16, momentum=BN_MOMENTUM),
            Debug("l0_bn1.txt", debug=debug)
        )
        self.gelu0 = nn.GELU()
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.gelu3 = nn.GELU()

        self.debug0 = Debug("l0_gelu.txt", debug)
        self.debug1 = Debug("l1_gelu.txt", debug)
        self.debug2 = Debug("l2_gelu.txt", debug)
        self.debug3 = Debug("l3_gelu.txt", debug)

        self.layer1 = self._make_layer(
            block=block,
            inchan=16,
            outchan=16,
            nblocks=layers[0],
            stride=1,
            prefix="l1"
        )
        self.layer2 = self._make_layer(
            block,
            inchan=16,
            outchan=32,
            nblocks=layers[1],
            stride=2,  # Triggers downsample != None, not a true stride
            prefix="l2"
        )
        self.layer3 = self._make_layer(
            block,
            inchan=32,
            outchan=32,
            nblocks=layers[2],
            stride=2,  # Triggers downsample != None, not a true stride
            prefix="l3"
        )
        self.classifier = nn.Sequential(
            POOL,
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, num_classes),
            Scale()
        )

    def forward(self, x: Tensor) -> Tensor:

        x = self.conv_bn_1(x)
        x = self.gelu0(x)
        x = self.debug0(x)

        x = self.layer1(x)
        x = self.gelu1(x)
        x = self.debug1(x)

        x = self.layer2(x)
        x = self.gelu2(x)
        x = self.debug2(x)

        x = self.layer3(x)
        x = self.gelu3(x)
        x = self.debug3(x)

        return self.classifier(x)

    def _make_layer(
            self,
            block: Type[BasicBlock],
            inchan: int,
            outchan: int,
            nblocks: int,
            stride: int = 1,
            prefix: str = "l1"
    ) -> nn.Sequential:

        downsample = None

        if stride != 1:
            downsample = conv_bn_down(
                inchan=inchan,
                outchan=outchan,
                filenames=["%s_ds_bn1.txt" % prefix,
                           "%s_ds_pool.txt" % prefix],
                debug=self.debug,
                kernel=3,
                stride=1,
                padding="same"
            )

        layers = []
        for i in range(0, nblocks):
            # Only need it for first iter
            if i == 1:
                downsample = None

            layers.append(
                block(
                    inchan=inchan,
                    outchan=outchan,
                    kernel=3,
                    stride=stride,
                    padding="same",
                    activation=nn.GELU(),
                    downsample=downsample,
                    prefix=prefix + "_%s" % str(i),
                    debug=self.debug
                )
            )

        return nn.Sequential(*layers)


def _resnet(
        block: Type[BasicBlock],
        layers: List[int],
        **kwargs: Any,
) -> ResNet:
    return ResNet(block, layers, **kwargs)


def resnet_test(**kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [1, 1, 1], **kwargs)


def resnet20(**kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [3, 3, 3], **kwargs)


def resnet32(**kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [5, 5, 5], **kwargs)


def resnet44(**kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [7, 7, 7], **kwargs)


def resnet56(**kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [9, 9, 9], **kwargs)


def resnet110(**kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [18, 18, 18], **kwargs)
