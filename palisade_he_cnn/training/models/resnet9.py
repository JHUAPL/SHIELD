# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import torch
import torch.nn as nn


class Scale(nn.Module):
    def __init__(self, scale: float = 0.125):
        super().__init__()
        self.scale = scale

    def forward(self, x) -> torch.Tensor:
        return self.scale * x


def conv_block(
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str = "same",
        pool: bool = False,
        gelu: bool = False
):
    layers = [nn.Conv2d(in_ch,
                        out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding),
              nn.BatchNorm2d(out_ch)
              ]
    if pool:
        layers.append(nn.AvgPool2d(2, 2))

    if gelu:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self,
                 c_in: int = 4,
                 c_out: int = 36,
                 num_classes: int = 10,
                 scale_out: float = 0.125):
        super().__init__()
        self.c_out = c_out
        self.conv1 = nn.Conv2d(c_in,
                               c_out,
                               kernel_size=(3, 3),
                               padding="same",
                               bias=True)
        self.conv2 = conv_block(c_out,
                                64,
                                kernel_size=1,
                                padding="same",
                                pool=False,
                                gelu=True)
        self.conv3 = conv_block(64,
                                128,
                                kernel_size=3,
                                padding="same",
                                pool=True,
                                gelu=True)
        self.res1 = nn.Sequential(
            conv_block(128,
                       128,
                       kernel_size=3,
                       padding="same",
                       pool=False,
                       gelu=True),
            conv_block(128,
                       128,
                       kernel_size=3,
                       padding="same",
                       pool=False,
                       gelu=True)

        )
        self.conv4 = conv_block(128,
                                256,
                                kernel_size=3,
                                padding="same",
                                pool=True,
                                gelu=True)
        self.conv5 = conv_block(256,
                                512,
                                kernel_size=3,
                                padding="same",
                                pool=True,
                                gelu=True)
        self.res2 = nn.Sequential(
            conv_block(512,
                       512,
                       kernel_size=3,
                       padding="same",
                       pool=False,
                       gelu=True),
            conv_block(512,
                       512,
                       kernel_size=3,
                       padding="same",
                       pool=False,
                       gelu=True)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, num_classes, bias=True),
            Scale(scale_out)
        )

    def set_conv1_weights(self,
                          weights: torch.Tensor,
                          bias: torch.Tensor):
        self.conv1.weight.data = weights
        self.conv1.weight.requires_grad = False
        self.conv1.bias.data = bias
        self.conv1.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        res1 = self.res1(x)
        x = x + res1
        x = self.conv4(x)
        x = self.conv5(x)
        res2 = self.res2(x)
        x = x + res2
        return self.classifier(x)
