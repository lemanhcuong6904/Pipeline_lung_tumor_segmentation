"""UNet architecture used by the lung-tumor segmentation notebook."""

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Two normalized 3x3 convolution--ReLU operations preserving spatial size."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.step = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.step(x)


class UNet(nn.Module):
    """Four-level UNet for one-channel lung-tumor segmentation masks."""

    def __init__(self) -> None:
        super().__init__()

        self.layer1 = DoubleConv(1, 64)
        self.layer2 = DoubleConv(64, 128)
        self.layer3 = DoubleConv(128, 256)
        self.layer4 = DoubleConv(256, 512)

        self.layer5 = DoubleConv(512 + 256, 256)
        self.layer6 = DoubleConv(256 + 128, 128)
        self.layer7 = DoubleConv(128 + 64, 64)
        self.layer8 = nn.Conv2d(64, 1, kernel_size=1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(x)
        x2 = self.layer2(self.maxpool(x1))
        x3 = self.layer3(self.maxpool(x2))
        x4 = self.layer4(self.maxpool(x3))

        x5 = self.layer5(torch.cat([self.upsample(x4), x3], dim=1))
        x6 = self.layer6(torch.cat([self.upsample(x5), x2], dim=1))
        x7 = self.layer7(torch.cat([self.upsample(x6), x1], dim=1))
        return self.layer8(x7)


if __name__ == "__main__":
    from torchsummary import summary
    summary(UNet(), input_size=(1, 256, 256), device="cpu")
    # Total params: 7,781,761 + BatchNorm = 7,787,393
