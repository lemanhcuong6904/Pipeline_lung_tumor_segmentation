"""TorchVision ResNet-50 adapted to single-channel CT classification."""

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50PyTorch(nn.Module):
    """Return one logit per slice; pretrained weights initialize the backbone."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        rgb_conv = self.backbone.conv1
        grayscale_conv = nn.Conv2d(
            1, rgb_conv.out_channels, kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride, padding=rgb_conv.padding, bias=False,
        )
        if pretrained:
            # This equals the original RGB convolution on a repeated grayscale input.
            with torch.no_grad():
                grayscale_conv.weight.copy_(rgb_conv.weight.sum(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(grayscale_conv.weight, mode="fan_out", nonlinearity="relu")
        self.backbone.conv1 = grayscale_conv
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


if __name__ == "__main__":
    from torchsummary import summary

    summary(ResNet50PyTorch(), input_size=(1, 256, 256), device="cpu")
    # 23,503,809