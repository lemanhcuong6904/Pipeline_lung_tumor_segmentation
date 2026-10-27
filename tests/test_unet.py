import subprocess
import sys
from pathlib import Path

import torch

from models.unet import DoubleConv, UNet


def test_double_conv_normalizes_each_convolution_output():
    """Each convolution needs its own BatchNorm before the activation."""
    block = DoubleConv(1, 8)

    assert sum(isinstance(layer, torch.nn.BatchNorm2d) for layer in block.step) == 2


def test_unet_returns_single_channel_mask_at_input_resolution():
    """A decoder that omits a skip connection or upsampling step changes this shape."""
    model = UNet()
    output = model(torch.randn(1, 1, 256, 256))

    assert output.shape == (1, 1, 256, 256)


def test_main_prints_torchsummary_parameter_count():
    """torchsummary must use the CPU when the instantiated model is on CPU."""
    script = Path(__file__).parents[1] / "models" / "unet.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Total params:" in result.stdout
