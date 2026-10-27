import subprocess
import sys
from pathlib import Path

import torch


def test_resnet50_classifies_grayscale_slices():
    from models.resnet50 import ResNet50

    model = ResNet50().eval()
    with torch.no_grad():
        output = model(torch.randn(2, 1, 64, 64))

    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()
    assert sum(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.modules()) == 53
    assert sum(p.numel() for p in model.parameters()) == 23_503_809


def test_original_resnet_stem_downsamples_256_input():
    from models.resnet50 import ResNet50

    model = ResNet50().to("meta").eval()
    sizes = {}
    hooks = [layer.register_forward_hook(lambda module, inputs, output, name=name: sizes.update({name: output.shape}))
             for name, layer in (("stem", model.stem), ("layer3", model.layer3), ("layer4", model.layer4))]
    with torch.no_grad():
        result = model(torch.empty(1, 1, 256, 256, device="meta"))
    for hook in hooks:
        hook.remove()
    assert result.shape == (1, 1)
    assert sizes["stem"] == (1, 64, 64, 64)
    assert sizes["layer3"] == (1, 1024, 16, 16)
    assert sizes["layer4"] == (1, 2048, 8, 8)


def test_main_prints_model_summary():
    script = Path(__file__).parents[1] / "models" / "resnet50.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "Total params:" in result.stdout


def test_gradcam_backward_hooks_work_for_layer3_and_layer4():
    from models.resnet50 import ResNet50

    model = ResNet50().eval()
    captured = {}
    hooks = []
    for name, layer in (("layer3", model.layer3[-1]), ("layer4", model.layer4[-1])):
        hooks.append(layer.register_forward_hook(lambda module, inputs, output, name=name: captured.update({name + "_activation": output.detach()})))
        hooks.append(layer.register_full_backward_hook(lambda module, grad_input, grad_output, name=name: captured.update({name + "_gradient": grad_output[0].detach()})))
    model(torch.randn(1, 1, 32, 32)).sum().backward()
    for hook in hooks:
        hook.remove()
    for name in ("layer3", "layer4"):
        assert captured[name + "_gradient"].shape == captured[name + "_activation"].shape
        assert torch.isfinite(captured[name + "_gradient"]).all()
