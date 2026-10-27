import pytest
import torch


@pytest.fixture(autouse=True)
def limit_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_each_window_drives_an_independent_encoder_and_output():
    from models.transunet import TransUNet

    torch.manual_seed(4)
    model = TransUNet(
        img_dim=32,
        embed_dim=32,
        transformer_depth=1,
        transformer_heads=4,
        base_channels=8,
        dropout=0,
    ).eval()
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    output = model(x)
    assert output.shape == (1, 1, 32, 32)
    output.square().mean().backward()
    assert all(x.grad[:, channel].abs().sum() > 0 for channel in range(3))
    weights = [branch[0][0].weight for branch in model.encoders]
    assert len({weight.data_ptr() for weight in weights}) == 3
    assert all(
        weight.grad is not None and weight.grad.abs().sum() > 0 for weight in weights
    )


def test_rejects_wrong_number_of_windows_and_wrong_image_size():
    from models.transunet import TransUNet

    model = TransUNet(
        img_dim=32,
        embed_dim=32,
        transformer_depth=1,
        transformer_heads=4,
        base_channels=8,
    ).eval()
    with pytest.raises(ValueError, match="3"):
        model(torch.zeros(1, 1, 32, 32))
    with pytest.raises(ValueError, match="32"):
        model(torch.zeros(1, 3, 48, 48))


def test_cross_attention_updates_each_window_using_the_other_windows():
    from models.transunet import CrossWindowAttention

    torch.manual_seed(4)
    module = CrossWindowAttention(
        channels=8, spatial_size=2, embed_dim=8, heads=2, dropout=0
    ).eval()
    features = torch.randn(1, 3, 8, 2, 2, requires_grad=True)
    output = module(features)
    assert output.shape == features.shape
    gradients = torch.autograd.grad(
        output[:, 0].square().sum(), features, retain_graph=True
    )[0]
    assert gradients[:, 1].abs().sum() > 0
    assert gradients[:, 2].abs().sum() > 0
    assert module.q_proj.weight.grad is None
    output.square().mean().backward()
    assert module.q_proj.weight.grad is not None
    assert module.k_proj.weight.grad is not None
    assert module.v_proj.weight.grad is not None


def test_transunet_cross_attention_is_configurable_and_updates_parameters():
    from models.transunet import TransUNet

    kwargs = {
        "img_dim": 32,
        "embed_dim": 32,
        "transformer_depth": 1,
        "transformer_heads": 4,
        "base_channels": 8,
        "dropout": 0,
        "cross_attention_dim": 16,
        "cross_attention_heads": 4,
    }
    model = TransUNet(**kwargs).eval()
    output = model(torch.randn(1, 3, 32, 32))
    output.square().mean().backward()
    assert output.shape == (1, 1, 32, 32)
    assert model.cross_window_attention.q_proj.weight.grad.abs().sum() > 0
    assert model.cross_window_attention.v_proj.weight.grad.abs().sum() > 0
    baseline = TransUNet(**kwargs, cross_attention_enabled=False)
    assert not any(
        key.startswith("cross_window_attention.") for key in baseline.state_dict()
    )


def test_rejects_invalid_cross_attention_dimensions():
    from models.transunet import CrossWindowAttention

    with pytest.raises(ValueError, match="divisible"):
        CrossWindowAttention(
            channels=8, spatial_size=2, embed_dim=7, heads=2, dropout=0
        )


def test_single_window_model_matches_original_architecture():
    from models.transunet import SingleWindowTransUNet

    model = SingleWindowTransUNet(
        img_dim=32,
        embed_dim=32,
        transformer_depth=1,
        transformer_heads=4,
        base_channels=8,
        dropout=0,
    ).eval()
    assert model.stem[0].in_channels == 1
    assert not hasattr(model, "fusions")
    output = model(torch.randn(1, 1, 32, 32))
    assert output.shape == (1, 1, 32, 32)
    output.square().mean().backward()
    assert model.stem[0].weight.grad is not None
    with pytest.raises(ValueError, match="1"):
        model(torch.randn(1, 3, 32, 32))


def test_channel_attention_reweights_fused_feature_channels():
    from models.transunet import ChannelAttention

    torch.manual_seed(7)
    module = ChannelAttention(channels=8, reduction=4)
    features = torch.randn(2, 8, 3, 3, requires_grad=True)
    output = module(features)
    assert output.shape == features.shape
    weights = output / features
    assert weights[0, :, 0, 0].std() > 0
    assert not torch.allclose(weights[0], weights[1])
    assert torch.allclose(weights[:, :, :1, :1].expand_as(weights), weights, atol=1e-5)
    output.square().mean().backward()
    assert module.excitation[2].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="reduction"):
        ChannelAttention(channels=8, reduction=0)


@pytest.mark.parametrize("cross_enabled", [False, True])
def test_channel_attention_applies_after_fusion_and_supports_cross(cross_enabled):
    from models.transunet import TransUNet

    torch.manual_seed(4)
    model = TransUNet(
        img_dim=32,
        embed_dim=32,
        transformer_depth=1,
        transformer_heads=4,
        base_channels=8,
        dropout=0,
        cross_attention_dim=16,
        cross_attention_enabled=cross_enabled,
        channel_attention_enabled=True,
    ).train()
    fused = []
    inputs = []
    fusion_hook = model.fusions[3].register_forward_hook(
        lambda _m, _a, out: fused.append(out)
    )
    attention_hook = model.channel_attention.register_forward_pre_hook(
        lambda _m, args: inputs.append(args[0])
    )
    output = model(torch.randn(2, 3, 32, 32))
    fusion_hook.remove()
    attention_hook.remove()
    assert inputs[0] is fused[0]
    assert output.shape == (2, 1, 32, 32)
    output.square().mean().backward()
    assert model.channel_attention.excitation[2].weight.grad.abs().sum() > 0
