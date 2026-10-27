import torch
from torchvision.models import ResNet50_Weights, resnet50


def test_pretrained_grayscale_conv_matches_repeated_rgb_input(monkeypatch):
    import models.resnet50_pytorch as module

    original = resnet50(weights=None)
    with torch.no_grad():
        original.conv1.weight[:, 0].fill_(0.1)
        original.conv1.weight[:, 1].fill_(0.2)
        original.conv1.weight[:, 2].fill_(0.3)
    called = []
    def build_backbone(*, weights):
        called.append(weights)
        return original
    monkeypatch.setattr(module, "resnet50", build_backbone)
    original_conv = original.conv1
    model = module.ResNet50PyTorch(pretrained=True).eval()
    images = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        actual = model.backbone.conv1(images)
        expected = original_conv(images.repeat(1, 3, 1, 1))
        output = model(images)
    assert called == [ResNet50_Weights.DEFAULT]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert output.shape == (2, 1)
    assert model.backbone.fc.out_features == 1


def test_unpretrained_model_accepts_single_channel_slices():
    from models.resnet50_pytorch import ResNet50PyTorch

    model = ResNet50PyTorch(pretrained=False).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 256, 256))
    assert model.backbone.conv1.in_channels == 1
    assert output.shape == (1, 1) and torch.isfinite(output).all()


def test_new_trainer_runs_its_own_loop_with_pretrained_model(tmp_path, monkeypatch):
    import trainers.train_binary_classification_pytorch as trainer
    from torch.utils.data import DataLoader, TensorDataset

    captured = []
    def fake_model(*, pretrained):
        captured.append(pretrained)
        return torch.nn.Linear(1, 1)
    loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.ones(2, 1)), batch_size=1)
    monkeypatch.setattr(trainer, "create_dataloader", lambda *args, **kwargs: loader)
    monkeypatch.setattr(trainer, "run_epoch", lambda *args, **kwargs: {
        "weighted_bce_loss": 1., "acc": .5, "precision": .5, "recall": .5, "f1": .5,
    })
    monkeypatch.setattr(trainer, "ResNet50PyTorch", fake_model)
    cfg = trainer.CFG | {"SAVE_ROOT": str(tmp_path), "DEVICE": "cpu", "AMP_DTYPE": "none",
                         "WANDB_ENABLED": False, "EPOCHS": 1, "TRAIN_BATCHES_PER_EPOCH": None}
    trainer.main(cfg)
    assert captured == [True]
    assert (tmp_path / cfg["EXP_NAME"] / "best.pt").is_file()
    monkeypatch.setattr(trainer, "load_checkpoint", lambda *args: {"model_state": torch.nn.Linear(1, 1).state_dict()})
    trainer.main(cfg | {"RESUME_PATH": "checkpoint.pt", "RESUME_WEIGHTS_ONLY": True})
    assert captured == [True, False]
