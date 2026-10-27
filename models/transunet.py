"""Three-window, multi-encoder ResNet-50 / ViT TransUNet for 2D segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class MultiHeadAttention(nn.Module):
    """Reference-style attention with stable, correctly-scaled FP32 QK math."""

    def __init__(self, embedding_dim: int, head_num: int, dropout: float) -> None:
        super().__init__()
        if embedding_dim % head_num:
            raise ValueError("embedding_dim must be divisible by head_num")
        self.head_num = head_num
        self.head_dim = embedding_dim // head_num
        self.scale = self.head_dim**-0.5
        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            batch, tokens, _ = x.shape
            qkv = self.qkv_layer(x.float()).reshape(
                batch, tokens, 3, self.head_num, self.head_dim
            )
            query, key, value = qkv.permute(2, 0, 3, 1, 4)
            attention = self.dropout(
                ((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            )
            return self.out_attention(
                (attention @ value).transpose(1, 2).reshape(batch, tokens, -1)
            )


class MLP(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self, embedding_dim: int, head_num: int, mlp_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, head_num, dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """Bottleneck ViT with learned spatial embeddings and no CLS token."""

    def __init__(
        self,
        img_dim: int,
        in_channels: int,
        embedding_dim: int,
        head_num: int,
        mlp_dim: int,
        block_num: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.img_dim = img_dim
        self.projection = nn.Linear(in_channels, embedding_dim)
        self.embedding = nn.Parameter(torch.zeros(1, img_dim * img_dim, embedding_dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.layer_blocks = nn.ModuleList(
            TransformerEncoderBlock(embedding_dim, head_num, mlp_dim, dropout)
            for _ in range(block_num)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The entire token path is FP32 even when the outer training uses AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            batch, _, height, width = x.shape
            if (height, width) != (self.img_dim, self.img_dim):
                raise ValueError(
                    f"ViT expects {self.img_dim}x{self.img_dim}, got {height}x{width}"
                )
            tokens = self.dropout(
                self.projection(x.float().flatten(2).transpose(1, 2)) + self.embedding
            )
            for block in self.layer_blocks:
                tokens = block(tokens)
            return tokens.transpose(1, 2).reshape(batch, -1, height, width)


class DecoderBottleneck(nn.Module):
    """Bilinear upsample then reference-style two-convolution decoder block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.layer(torch.cat([skip, x], dim=1) if skip is not None else x)


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel gating on a fused feature map."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive")
        self.channels = channels
        hidden = max(1, channels // reduction)
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                "channel attention expects (batch, channels, height, width)"
            )
        with torch.autocast(device_type=features.device.type, enabled=False):
            pooled = features.float().mean(dim=(-2, -1))
            weights = torch.sigmoid(self.excitation(pooled))[:, :, None, None]
            return (features.float() * weights).to(features.dtype)


class CrossWindowAttention(nn.Module):
    """Let each window query spatial features from the other two windows.

    The input and output are (B, 3, C, H, W); attention uses a lower-dimensional
    token space and adds a learned residual to each original encoder feature.
    """

    def __init__(
        self,
        channels: int,
        spatial_size: int,
        embed_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if (
            channels < 1
            or spatial_size < 1
            or embed_dim < 1
            or heads < 1
            or embed_dim % heads
        ):
            raise ValueError(
                "cross-attention dimensions must be positive and embed_dim divisible by heads"
            )
        if not 0 <= dropout < 1:
            raise ValueError("cross-attention dropout must be in [0, 1)")
        self.channels, self.spatial_size = channels, spatial_size
        self.embed_dim, self.heads, self.dropout = embed_dim, heads, dropout
        self.input_projections = nn.ModuleList(
            nn.Linear(channels, embed_dim) for _ in range(3)
        )
        self.output_projections = nn.ModuleList(
            nn.Linear(embed_dim, channels) for _ in range(3)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.position = nn.Parameter(
            torch.zeros(1, 1, spatial_size * spatial_size, embed_dim)
        )
        self.window_embedding = nn.Parameter(torch.zeros(1, 3, 1, embed_dim))
        self.residual_scale = nn.Parameter(torch.full((3,), 0.1))
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.window_embedding, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if (
            features.ndim != 5
            or features.shape[1] != 3
            or features.shape[2] != self.channels
            or features.shape[-2:] != (self.spatial_size, self.spatial_size)
        ):
            raise ValueError(
                "cross-attention expects (batch, 3, channels, spatial_size, spatial_size)"
            )
        batch, _, _, height, width = features.shape
        count, dim, heads = height * width, self.embed_dim, self.heads
        # Use FP32 attention, as in the existing ViT, even under outer AMP.
        with torch.autocast(device_type=features.device.type, enabled=False):
            flattened = (
                features.float()
                .permute(0, 1, 3, 4, 2)
                .reshape(batch, 3, count, self.channels)
            )
            tokens = torch.stack(
                [
                    projection(flattened[:, index])
                    for index, projection in enumerate(self.input_projections)
                ],
                dim=1,
            )
            tokens = self.norm(tokens) + self.position + self.window_embedding
            # For wide, lung and mediastinal queries respectively, exclude the
            # querying branch from keys and values.
            others = torch.stack(
                (tokens[:, [1, 2]], tokens[:, [0, 2]], tokens[:, [0, 1]]), dim=1
            )
            queries = self.q_proj(tokens.reshape(batch * 3, count, dim))
            keys = self.k_proj(others.reshape(batch * 3, 2 * count, dim))
            values = self.v_proj(others.reshape(batch * 3, 2 * count, dim))
            head_dim = dim // heads
            queries = queries.reshape(batch * 3, count, heads, head_dim).transpose(1, 2)
            keys = keys.reshape(batch * 3, 2 * count, heads, head_dim).transpose(1, 2)
            values = values.reshape(batch * 3, 2 * count, heads, head_dim).transpose(
                1, 2
            )
            attended = (
                F.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    dropout_p=self.dropout if self.training else 0.0,
                )
                .transpose(1, 2)
                .reshape(batch, 3, count, dim)
            )
            delta = torch.stack(
                [
                    projection(attended[:, index])
                    for index, projection in enumerate(self.output_projections)
                ],
                dim=1,
            )
            delta = delta.reshape(batch, 3, height, width, self.channels).permute(
                0, 1, 4, 2, 3
            )
            result = features.float() + delta * self.residual_scale.view(1, 3, 1, 1, 1)
            return result.to(dtype=features.dtype)


def _resnet50_modules(in_channels: int, pretrained: bool) -> tuple[nn.Module, ...]:
    """Create ResNet-50 and adapt its first convolution to 1/3/5 CT slices."""
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    if in_channels != 3:
        original = backbone.conv1
        adapted = nn.Conv2d(
            in_channels,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False,
        )
        with torch.no_grad():
            adapted.weight.copy_(
                original.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            )
        backbone.conv1 = adapted
    return (
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
    )


class SingleWindowTransUNet(nn.Module):
    """Single-encoder TransUNet matching the 1-channel reference architecture."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        embed_dim: int = 768,
        transformer_depth: int = 8,
        transformer_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        img_dim: int = 256,
        patch_dim: int = 16,
        backbone_pretrained: bool = False,
        mlp_dim: int | None = None,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("SingleWindowTransUNet expects 1 input channel")
        if patch_dim != 16 or img_dim < 16 or img_dim % patch_dim:
            raise ValueError("img_dim must be divisible by patch_dim=16")
        if transformer_heads < 1 or embed_dim < 1 or embed_dim % transformer_heads:
            raise ValueError(
                "embed_dim must be positive and divisible by transformer_heads"
            )
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        conv1, bn1, relu, maxpool, layer1, layer2, layer3 = _resnet50_modules(
            1, backbone_pretrained
        )
        self.stem = nn.Sequential(conv1, bn1, relu)
        self.maxpool, self.encoder1, self.encoder2, self.encoder3 = (
            maxpool,
            layer1,
            layer2,
            layer3,
        )
        self.vit = ViT(
            img_dim // patch_dim,
            1024,
            embed_dim,
            transformer_heads,
            int(mlp_dim or embed_dim * mlp_ratio),
            transformer_depth,
            dropout,
        )
        self.from_vit = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = DecoderBottleneck(1024, base_channels * 2)
        self.decoder2 = DecoderBottleneck(base_channels * 2 + 256, base_channels)
        self.decoder3 = DecoderBottleneck(
            base_channels + 64, max(1, base_channels // 2)
        )
        self.decoder4 = DecoderBottleneck(
            max(1, base_channels // 2), max(1, base_channels // 8)
        )
        self.head = nn.Conv2d(max(1, base_channels // 8), 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        required_size = self.vit.img_dim * 16
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                "SingleWindowTransUNet expects input shaped (batch, 1, height, width)"
            )
        if x.shape[-2:] != (required_size, required_size):
            raise ValueError(
                f"SingleWindowTransUNet expects {required_size}x{required_size} input"
            )
        skip1 = self.stem(x)
        skip2 = self.encoder1(self.maxpool(skip1))
        skip3 = self.encoder2(skip2)
        encoded = self.encoder3(skip3)
        with torch.autocast(device_type=x.device.type, enabled=False):
            bottleneck = self.from_vit(self.vit(encoded))
        decoded = self.decoder1(bottleneck, skip3)
        decoded = self.decoder2(decoded, skip2)
        decoded = self.decoder3(decoded, skip1)
        return self.head(self.decoder4(decoded))


class TransUNet(nn.Module):
    """Three independent encoders; input channel order: wide, lung, mediastinal.

    Returns one channel of unnormalized logits with the input spatial size.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        embed_dim: int = 768,
        transformer_depth: int = 8,
        transformer_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        img_dim: int = 256,
        patch_dim: int = 16,
        backbone_pretrained: bool = False,
        mlp_dim: int | None = None,
        cross_attention_enabled: bool = True,
        cross_attention_dim: int = 128,
        cross_attention_heads: int = 4,
        cross_attention_dropout: float = 0.0,
        channel_attention_enabled: bool = False,
        channel_attention_reduction: int = 16,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("in_channels must be 3 (wide, lung, mediastinal)")
        if patch_dim != 16 or img_dim < 16 or img_dim % patch_dim:
            raise ValueError(
                "img_dim must be divisible by patch_dim=16 for the ResNet-50 /16 bottleneck"
            )
        if transformer_heads < 1 or embed_dim < 1 or embed_dim % transformer_heads:
            raise ValueError(
                "embed_dim must be positive and divisible by transformer_heads"
            )
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        self.encoders = nn.ModuleList()
        for _ in range(3):
            conv1, bn1, relu, maxpool, layer1, layer2, layer3 = _resnet50_modules(
                1, backbone_pretrained
            )
            self.encoders.append(
                nn.ModuleList(
                    (
                        nn.Sequential(conv1, bn1, relu),
                        nn.Sequential(maxpool, layer1),
                        layer2,
                        layer3,
                    )
                )
            )
        self.fusions = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels * 3, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for channels in (64, 256, 512, 1024)
        )
        self.cross_window_attention = (
            CrossWindowAttention(
                1024,
                img_dim // patch_dim,
                cross_attention_dim,
                cross_attention_heads,
                cross_attention_dropout,
            )
            if cross_attention_enabled
            else None
        )
        self.channel_attention = (
            ChannelAttention(1024, channel_attention_reduction)
            if channel_attention_enabled
            else None
        )
        self.vit = ViT(
            img_dim // patch_dim,
            1024,
            embed_dim,
            transformer_heads,
            int(mlp_dim or embed_dim * mlp_ratio),
            transformer_depth,
            dropout,
        )
        self.from_vit = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = DecoderBottleneck(1024, base_channels * 2)
        self.decoder2 = DecoderBottleneck(base_channels * 2 + 256, base_channels)
        self.decoder3 = DecoderBottleneck(
            base_channels + 64, max(1, base_channels // 2)
        )
        self.decoder4 = DecoderBottleneck(
            max(1, base_channels // 2), max(1, base_channels // 8)
        )
        self.head = nn.Conv2d(max(1, base_channels // 8), 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        required_size = self.vit.img_dim * 16
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("TransUNet expects input shaped (batch, 3, height, width)")
        if x.shape[-2:] != (required_size, required_size):
            raise ValueError(
                f"TransUNet expects {required_size}x{required_size} input, got {tuple(x.shape[-2:])}"
            )
        features = []
        for channel, encoder in enumerate(self.encoders):
            branch = x[:, channel : channel + 1]
            levels = []
            for stage in encoder:
                branch = stage(branch)
                levels.append(branch)
            features.append(levels)
        skips = [
            self.fusions[level](
                torch.cat([branch[level] for branch in features], dim=1)
            )
            for level in range(3)
        ]
        bottleneck_windows = torch.stack([branch[3] for branch in features], dim=1)
        if self.cross_window_attention is not None:
            bottleneck_windows = self.cross_window_attention(bottleneck_windows)
        encoded = self.fusions[3](torch.cat(bottleneck_windows.unbind(dim=1), dim=1))
        if self.channel_attention is not None:
            encoded = self.channel_attention(encoded)
        skip1, skip2, skip3 = skips
        with torch.autocast(device_type=x.device.type, enabled=False):
            bottleneck = self.from_vit(self.vit(encoded))
        decoded = self.decoder1(bottleneck, skip3)
        decoded = self.decoder2(decoded, skip2)
        decoded = self.decoder3(decoded, skip1)
        return self.head(self.decoder4(decoded))


if __name__ == "__main__":
    from torchsummary import summary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransUNet(backbone_pretrained=False).to(device).eval()
    with torch.no_grad():
        summary(model, input_size=(3, 256, 256), device=device)
    # torchsummary counts module weights, but omits direct parameters such as
    # ViT's learned positional embedding; report the complete total as well.
    print(f"Actual total parameters: {sum(p.numel() for p in model.parameters()):,}")
