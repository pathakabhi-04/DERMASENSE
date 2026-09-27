"""ViT-B/16 backbones for docs/academic_vit_spec.md.

- ViTClassifier: torchvision vit_b_16 (IMAGENET1K_V1), CLS token -> Linear(768, 8).
- ViTEncoder: the same ViT at 512x512 (position embeddings interpolated from
  14x14 to 32x32), followed by a simple feature pyramid (Li et al. 2022,
  ViTDet) that turns the single stride-16 map into the five maps ResNet-50
  gives the U-Net decoder: strides 2/4/8/16/32 with 64/256/512/1024/2048
  channels. The decoder itself is unchanged (src/academic/model.py).
- forward_with_attention(): re-runs the encoder blocks by hand to return
  each block's head-averaged attention matrix (torchvision calls
  MultiheadAttention with need_weights=False). tests: must reproduce the
  model's own forward exactly (see scripts/academic_vit/preflight.py).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models.vision_transformer import VisionTransformer, interpolate_embeddings

HIDDEN = 768
PATCH = 16


def _vit(image_size: int, pretrained: bool) -> VisionTransformer:
    if image_size == 224:
        return vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
    net = VisionTransformer(image_size=image_size, patch_size=PATCH, num_layers=12, num_heads=12,
                            hidden_dim=HIDDEN, mlp_dim=3072)
    if pretrained:
        state = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).state_dict()
        state = interpolate_embeddings(image_size=image_size, patch_size=PATCH, model_state=state)
        net.load_state_dict(state)
    return net


def tokens(vit: VisionTransformer, x: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] -> [B,1+N,768], the encoder output (after its final LayerNorm)."""
    t = vit._process_input(x)
    t = torch.cat([vit.class_token.expand(t.shape[0], -1, -1), t], dim=1)
    return vit.encoder(t)


def forward_with_attention(vit: VisionTransformer, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Same as tokens(), plus each block's head-averaged attention [B, 1+N, 1+N].

    Mirrors torchvision's Encoder.forward and EncoderBlock.forward line by line.
    """
    t = vit._process_input(x)
    t = torch.cat([vit.class_token.expand(t.shape[0], -1, -1), t], dim=1)
    enc = vit.encoder
    t = enc.dropout(t + enc.pos_embedding)
    attns = []
    for block in enc.layers:
        h = block.ln_1(t)
        h, w = block.self_attention(h, h, h, need_weights=True, average_attn_weights=True)
        attns.append(w)
        t = block.dropout(h) + t
        t = t + block.mlp(block.ln_2(t))
    return enc.ln(t), attns


def rollout(attns: list[torch.Tensor]) -> torch.Tensor:
    """Attention rollout (Abnar & Zuidema 2020): residual 0.5A + 0.5I, rows
    renormalised, multiplied through all blocks. Returns [B, 1+N, 1+N]."""
    eye = torch.eye(attns[0].shape[-1], device=attns[0].device)
    joint = None
    for a in attns:
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(dim=-1, keepdim=True)
        joint = a if joint is None else a @ joint
    return joint


class ViTClassifier(nn.Module):
    """CV-4 arm: CLS token -> Linear(768, 8), the torchvision head slot."""

    def __init__(self, num_classes: int = 8, pretrained: bool = True) -> None:
        super().__init__()
        self.vit = _vit(224, pretrained)
        self.vit.heads = nn.Sequential(nn.Linear(HIDDEN, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

    def param_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        head = list(self.vit.heads.parameters())
        head_ids = {id(p) for p in head}
        return [{"params": [p for p in self.parameters() if id(p) not in head_ids], "lr": backbone_lr},
                {"params": head, "lr": head_lr}]


def _up(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 2, stride=2), nn.BatchNorm2d(cout), nn.GELU())


def _proj(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ViTEncoder(nn.Module):
    """ViT-B/16 + simple feature pyramid, a drop-in for ResNet50Encoder."""

    feature_dim = HIDDEN

    def __init__(self, image_size: int = 512, pretrained: bool = True) -> None:
        super().__init__()
        self.vit = _vit(image_size, pretrained)
        self.vit.heads = nn.Identity()
        self.grid = image_size // PATCH
        self.to_s0 = nn.Sequential(_up(HIDDEN, 256), _up(256, 128), _up(128, 128), _proj(128, 64))    # /2
        self.to_s1 = nn.Sequential(_up(HIDDEN, 256), _up(256, 256), _proj(256, 256))                  # /4
        self.to_s2 = nn.Sequential(_up(HIDDEN, 512), _proj(512, 512))                                  # /8
        self.to_s3 = _proj(HIDDEN, 1024)                                                               # /16
        self.to_s4 = nn.Sequential(nn.MaxPool2d(2), _proj(HIDDEN, 2048))                               # /32

    def patch_map(self, t: torch.Tensor) -> torch.Tensor:
        """[B,1+N,768] tokens -> [B,768,g,g] (CLS dropped)."""
        b = t.shape[0]
        return t[:, 1:].transpose(1, 2).reshape(b, HIDDEN, self.grid, self.grid)

    def pyramid(self, m: torch.Tensor) -> list[torch.Tensor]:
        return [self.to_s0(m), self.to_s1(m), self.to_s2(m), self.to_s3(m), self.to_s4(m)]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.pyramid(self.patch_map(tokens(self.vit, x)))
