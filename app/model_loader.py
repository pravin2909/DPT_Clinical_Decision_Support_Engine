"""
Model definitions and loading utilities.
Contains the ViT architecture (same as training) and MobileNet loader.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io

from app.config import (
    VIT_IMG_SIZE, VIT_PATCH_SIZE, VIT_EMBED_DIM, VIT_DEPTH,
    VIT_NUM_HEADS, VIT_MLP_DIM, VIT_DROPOUT,
    MOBILENET_IMG_SIZE, NORMALIZE_MEAN, NORMALIZE_STD,
)


# ═══════════════════════════════════════
#  Vision Transformer Architecture
#  (must match the training architecture)
# ═══════════════════════════════════════

class ShiftedPatchTokenization(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.shift_size = patch_size // 2
        self.proj = nn.Linear(5 * in_chans * patch_size ** 2, embed_dim)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.shift_size
        x_padded = F.pad(x, (p, p, p, p))

        orig  = x_padded[:, :, p:-p,   p:-p]
        left  = x_padded[:, :, p:-p,   0:-2*p]
        right = x_padded[:, :, p:-p,   2*p:]
        up    = x_padded[:, :, 0:-2*p, p:-p]
        down  = x_padded[:, :, 2*p:,   p:-p]

        x_cat = torch.cat([orig, left, right, up, down], dim=1)
        patches = F.unfold(x_cat, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.permute(0, 2, 1).contiguous()
        tokens = self.proj(patches.view(B, self.num_patches, -1))
        return tokens


class PatchEncoder(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed


class MultiHeadAttentionLSA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale * self.temperature
        attn = attn - attn.mean(dim=-1, keepdim=True)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionLSA(embed_dim, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=72, patch_size=6, in_channels=3, num_classes=4,
                 embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_tokenization = ShiftedPatchTokenization(img_size, patch_size, in_channels, embed_dim)
        self.patch_encoder = PatchEncoder(self.num_patches, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * self.num_patches, 2048), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        tokens = self.patch_tokenization(x)
        tokens = self.patch_encoder(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.head(self.norm(tokens))


# ═══════════════════════════════════════
#  Image Preprocessing
# ═══════════════════════════════════════

def get_mobilenet_transform():
    """Preprocessing for MobileNet V2 router."""
    return transforms.Compose([
        transforms.Resize((MOBILENET_IMG_SIZE, MOBILENET_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


def get_vit_transform():
    """Preprocessing for ViT classifiers."""
    return transforms.Compose([
        transforms.Resize((VIT_IMG_SIZE, VIT_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


# ═══════════════════════════════════════
#  Model Loaders
# ═══════════════════════════════════════

def load_mobilenet_router(model_path: str, device: torch.device) -> tuple:
    """Load the MobileNet V2 router model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, checkpoint['num_classes']),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint['idx_to_class'], checkpoint['classes']


def load_vit_classifier(model_path: str, num_classes: int, device: torch.device) -> tuple:
    """Load a domain-specific ViT classifier."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = VisionTransformer(
        img_size=VIT_IMG_SIZE, patch_size=VIT_PATCH_SIZE,
        num_classes=num_classes, embed_dim=VIT_EMBED_DIM,
        depth=VIT_DEPTH, num_heads=VIT_NUM_HEADS,
        mlp_dim=VIT_MLP_DIM, dropout=VIT_DROPOUT,
    )

    # Handle naming differences between training scripts:
    #   Eye ViT (original)     → "shifted_patch_tokenization.*"
    #   Dental/Skin ViT (new)  → "patch_tokenization.*"
    # Remap keys to match our model definition ("patch_tokenization")
    state_dict = checkpoint['model_state_dict']
    remapped = {}
    for key, value in state_dict.items():
        new_key = key.replace("shifted_patch_tokenization.", "patch_tokenization.")
        remapped[new_key] = value

    model.load_state_dict(remapped)
    model = model.to(device)
    model.eval()

    idx_to_class = checkpoint.get('idx_to_class', None)
    classes = checkpoint.get('classes', None)

    return model, idx_to_class, classes


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load a PIL Image from raw bytes."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
