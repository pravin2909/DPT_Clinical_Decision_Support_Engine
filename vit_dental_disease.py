"""
═══════════════════════════════════════════════════════════════════════
  DPT Clinical Decision Support Engine
  Dental Disease ViT — Kaggle Notebook Training Script
  
  Dataset : salmansajid05/oral-diseases  (6 classes)
  Classes : Calculus, Caries, Gingivitis, Hypodontia,
            Tooth Discoloration, Ulcers
  Output  : vit_dental_disease.pth
  
  HOW TO USE:
    1. Create a new Kaggle Notebook
    2. Turn on GPU (Settings → Accelerator → GPU T4 x2 or P100)
    3. Paste this entire script into a single code cell
    4. Click Run — it will download data, train, and save the model
    5. Download vit_dental_disease.pth from the Output tab
═══════════════════════════════════════════════════════════════════════
"""

# ──────────────────────────────────────
# 0. Install dependencies (Kaggle-safe)
# ──────────────────────────────────────
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'])

import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import kagglehub
from collections import Counter

# ──────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────
DATASET_SLUG = 'salmansajid05/oral-diseases'

class Config:
    seed = 42
    data_dir = '/kaggle/working/dental_data'
    batch_size = 32
    epochs = 30
    lr = 3e-4
    min_lr = 1e-6
    weight_decay = 1e-2
    warmup_epochs = 3
    num_workers = 4
    pin_memory = True
    output = '/kaggle/working/vit_dental_disease.pth'
    val_split = 0.2
    label_smoothing = 0.1
    early_stop_patience = 8
    img_size = 72
    patch_size = 6
    embed_dim = 64
    depth = 4
    num_heads = 4
    mlp_dim = 128
    dropout = 0.1

cfg = Config()

# ──────────────────────────────────────
# 2. Reproducibility
# ──────────────────────────────────────
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(cfg.seed)

# ──────────────────────────────────────
# 3. Dataset Download & Preparation
# ──────────────────────────────────────
def download_dataset():
    """Download and organize into ImageFolder format."""
    path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"✓ Downloaded dataset to: {path}")

    # Find the root directory containing class folders
    dataset_root = None
    for root, dirs, files in os.walk(path):
        lower_dirs = [d.lower() for d in dirs]
        if any(kw in ' '.join(lower_dirs) for kw in ['caries', 'calculus', 'gingivitis']):
            dataset_root = root
            break

    if dataset_root is None:
        dataset_root = path
        print(f"⚠ Using fallback root: {dataset_root}")

    print(f"✓ Dataset root: {dataset_root}")
    print(f"  Folders: {os.listdir(dataset_root)}")

    # Copy into clean ImageFolder structure
    os.makedirs(cfg.data_dir, exist_ok=True)
    for cls_folder in os.listdir(dataset_root):
        src = os.path.join(dataset_root, cls_folder)
        if os.path.isdir(src):
            dst = os.path.join(cfg.data_dir, cls_folder)
            if not os.path.exists(dst):
                os.system(f'cp -r "{src}" "{dst}"')

    # Print stats
    print("\n📊 Dataset Statistics:")
    total = 0
    for cls_folder in sorted(os.listdir(cfg.data_dir)):
        cls_path = os.path.join(cfg.data_dir, cls_folder)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path)
                        if os.path.isfile(os.path.join(cls_path, f))])
            print(f"   {cls_folder}: {count} images")
            total += count
    print(f"   Total: {total} images\n")
    return cfg.data_dir


def get_loaders():
    """Create train/val loaders with proper augmentation."""
    train_transform = transforms.Compose([
        transforms.Resize((cfg.img_size + 12, cfg.img_size + 12)),
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(cfg.data_dir)
    num_val = int(len(full_dataset) * cfg.val_split)
    num_train = len(full_dataset) - num_val

    # Deterministic split
    generator = torch.Generator().manual_seed(cfg.seed)
    train_indices, val_indices = random_split(range(len(full_dataset)), [num_train, num_val], generator=generator)

    # Create separate datasets with different transforms
    train_dataset = datasets.ImageFolder(cfg.data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(cfg.data_dir, transform=val_transform)

    train_subset = Subset(train_dataset, train_indices.indices)
    val_subset = Subset(val_dataset, val_indices.indices)

    train_loader = DataLoader(
        train_subset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )

    classes = full_dataset.classes
    class_to_idx = full_dataset.class_to_idx

    # Print class distribution for train split
    train_labels = [full_dataset.targets[i] for i in train_indices.indices]
    dist = Counter(train_labels)
    print("📊 Train class distribution:")
    for cls_name, cls_idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        print(f"   [{cls_idx}] {cls_name}: {dist.get(cls_idx, 0)} images")

    return train_loader, val_loader, classes, class_to_idx


# ──────────────────────────────────────
# 4. Model Architecture
# ──────────────────────────────────────
class ShiftedPatchTokenization(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.shift_size = patch_size // 2
        self.proj = nn.Linear(5 * in_chans * patch_size ** 2, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.shift_size
        x_padded = F.pad(x, (p, p, p, p))

        orig  = x_padded[:, :, p:-p,    p:-p]
        left  = x_padded[:, :, p:-p,    0:-2*p]
        right = x_padded[:, :, p:-p,    2*p:]
        up    = x_padded[:, :, 0:-2*p,  p:-p]
        down  = x_padded[:, :, 2*p:,    p:-p]

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
        assert embed_dim % num_heads == 0
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
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=72, patch_size=6, in_channels=3, num_classes=6,
                 embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0
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
            nn.Linear(embed_dim * self.num_patches, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        tokens = self.patch_tokenization(x)
        tokens = self.patch_encoder(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.head(self.norm(tokens))


# ──────────────────────────────────────
# 5. Learning Rate Scheduler with Warmup
# ──────────────────────────────────────
class WarmupCosineScheduler:
    """Linear warmup then cosine annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ──────────────────────────────────────
# 6. Training & Validation Loops
# ──────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc='  Train', leave=False)
    for imgs, lbls in pbar:
        imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            out = model(imgs)
            loss = criterion(out, lbls)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == lbls).sum().item()
        total += lbls.size(0)
        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{100.*correct/total:.1f}%")

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, lbls in tqdm(loader, desc='  Val  ', leave=False):
        imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
        out = model(imgs)
        loss = criterion(out, lbls)
        running_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


# ──────────────────────────────────────
# 7. Main Training Loop
# ──────────────────────────────────────
def main():
    print("=" * 60)
    print("  🦷 DPT — DENTAL DISEASE ViT TRAINING")
    print("=" * 60)

    # Download & prepare
    download_dataset()
    train_loader, val_loader, classes, class_to_idx = get_loaders()
    num_classes = len(classes)

    print(f"\n🏷️  Classes ({num_classes}):")
    for name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        print(f"   {idx}: {name}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n⚡ Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Model
    model = VisionTransformer(
        img_size=cfg.img_size, patch_size=cfg.patch_size,
        num_classes=num_classes, embed_dim=cfg.embed_dim,
        depth=cfg.depth, num_heads=cfg.num_heads,
        mlp_dim=cfg.mlp_dim, dropout=cfg.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model: {total_params:,} params ({trainable_params:,} trainable)")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, cfg.warmup_epochs, cfg.epochs, cfg.min_lr, cfg.lr)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Training
    best_acc = 0.0
    patience_counter = 0

    print(f"\n🚀 Training for {cfg.epochs} epochs (patience={cfg.early_stop_patience})\n")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10} {'Status':>8}")
    print("─" * 72)

    for epoch in range(cfg.epochs):
        lr = scheduler.step(epoch)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        # Check improvement
        status = ""
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            status = "✓ BEST"

            # Save checkpoint with all metadata
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_acc,
                'classes': classes,
                'class_to_idx': class_to_idx,
                'idx_to_class': {v: k for k, v in class_to_idx.items()},
                'num_classes': num_classes,
                'model_config': {
                    'img_size': cfg.img_size,
                    'patch_size': cfg.patch_size,
                    'embed_dim': cfg.embed_dim,
                    'depth': cfg.depth,
                    'num_heads': cfg.num_heads,
                    'mlp_dim': cfg.mlp_dim,
                    'dropout': cfg.dropout,
                },
                'normalization': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                },
                'domain': 'dental',
            }
            torch.save(checkpoint, cfg.output)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch + 1} (no improvement for {cfg.early_stop_patience} epochs)")
                break

        print(f"{epoch+1:>4}/{cfg.epochs}  {train_loss:>10.4f}  {100*train_acc:>9.2f}%  {val_loss:>9.4f}  {100*val_acc:>8.2f}%  {lr:>9.2e}  {status}")

    # Final summary
    print("\n" + "=" * 60)
    print(f"  ✅ TRAINING COMPLETE")
    print(f"  Best Validation Accuracy: {100*best_acc:.2f}%")
    print(f"  Model saved to: {cfg.output}")
    print(f"\n  📋 Class Labels:")
    ckpt = torch.load(cfg.output, weights_only=False)
    for idx, name in sorted(ckpt['idx_to_class'].items()):
        print(f"     {idx}: {name}")
    print("=" * 60)


if __name__ == '__main__':
    main()
