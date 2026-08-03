"""
═══════════════════════════════════════════════════════════════════════
  DPT Clinical Decision Support Engine
  MobileNet V2 Image Router — Kaggle Notebook Training Script

  PURPOSE : Classify an input medical image as one of 3 domains:
            eye | dental | skin
            Then route it to the correct domain-specific ViT.

  DATASETS:
    Eye    → gunavenkatdoddi/eye-diseases-classification
    Dental → salmansajid05/oral-diseases
    Skin   → ismailpromus/skin-diseases-image-dataset

  OUTPUT : mobilenet_router.pth

  HOW TO USE:
    1. Create a new Kaggle Notebook
    2. Turn on GPU (Settings → Accelerator → GPU T4 x2 or P100)
    3. Paste this entire script into a single code cell
    4. Click Run — it will download all 3 datasets, build a combined
       3-class dataset, train MobileNet V2, and save the model
    5. Download mobilenet_router.pth from the Output tab
═══════════════════════════════════════════════════════════════════════
"""

# ──────────────────────────────────────
# 0. Install dependencies
# ──────────────────────────────────────
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'])

import os
import random
import shutil
import numpy as np
from tqdm import tqdm
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
import kagglehub
from collections import Counter

# ──────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────
DATASETS = {
    'eye':    'gunavenkatdoddi/eye-diseases-classification',
    'dental': 'salmansajid05/oral-diseases',
    'skin':   'ismailpromus/skin-diseases-image-dataset',
}

class Config:
    seed = 42
    combined_dir = '/kaggle/working/mobilenet_data'   # Final 3-class dataset
    max_images_per_class = 4000       # Balance: sample up to N images per domain
    batch_size = 64
    epochs = 15
    lr = 1e-3                         # Higher LR for pretrained fine-tuning
    min_lr = 1e-6
    weight_decay = 1e-4
    warmup_epochs = 2
    num_workers = 4
    pin_memory = True
    output = '/kaggle/working/mobilenet_router.pth'
    val_split = 0.2
    label_smoothing = 0.05
    early_stop_patience = 6
    img_size = 224                    # MobileNet V2 standard input size
    num_classes = 3                   # eye, dental, skin
    freeze_backbone_epochs = 3        # Freeze backbone for first N epochs

cfg = Config()

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

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
# 3. Dataset Download & Custom Dataset
# ──────────────────────────────────────
def find_all_images(root_dir):
    """Recursively find all image files under a directory."""
    images = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                images.append(os.path.join(dirpath, fname))
    return images


def download_and_prepare():
    """
    Download all 3 datasets, collect images from each,
    sample up to max_images_per_class, and organize into:
      mobilenet_data/
        eye/      ← sampled eye images
        dental/   ← sampled dental images
        skin/     ← sampled skin images
    """
    print("=" * 60)
    print("  📦 STEP 1: DOWNLOAD & PREPARE CUSTOM 3-CLASS DATASET")
    print("=" * 60)

    os.makedirs(cfg.combined_dir, exist_ok=True)

    for domain, slug in DATASETS.items():
        print(f"\n{'─' * 50}")
        print(f"  Downloading: {domain.upper()} → {slug}")
        print(f"{'─' * 50}")

        # Download
        path = kagglehub.dataset_download(slug)
        print(f"  ✓ Downloaded to: {path}")

        # Find all images
        all_images = find_all_images(path)
        print(f"  ✓ Found {len(all_images)} total images")

        # Sample (balanced)
        if len(all_images) > cfg.max_images_per_class:
            sampled = random.sample(all_images, cfg.max_images_per_class)
            print(f"  ✓ Sampled {cfg.max_images_per_class} images (from {len(all_images)})")
        else:
            sampled = all_images
            print(f"  ✓ Using all {len(sampled)} images (below max of {cfg.max_images_per_class})")

        # Copy to combined directory
        dst_dir = os.path.join(cfg.combined_dir, domain)
        os.makedirs(dst_dir, exist_ok=True)

        copied = 0
        for img_path in sampled:
            fname = f"{domain}_{copied:05d}{os.path.splitext(img_path)[1]}"
            dst_path = os.path.join(dst_dir, fname)
            if not os.path.exists(dst_path):
                shutil.copy2(img_path, dst_path)
            copied += 1

        print(f"  ✓ Copied {copied} images → {dst_dir}")

    # Print final dataset summary
    print(f"\n{'=' * 60}")
    print(f"  📊 CUSTOM DATASET SUMMARY")
    print(f"{'=' * 60}")
    total = 0
    for cls_folder in sorted(os.listdir(cfg.combined_dir)):
        cls_path = os.path.join(cfg.combined_dir, cls_folder)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path)
                        if os.path.isfile(os.path.join(cls_path, f))])
            print(f"  {cls_folder:>8}: {count:>5} images")
            total += count
    print(f"  {'TOTAL':>8}: {total:>5} images")
    print(f"{'=' * 60}\n")

    return cfg.combined_dir


# ──────────────────────────────────────
# 4. Data Loaders
# ──────────────────────────────────────
def get_loaders():
    """Create train/val loaders with proper augmentation for MobileNet."""
    train_transform = transforms.Compose([
        transforms.Resize((cfg.img_size + 32, cfg.img_size + 32)),
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Get indices for split
    full_dataset = datasets.ImageFolder(cfg.combined_dir)
    num_val = int(len(full_dataset) * cfg.val_split)
    num_train = len(full_dataset) - num_val

    generator = torch.Generator().manual_seed(cfg.seed)
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [num_train, num_val], generator=generator
    )

    # Separate datasets with different transforms
    train_dataset = datasets.ImageFolder(cfg.combined_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(cfg.combined_dir, transform=val_transform)

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

    # Print distribution
    train_labels = [full_dataset.targets[i] for i in train_indices.indices]
    dist = Counter(train_labels)
    print("📊 Train class distribution:")
    for cls_name, cls_idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        print(f"   [{cls_idx}] {cls_name}: {dist.get(cls_idx, 0)} images")
    print()

    return train_loader, val_loader, classes, class_to_idx


# ──────────────────────────────────────
# 5. MobileNet V2 Model
# ──────────────────────────────────────
def build_mobilenet(num_classes, pretrained=True):
    """
    Build MobileNet V2 with pretrained ImageNet weights.
    Replace the final classifier for 3-class output.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)

    # Replace classifier head
    # Original: Sequential(Dropout(0.2), Linear(1280, 1000))
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )

    return model


def freeze_backbone(model):
    """Freeze all feature layers, only train classifier."""
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_backbone(model):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.features.parameters():
        param.requires_grad = True


# ──────────────────────────────────────
# 6. LR Scheduler
# ──────────────────────────────────────
class WarmupCosineScheduler:
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
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ──────────────────────────────────────
# 7. Training & Validation
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

    for imgs, lbls in tqdm(loader, desc='  Val  ', leave=False):
        imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
        out = model(imgs)
        loss = criterion(out, lbls)
        running_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == lbls).sum().item()
        total += lbls.size(0)

    return running_loss / total, correct / total


# ──────────────────────────────────────
# 8. Main
# ──────────────────────────────────────
def main():
    print("=" * 60)
    print("  🧭 DPT — MOBILENET V2 IMAGE ROUTER TRAINING")
    print("=" * 60)

    # Step 1: Download & prepare custom 3-class dataset
    download_and_prepare()

    # Step 2: Create data loaders
    train_loader, val_loader, classes, class_to_idx = get_loaders()
    num_classes = len(classes)

    print(f"🏷️  Router Classes ({num_classes}):")
    for name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        print(f"   {idx}: {name}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n⚡ Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Model
    model = build_mobilenet(num_classes, pretrained=True).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 MobileNet V2: {total_params:,} total params")

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # ── Phase 1: Freeze backbone, train classifier only ──
    print(f"\n🔒 Phase 1: Frozen backbone (epochs 1-{cfg.freeze_backbone_epochs})")
    freeze_backbone(model)
    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Training {trainable_p1:,} params (classifier only)")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = WarmupCosineScheduler(optimizer, 1, cfg.freeze_backbone_epochs, cfg.min_lr, cfg.lr)

    best_acc = 0.0
    patience_counter = 0

    print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10} {'Status':>8}")
    print("─" * 72)

    for epoch in range(cfg.freeze_backbone_epochs):
        lr = scheduler.step(epoch)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        status = ""
        if val_acc > best_acc:
            best_acc = val_acc
            status = "✓ BEST"
            _save_checkpoint(model, optimizer, epoch + 1, best_acc, classes, class_to_idx)

        print(f"{epoch+1:>4}/{cfg.epochs}  {train_loss:>10.4f}  {100*train_acc:>9.2f}%  {val_loss:>9.4f}  {100*val_acc:>8.2f}%  {lr:>9.2e}  {status}")

    # ── Phase 2: Unfreeze backbone, full fine-tuning ──
    remaining_epochs = cfg.epochs - cfg.freeze_backbone_epochs
    print(f"\n🔓 Phase 2: Full fine-tuning (epochs {cfg.freeze_backbone_epochs+1}-{cfg.epochs})")
    unfreeze_backbone(model)
    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Training {trainable_p2:,} params (all layers)")

    # New optimizer with lower LR for backbone
    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': cfg.lr * 0.1},   # backbone: low LR
        {'params': model.classifier.parameters(), 'lr': cfg.lr * 0.5}, # head: moderate LR
    ], weight_decay=cfg.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, 1, remaining_epochs, cfg.min_lr, cfg.lr * 0.5)

    print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10} {'Status':>8}")
    print("─" * 72)

    for epoch in range(remaining_epochs):
        lr = scheduler.step(epoch)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        status = ""
        global_epoch = cfg.freeze_backbone_epochs + epoch + 1
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            status = "✓ BEST"
            _save_checkpoint(model, optimizer, global_epoch, best_acc, classes, class_to_idx)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"\n⏹️  Early stopping (no improvement for {cfg.early_stop_patience} epochs)")
                break

        print(f"{global_epoch:>4}/{cfg.epochs}  {train_loss:>10.4f}  {100*train_acc:>9.2f}%  {val_loss:>9.4f}  {100*val_acc:>8.2f}%  {lr:>9.2e}  {status}")

    # Final summary
    print("\n" + "=" * 60)
    print(f"  ✅ MOBILENET ROUTER TRAINING COMPLETE")
    print(f"  Best Validation Accuracy: {100*best_acc:.2f}%")
    print(f"  Model saved to: {cfg.output}")
    ckpt = torch.load(cfg.output, weights_only=False)
    print(f"\n  📋 Router Labels:")
    for idx, name in sorted(ckpt['idx_to_class'].items()):
        print(f"     {idx} → {name}")
    print(f"\n  🔀 Routing Logic:")
    print(f"     Input Image → MobileNet → predicted class →")
    print(f"       eye    → Eye ViT    (4 classes)")
    print(f"       dental → Dental ViT (6 classes)")
    print(f"       skin   → Skin ViT   (10 classes)")
    print("=" * 60)


def _save_checkpoint(model, optimizer, epoch, best_acc, classes, class_to_idx):
    """Save checkpoint with all metadata needed for inference."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_acc,
        'classes': classes,
        'class_to_idx': class_to_idx,
        'idx_to_class': {v: k for k, v in class_to_idx.items()},
        'num_classes': len(classes),
        'model_config': {
            'architecture': 'mobilenet_v2',
            'img_size': cfg.img_size,
            'pretrained': True,
            'classifier': [1280, 512, len(classes)],
        },
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
        'routing_map': {
            'eye': {
                'vit_model': 'vit_eye_disease.pth',
                'num_classes': 4,
                'classes': ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal'],
            },
            'dental': {
                'vit_model': 'vit_dental_disease.pth',
                'num_classes': 6,
                'classes': ['Calculus', 'Caries', 'Gingivitis', 'Hypodontia',
                           'Tooth Discoloration', 'Ulcers'],
            },
            'skin': {
                'vit_model': 'vit_skin_disease.pth',
                'num_classes': 10,
                'classes': ['Eczema', 'Melanoma', 'Atopic Dermatitis', 'BCC',
                           'Melanocytic Nevi', 'Benign Keratosis', 'Psoriasis/Lichen Planus',
                           'Seborrheic Keratoses', 'Tinea/Ringworm', 'Warts/Molluscum'],
            },
        },
        'domain': 'router',
    }
    torch.save(checkpoint, cfg.output)


if __name__ == '__main__':
    main()
