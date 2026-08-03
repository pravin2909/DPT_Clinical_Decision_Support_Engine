"""
═══════════════════════════════════════════════════════════════════════
  DPT Clinical Decision Support Engine
  MODEL EVALUATION SCRIPT — Kaggle Notebook

  PURPOSE : Evaluate all 4 models (MobileNet Router + 3 ViTs)
            and generate performance plots & metrics.

  HOW TO USE:
    1. Create a new Kaggle Notebook with GPU enabled
    2. Upload all 4 .pth files as a Dataset:
       - mobilenet_router.pth
       - vit_eye_disease.pth
       - vit_dental_disease.pth
       - vit_skin_disease.pth
    3. Update MODEL_DIR below to match the upload path
    4. Paste this entire script and run
    5. Download the saved PNG plots from /kaggle/working/plots/
═══════════════════════════════════════════════════════════════════════
"""

import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'])

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from collections import Counter
from tqdm import tqdm
import kagglehub
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_fscore_support, accuracy_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════
# CONFIGURATION — UPDATE MODEL_DIR
# ═══════════════════════════════════════
MODEL_DIR = '/kaggle/input/dpt-models'  # ← Path where you uploaded the 4 .pth files

MOBILENET_PATH = os.path.join(MODEL_DIR, 'mobilenet_router.pth')
EYE_VIT_PATH = os.path.join(MODEL_DIR, 'vit_eye_disease.pth')
DENTAL_VIT_PATH = os.path.join(MODEL_DIR, 'vit_dental_disease.pth')
SKIN_VIT_PATH = os.path.join(MODEL_DIR, 'vit_skin_disease.pth')

PLOTS_DIR = '/kaggle/working/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


# ═══════════════════════════════════════
# ViT ARCHITECTURE (must match training)
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
# MODEL LOADERS
# ═══════════════════════════════════════

def load_vit(path, num_classes):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = VisionTransformer(img_size=72, patch_size=6, num_classes=num_classes,
                              embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1)
    sd = ckpt['model_state_dict']
    remapped = {k.replace("shifted_patch_tokenization.", "patch_tokenization."): v for k, v in sd.items()}
    model.load_state_dict(remapped)
    model = model.to(DEVICE).eval()
    return model, ckpt.get('idx_to_class', None), ckpt.get('classes', None)


def load_mobilenet(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(1280, 512), nn.ReLU(),
        nn.Dropout(p=0.2), nn.Linear(512, ckpt['num_classes']),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE).eval()
    return model, ckpt.get('idx_to_class', None), ckpt.get('classes', None)


# ═══════════════════════════════════════
# DATASET PREPARATION
# ═══════════════════════════════════════

def find_all_images(root):
    imgs = []
    for dp, _, fns in os.walk(root):
        for f in fns:
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS:
                imgs.append(os.path.join(dp, f))
    return imgs


def prepare_mobilenet_dataset():
    """Download all 3 datasets and create the 3-class router dataset."""
    import shutil
    combined = '/kaggle/working/mobilenet_eval_data'
    datasets_info = {
        'eye':    'gunavenkatdoddi/eye-diseases-classification',
        'dental': 'salmansajid05/oral-diseases',
        'skin':   'ismailpromus/skin-diseases-image-dataset',
    }
    for domain, slug in datasets_info.items():
        path = kagglehub.dataset_download(slug)
        images = find_all_images(path)
        if len(images) > 4000:
            images = random.sample(images, 4000)
        dst = os.path.join(combined, domain)
        os.makedirs(dst, exist_ok=True)
        for i, img in enumerate(images):
            ext = os.path.splitext(img)[1]
            shutil.copy2(img, os.path.join(dst, f"{domain}_{i:05d}{ext}"))
    return combined


def find_image_folder(root):
    """Recursively find the first directory that contains class subdirectories with images."""
    for dp, dirs, _ in os.walk(root):
        has_images = False
        for d in dirs:
            sub = os.path.join(dp, d)
            if any(os.path.splitext(f)[1].lower() in IMG_EXTENSIONS for f in os.listdir(sub) if os.path.isfile(os.path.join(sub, f))):
                has_images = True
                break
        if has_images and len(dirs) > 1:
            return dp
    return root


# ═══════════════════════════════════════
# EVALUATION ENGINE
# ═══════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, loader, num_classes):
    """Run inference on all images, return labels, predictions, and probabilities."""
    all_labels = []
    all_preds = []
    all_probs = []

    for imgs, lbls in tqdm(loader, desc='   Evaluating', leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.extend(lbls.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ═══════════════════════════════════════
# PLOTTING FUNCTIONS
# ═══════════════════════════════════════

# Color palette
COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
          '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4']


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path, num_classes=None):
    """Beautiful confusion matrix heatmap."""
    labels = list(range(num_classes)) if num_classes else None
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    cm_pct = cm.astype('float') / row_sums * 100

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(class_names)*2), max(6, len(class_names)*0.8)))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=axes[0], cbar_kws={'shrink': 0.8})
    axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True', fontsize=12, fontweight='bold')
    axes[0].set_title('Counts', fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='both', labelsize=8)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')

    # Percentages
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Oranges', xticklabels=class_names,
                yticklabels=class_names, ax=axes[1], cbar_kws={'shrink': 0.8, 'format': '%.0f%%'})
    axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True', fontsize=12, fontweight='bold')
    axes[1].set_title('Percentages (%)', fontsize=13, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=8)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')

    fig.suptitle(f'Confusion Matrix — {model_name}', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"   ✓ Saved: {save_path}")


def plot_per_class_accuracy(y_true, y_pred, class_names, model_name, save_path, num_classes=None):
    """Bar chart showing per-class accuracy."""
    labels = list(range(num_classes)) if num_classes else None
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1)
    row_sums[row_sums == 0] = 1
    per_class_acc = cm.diagonal() / row_sums * 100
    overall_acc = accuracy_score(y_true, y_pred) * 100

    fig, ax = plt.subplots(figsize=(max(10, len(class_names)*1.2), 6))
    bars = ax.bar(range(len(class_names)), per_class_acc, color=COLORS[:len(class_names)],
                  edgecolor='white', linewidth=1.5, width=0.7)

    # Add value labels
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Overall accuracy line
    ax.axhline(y=overall_acc, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Overall Accuracy: {overall_acc:.2f}%')

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Class Accuracy — {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"   ✓ Saved: {save_path}")


def plot_roc_curves(y_true, y_probs, class_names, model_name, save_path, num_classes=None):
    """One-vs-Rest ROC curves with AUC for each class."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(10, 8))

    mean_auc_list = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        mean_auc_list.append(roc_auc)
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], linewidth=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.500)')

    mean_auc = np.mean(mean_auc_list)
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'ROC Curves — {model_name}\nMean AUC: {mean_auc:.3f}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"   ✓ Saved: {save_path}")


def plot_confidence_distribution(y_true, y_pred, y_probs, model_name, save_path, num_classes=None):
    """Histogram of prediction confidences, split by correct vs incorrect."""
    max_probs = y_probs.max(axis=1)
    correct = y_true == y_pred
    incorrect = ~correct

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 30)

    ax.hist(max_probs[correct], bins=bins, alpha=0.7, color='#2ecc71',
            label=f'Correct ({correct.sum()})', edgecolor='white')
    ax.hist(max_probs[incorrect], bins=bins, alpha=0.7, color='#e74c3c',
            label=f'Incorrect ({incorrect.sum()})', edgecolor='white')

    # Mean confidence lines
    if correct.sum() > 0:
        ax.axvline(max_probs[correct].mean(), color='#27ae60', linestyle='--', linewidth=2,
                   label=f'Mean correct: {max_probs[correct].mean():.2f}')
    if incorrect.sum() > 0:
        ax.axvline(max_probs[incorrect].mean(), color='#c0392b', linestyle='--', linewidth=2,
                   label=f'Mean incorrect: {max_probs[incorrect].mean():.2f}')

    ax.set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title(f'Confidence Distribution — {model_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"   ✓ Saved: {save_path}")


def plot_precision_recall_f1(y_true, y_pred, class_names, model_name, save_path, num_classes=None):
    """Grouped bar chart of Precision, Recall, F1 per class."""
    labels = list(range(num_classes)) if num_classes else None
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(class_names)*1.5), 6))
    bars1 = ax.bar(x - width, precision * 100, width, label='Precision', color='#3498db', edgecolor='white')
    bars2 = ax.bar(x, recall * 100, width, label='Recall', color='#e74c3c', edgecolor='white')
    bars3 = ax.bar(x + width, f1 * 100, width, label='F1-Score', color='#2ecc71', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Precision / Recall / F1-Score — {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"   ✓ Saved: {save_path}")


def print_classification_report(y_true, y_pred, class_names, model_name, num_classes=None):
    """Print sklearn classification report."""
    labels = list(range(num_classes)) if num_classes else None
    print(f"\n{'=' * 60}")
    print(f"  📋 Classification Report — {model_name}")
    print(f"{'=' * 60}")
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0))


# ═══════════════════════════════════════
# FULL EVALUATION FOR ONE MODEL
# ═══════════════════════════════════════

def evaluate_full(model, loader, num_classes, class_names, model_name, prefix):
    """Run all evaluations for a single model."""
    print(f"\n{'━' * 60}")
    print(f"  🔬 Evaluating: {model_name}")
    print(f"{'━' * 60}")

    y_true, y_pred, y_probs = evaluate_model(model, loader, num_classes)

    overall_acc = accuracy_score(y_true, y_pred) * 100
    print(f"  ✅ Overall Accuracy: {overall_acc:.2f}%")
    print(f"  📊 Total samples evaluated: {len(y_true)}")

    # Clean class names for display
    short_names = []
    for name in class_names:
        # Remove dataset numbering like "1. Eczema 1677" → "Eczema"
        clean = name
        if '. ' in clean:
            clean = clean.split('. ', 1)[1]
        # Remove trailing numbers/counts
        parts = clean.rsplit(' ', 1)
        if len(parts) == 2 and parts[1].replace('k', '').replace('.', '').isdigit():
            clean = parts[0]
        # Remove trailing " - XXXX" counts
        if ' - ' in clean:
            base = clean.rsplit(' - ', 1)
            if base[1].replace('k', '').replace('.', '').isdigit():
                clean = base[0]
        short_names.append(clean[:25])  # Cap length

    # Generate all plots
    print_classification_report(y_true, y_pred, short_names, model_name, num_classes)

    plot_confusion_matrix(y_true, y_pred, short_names, model_name,
                         os.path.join(PLOTS_DIR, f'{prefix}_confusion_matrix.png'), num_classes)

    plot_per_class_accuracy(y_true, y_pred, short_names, model_name,
                           os.path.join(PLOTS_DIR, f'{prefix}_per_class_accuracy.png'), num_classes)

    plot_roc_curves(y_true, y_probs, short_names, model_name,
                    os.path.join(PLOTS_DIR, f'{prefix}_roc_curves.png'), num_classes)

    plot_confidence_distribution(y_true, y_pred, y_probs, model_name,
                                os.path.join(PLOTS_DIR, f'{prefix}_confidence_dist.png'), num_classes)

    plot_precision_recall_f1(y_true, y_pred, short_names, model_name,
                            os.path.join(PLOTS_DIR, f'{prefix}_precision_recall_f1.png'), num_classes)

    return overall_acc


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    print("=" * 60)
    print("  📊 DPT MODEL EVALUATION SUITE")
    print("=" * 60)

    results = {}

    # ─────────────────────────────────
    # 1. SKIN DISEASE ViT
    # ─────────────────────────────────
    print("\n\n📦 Downloading Skin Disease dataset...")
    skin_path = kagglehub.dataset_download('ismailpromus/skin-diseases-image-dataset')
    skin_root = find_image_folder(skin_path)
    print(f"   Dataset root: {skin_root}")

    skin_transform = transforms.Compose([
        transforms.Resize((72, 72)), transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    skin_dataset = datasets.ImageFolder(skin_root, transform=skin_transform)
    skin_num = len(skin_dataset)
    num_val = int(skin_num * 0.2)
    _, skin_val = random_split(skin_dataset, [skin_num - num_val, num_val],
                               generator=torch.Generator().manual_seed(SEED))
    skin_loader = DataLoader(skin_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    skin_model, _, skin_classes = load_vit(SKIN_VIT_PATH, 10)
    results['Skin ViT'] = evaluate_full(
        skin_model, skin_loader, 10, skin_classes,
        'Skin Disease ViT (10 classes)', 'skin_vit'
    )
    del skin_model
    torch.cuda.empty_cache()

    # ─────────────────────────────────
    # 2. EYE DISEASE ViT
    # ─────────────────────────────────
    print("\n\n📦 Downloading Eye Disease dataset...")
    eye_path = kagglehub.dataset_download('gunavenkatdoddi/eye-diseases-classification')
    eye_root = find_image_folder(eye_path)
    print(f"   Dataset root: {eye_root}")

    eye_transform = transforms.Compose([
        transforms.Resize((72, 72)), transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    eye_dataset = datasets.ImageFolder(eye_root, transform=eye_transform)
    eye_num = len(eye_dataset)
    num_val = int(eye_num * 0.2)
    _, eye_val = random_split(eye_dataset, [eye_num - num_val, num_val],
                              generator=torch.Generator().manual_seed(SEED))
    eye_loader = DataLoader(eye_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    eye_model, _, eye_classes = load_vit(EYE_VIT_PATH, 4)
    results['Eye ViT'] = evaluate_full(
        eye_model, eye_loader, 4, eye_classes,
        'Eye Disease ViT (4 classes)', 'eye_vit'
    )
    del eye_model
    torch.cuda.empty_cache()

    # ─────────────────────────────────
    # 3. DENTAL DISEASE ViT
    # ─────────────────────────────────
    print("\n\n📦 Downloading Dental Disease dataset...")
    dental_path = kagglehub.dataset_download('salmansajid05/oral-diseases')
    dental_root = find_image_folder(dental_path)
    print(f"   Dataset root: {dental_root}")

    dental_transform = transforms.Compose([
        transforms.Resize((72, 72)), transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    dental_dataset = datasets.ImageFolder(dental_root, transform=dental_transform)
    dental_num = len(dental_dataset)
    num_val = int(dental_num * 0.2)
    _, dental_val = random_split(dental_dataset, [dental_num - num_val, num_val],
                                 generator=torch.Generator().manual_seed(SEED))
    dental_loader = DataLoader(dental_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    dental_model, _, dental_classes = load_vit(DENTAL_VIT_PATH, 7)
    results['Dental ViT'] = evaluate_full(
        dental_model, dental_loader, 7, dental_classes,
        'Dental Disease ViT (7 classes)', 'dental_vit'
    )
    del dental_model
    torch.cuda.empty_cache()

    # ─────────────────────────────────
    # 4. MOBILENET ROUTER
    # ─────────────────────────────────
    print("\n\n📦 Preparing MobileNet Router dataset...")
    mobilenet_data_dir = prepare_mobilenet_dataset()

    mn_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    mn_dataset = datasets.ImageFolder(mobilenet_data_dir, transform=mn_transform)
    mn_num = len(mn_dataset)
    num_val = int(mn_num * 0.2)
    _, mn_val = random_split(mn_dataset, [mn_num - num_val, num_val],
                             generator=torch.Generator().manual_seed(SEED))
    mn_loader = DataLoader(mn_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    mn_model, _, mn_classes = load_mobilenet(MOBILENET_PATH)
    results['MobileNet Router'] = evaluate_full(
        mn_model, mn_loader, 3, mn_classes,
        'MobileNet V2 Router (3 classes)', 'mobilenet_router'
    )
    del mn_model
    torch.cuda.empty_cache()

    # ─────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  🏆 FINAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<25} {'Accuracy':>10}")
    print("  " + "─" * 37)
    for model_name, acc in results.items():
        bar = '█' * int(acc / 5) + '░' * (20 - int(acc / 5))
        print(f"  {model_name:<25} {acc:>8.2f}%  {bar}")

    print(f"\n  📁 All plots saved to: {PLOTS_DIR}/")
    print(f"  📊 Total plots generated: {len(os.listdir(PLOTS_DIR))}")
    print("=" * 60)

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    accs = list(results.values())
    bars = ax.barh(names, accs, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'],
                   edgecolor='white', height=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('DPT Engine — Model Accuracy Comparison', fontsize=15, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'overall_comparison.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"   ✓ Saved: {PLOTS_DIR}/overall_comparison.png")


if __name__ == '__main__':
    main()
