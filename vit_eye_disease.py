import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import kagglehub

# Dataset: Eye Diseases Classification by Guna Venkat Doddi
# Classes: normal, cataract, diabetic_retinopathy, glaucoma
# https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification

DATASET_SLUG = 'gunavenkatdoddi/eye-diseases-classification'

CLASSES = ['normal', 'cataract', 'diabetic_retinopathy', 'glaucoma']
NUM_CLASSES = len(CLASSES)


class Args:
    data_dir = 'data'
    batch_size = 32
    epochs = 20
    lr = 3e-4
    weight_decay = 1e-2
    num_workers = 2
    output = 'vit_eye_disease.pth'
    val_split = 0.2


def download_dataset(data_dir):
    path = kagglehub.dataset_download(DATASET_SLUG)
    print("Downloaded dataset to:", path)
    dataset_path = os.path.join(path, 'dataset')
    os.makedirs(data_dir, exist_ok=True)
    for cls in ['normal', 'cataract', 'diabetic_retinopathy', 'glaucoma']:
        src = os.path.join(dataset_path, cls)
        dst = os.path.join(data_dir, cls)
        if not os.path.exists(dst):
            os.system(f'cp -r "{src}" "{dst}"')
    return data_dir


def get_loaders(data_dir, batch_size, num_workers, val_split):
    full_transform = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=full_transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, dataset.classes


class ShiftedPatchTokenization(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.shift_size = patch_size // 2
        self.embed_dim = embed_dim
        self.proj = nn.Linear(5 * in_chans * patch_size ** 2, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.shift_size
        x_padded = F.pad(x, (p, p, p, p))

        orig = x_padded[:, :, p:-p, p:-p]
        left = x_padded[:, :, p:-p, 0:-2*p]
        right = x_padded[:, :, p:-p, 2*p:]
        up = x_padded[:, :, 0:-2*p, p:-p]
        down = x_padded[:, :, 2*p:, p:-p]

        x_cat = torch.cat([orig, left, right, up, down], dim=1)
        patches = F.unfold(x_cat, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.permute(0, 2, 1)
        patches = patches.view(B, self.num_patches, -1)
        tokens = self.proj(patches)
        spatial = patches.view(B, self.grid_size, self.grid_size, -1)

        return tokens, spatial


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
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
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
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn * self.temperature
        attn = attn - attn.mean(dim=-1, keepdim=True)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionLSA(embed_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_attn = self.attn(self.norm1(x))
        x = x + self.dropout1(x_attn)
        x_mlp = self.mlp(self.norm2(x))
        x = x + x_mlp
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=72, patch_size=6, in_channels=3, num_classes=4,
                 embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.num_patches = (img_size // patch_size) ** 2

        self.shifted_patch_tokenization = ShiftedPatchTokenization(
            img_size, patch_size, in_channels, embed_dim
        )
        self.patch_encoder = PatchEncoder(self.num_patches, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
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
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        tokens, _ = self.shifted_patch_tokenization(x)
        tokens = self.patch_encoder(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens)


def build_model(num_classes):
    return VisionTransformer(num_classes=num_classes)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc='Train')
    for imgs, lbls in pbar:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                out = model(imgs)
                loss = criterion(out, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=100.*correct/total)
    return running_loss/total, correct/total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc='Validate'):
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            loss = criterion(out, lbls)
            running_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
    return running_loss/total, correct/total


def main():
    args = Args()
    data_path = download_dataset(args.data_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, classes = get_loaders(data_path, args.batch_size, args.num_workers, args.val_split)
    print('Classes:', classes)

    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'classes': classes
            }, args.output)
            print(f"Saved best model (acc={best_acc:.4f}) to {args.output}")

if __name__ == '__main__':
    main()
