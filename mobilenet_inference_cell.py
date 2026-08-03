"""
═══════════════════════════════════════════════════════════
  MobileNet Router — Inference Cell (paste after training)
  Upload an image → See which domain it's classified as
═══════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io
import ipywidgets as widgets
from IPython.display import display, clear_output

# ──────────────────────────────────────
# 1. Load Model
# ──────────────────────────────────────
MODEL_PATH = '/kaggle/working/mobilenet_router.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load checkpoint
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
idx_to_class = ckpt['idx_to_class']
num_classes = ckpt['num_classes']
norm_mean = ckpt['normalization']['mean']
norm_std = ckpt['normalization']['std']
img_size = ckpt['model_config']['img_size']

# Build model
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1280, 512),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(512, num_classes),
)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✅ Model loaded from {MODEL_PATH}")
print(f"   Classes: {idx_to_class}")
print(f"   Best val acc: {100*ckpt['best_val_acc']:.2f}%\n")

# ──────────────────────────────────────
# 2. Preprocessing
# ──────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# ──────────────────────────────────────
# 3. Inference Function
# ──────────────────────────────────────
DOMAIN_INFO = {
    'eye':    {'emoji': '👁️',  'color': '#3498db', 'routes_to': 'Eye ViT (4 classes)'},
    'dental': {'emoji': '🦷', 'color': '#e74c3c', 'routes_to': 'Dental ViT (6 classes)'},
    'skin':   {'emoji': '🔬', 'color': '#2ecc71', 'routes_to': 'Skin ViT (10 classes)'},
}

def predict_image(image_bytes):
    """Run inference on an uploaded image."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    inp = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)[0]

    # Get results
    pred_idx = probs.argmax().item()
    pred_class = idx_to_class[pred_idx]
    pred_prob = probs[pred_idx].item()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1.3]})

    # Left: Show image
    axes[0].imshow(img)
    axes[0].set_title('Uploaded Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Right: Confidence bars
    class_names = [idx_to_class[i] for i in range(num_classes)]
    probabilities = [probs[i].item() * 100 for i in range(num_classes)]
    colors = [DOMAIN_INFO.get(name, {}).get('color', '#95a5a6') for name in class_names]

    bars = axes[1].barh(class_names, probabilities, color=colors, edgecolor='white', height=0.6)
    axes[1].set_xlim(0, 105)
    axes[1].set_xlabel('Confidence (%)', fontsize=12)
    axes[1].set_title('Classification Result', fontsize=14, fontweight='bold')

    # Add percentage labels on bars
    for bar, prob in zip(bars, probabilities):
        axes[1].text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                    f'{prob:.1f}%', va='center', fontsize=12, fontweight='bold')

    # Highlight predicted class
    info = DOMAIN_INFO.get(pred_class, {'emoji': '❓', 'routes_to': 'Unknown'})
    fig.suptitle(
        f"{info.get('emoji', '')}  Predicted: {pred_class.upper()}  ({pred_prob*100:.1f}% confidence)\n"
        f"→ Routes to: {info.get('routes_to', 'N/A')}",
        fontsize=16, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    plt.show()

    # Print routing info
    print(f"\n{'─' * 50}")
    print(f"  📌 PREDICTION: {pred_class.upper()}")
    print(f"  📊 CONFIDENCE: {pred_prob*100:.1f}%")
    print(f"  🔀 ROUTES TO:  {info.get('routes_to', 'N/A')}")
    print(f"{'─' * 50}\n")


# ──────────────────────────────────────
# 4. Upload Widget
# ──────────────────────────────────────
output = widgets.Output()

upload_btn = widgets.FileUpload(
    accept='image/*',
    multiple=False,
    description='📤 Upload Image',
    layout=widgets.Layout(width='250px', height='40px'),
)

title_label = widgets.HTML(
    value='<h2 style="color: #2c3e50; font-family: Arial;">🧭 MobileNet Router — Upload an Image to Classify</h2>'
    '<p style="color: #7f8c8d;">Supported: JPG, PNG, BMP, WebP | The model will predict: eye, dental, or skin</p>'
)

def on_upload(change):
    with output:
        clear_output(wait=True)
        for filename, file_info in upload_btn.value.items():
            print(f"📂 Processing: {filename}")
            predict_image(file_info['content'])

upload_btn.observe(on_upload, names='value')

# Display
display(title_label)
display(upload_btn)
display(output)

print("👆 Click the button above to upload a medical image for classification.\n")
