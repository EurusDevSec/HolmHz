import torch
from holmhz.data import create_dataloader
from holmhz.detectors import EfficientNetDetector

# 1. Tạo DataLoader (từ Task 1.3)
# Note: create_dataloader dùng is_training để control shuffle + augmentation
val_loader = create_dataloader(
    manifest_path="data/manifests/val.json",
    batch_size=4,
    is_training=False,   # No shuffle, no augmentation (val mode)
    num_workers=0,       # Windows safe
)

# 2. Tạo Model (Task 1.4)
model = EfficientNetDetector(pretrained=False, freeze_backbone=True)
model.eval()

# 3. Lấy 1 batch
batch = next(iter(val_loader))
images = batch["image"]   # [4, 3, 224, 224]
labels = batch["label"]   # [4]

print(f"Images shape: {images.shape}")  # [4, 3, 224, 224]
print(f"Labels: {labels.tolist()}")      # [0.0, 1.0, ...]

# 4. Forward pass
with torch.no_grad():
    logits = model(images)  # [4, 1]
    probs = torch.sigmoid(logits)

print(f"Logits shape: {logits.shape}")     # [4, 1]
print(f"Logits: {logits.squeeze().tolist()}")
print(f"Probs: {probs.squeeze().tolist()}")   # [0.xx, ...]

# 5. Simulate loss (BCEWithLogitsLoss — Task 1.5 sẽ dùng chính thức)
loss_fn = torch.nn.BCEWithLogitsLoss()
loss = loss_fn(logits.squeeze(), labels)
print(f"Loss: {loss.item():.4f}")  # Some number > 0
print("✅ Integration test passed!")