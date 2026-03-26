"""
Data transforms cho HolmHz.

Triết lý:
- Train: augment MẠNH (JPEG, blur, flip, color jitter) để chống overfitting
- Val/Test: KHÔNG augment, chỉ resize + normalize (đo đúng sức thật)

Pattern từ:
- CNNDetection: JPEG compression + Gaussian blur là augmentation QUAN TRỌNG NHẤT
- UniversalFakeDetect: Preprocessing PHẢI match backbone (ImageNet vs CLIP)

Tại sao Albumentations mà không phải torchvision.transforms?
→ Nhanh hơn 2-5x (OpenCV backend)
→ Hỗ trợ JPEG compression augmentation (torchvision không có)
→ Được dùng trong production và paper
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


# === ImageNet Statistics ===
# EfficientNet-B0 được pre-train trên ImageNet với mean/std này.
# PHẢI dùng cùng giá trị — nếu khác, model "nhìn thế giới khác" → kết quả sai.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Kích thước chuẩn cho EfficientNet-B0
DEFAULT_IMAGE_SIZE = 224


def get_train_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
    """
    Transforms cho TRAINING v3 — Fix iPhone/Social Media False Positive.

    v3 thay đổi (Fix JPEG compression bias):
    - ⭐ Thêm DEDICATED JPEG stage (p=0.7, quality 50-95)
      → Riêng biệt, KHÔNG nằm trong OneOf → tỷ lệ cao được áp dụng
      → Mô phỏng: iPhone HEIC→JPEG, Facebook (q=80-85), Instagram (q=85-90)
    - ⭐ Thêm Social Media Resize Simulation (p=0.3)
      → Downscale 50-90% rồi scale back → double-compression artifacts
    - Giữ nguyên các aug khác từ v2

    Lý do: Model v2 học JPEG compression artifacts = "ảnh gốc" signal.
    iPhone/Facebook re-encode tạo double-encoding artifacts → false positive.
    JPEG aug mạnh buộc model bỏ qua compression, học authenticity features.
    """
    return A.Compose([
        # 1. Random crop + resize — phá spatial artifacts
        A.OneOf([
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
            ),
            A.Resize(image_size, image_size),
        ], p=1.0),

        # 2. Lật ngang
        A.HorizontalFlip(p=0.5),

        # 3. ⭐ DEDICATED JPEG Compression — mô phỏng social media pipeline
        # p=0.7: 70% ảnh sẽ bị compress → model PHẢI immune với JPEG artifacts
        # quality 50-95: bao phủ Facebook (80-85), Instagram (85-90), WhatsApp (60-75)
        A.ImageCompression(quality_range=(50, 95), p=0.7),

        # 4. ⭐ Social Media Resize Simulation (p=0.3)
        # Facebook/IG resize ảnh trước khi compress → double artifacts
        A.Downscale(scale_range=(0.5, 0.9), p=0.3),

        # 5. Nhóm augmentation phụ (blur, noise)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.GaussNoise(std_range=(0.01, 0.05)),
        ], p=0.3),

        # 6. Thay đổi màu sắc
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.05,
            p=0.5,
        ),

        # 7. Normalize → ImageNet standard
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # 8. numpy → PyTorch tensor [C, H, W]
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
    """
    Transforms cho VALIDATION và TEST — KHÔNG augment.

    Chỉ resize + normalize (giống điều kiện inference khi deploy).
    Muốn đo đúng sức mạnh thật của model.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])