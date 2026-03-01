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
    Transforms cho TRAINING v2 — augment MẠNH HƠN để chống shortcut learning.

    Thay đổi so với v1 (Task 1.7 OOD Improvement):
    - Thêm RandomResizedCrop (0.7-1.0, p=0.5) — phá spatial artifacts
    - OneOf p: 0.3 → 0.5 (áp dụng nhiều hơn)
    - JPEG quality: 60-100 → 30-100 (aggressive hơn)
    - GaussianBlur: 3-7 → 3-9 (mạnh hơn)
    - Thêm Downscale (0.25-0.9) — mô phỏng multi-resolution
    - ColorJitter p: 0.3 → 0.5

    Lý do: Model v1 học shortcut từ preprocessing artifacts (cifake 32x32 upscale,
    ffhq face alignment). Augmentation mạnh hơn → phá các artifacts này.
    """
    return A.Compose([
        # 1. Random crop + resize (50% chance) — phá spatial artifacts
        # NẾU không crop → chỉ resize bình thường
        A.OneOf([
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
            ),
            A.Resize(image_size, image_size),
        ], p=1.0),  # Luôn chọn 1 trong 2

        # 2. Lật ngang ngẫu nhiên (50% chance)
        # Khuôn mặt đối xứng → lật không thay đổi Real/Fake
        A.HorizontalFlip(p=0.5),

        # 3. Nhóm augmentation chính — TĂNG p từ 0.3 → 0.5
        A.OneOf([
            # ⭐ JPEG Compression — aggressive hơn (quality 30-100)
            A.ImageCompression(quality_range=(30, 100)),
            # Gaussian Blur — mạnh hơn (3-9)
            A.GaussianBlur(blur_limit=(3, 9)),
            # Gaussian Noise
            A.GaussNoise(std_range=(0.01, 0.05)),
            # Downscale — mô phỏng ảnh resolution thấp
            A.Downscale(scale_range=(0.25, 0.9)),
        ], p=0.5),

        # 4. Thay đổi màu sắc — TĂNG p từ 0.3 → 0.5
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.05,
            p=0.5,
        ),

        # 5. Normalize (BẮT BUỘC, luôn áp dụng)
        # Đưa pixel từ [0,255] → chuẩn ImageNet
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # 6. Chuyển numpy array → PyTorch tensor [C, H, W]
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