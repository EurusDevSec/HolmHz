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
    Transforms cho TRAINING — augment mạnh để chống overfitting.

    Mô phỏng điều kiện thực tế: ảnh trên mạng bị nén JPEG, resize,
    chụp lại màn hình, thay đổi ánh sáng...
    Model phải chịu được tất cả biến dạng này.
    """
    return A.Compose([
        # 1. Resize về kích thước chuẩn
        A.Resize(image_size, image_size),

        # 2. Lật ngang ngẫu nhiên (50% chance)
        # Khuôn mặt đối xứng → lật không thay đổi Real/Fake
        A.HorizontalFlip(p=0.5),

        # 3. Nhóm augmentation chính (30% chance áp dụng 1 trong 3)
        A.OneOf([
            # ⭐ JPEG Compression — QUAN TRỌNG NHẤT cho deepfake detection
            # Ảnh trên mạng luôn bị nén JPEG (quality 60-100)
            A.ImageCompression(quality_range=(60, 100)),
            # Gaussian Blur — mô phỏng ảnh share qua MXH bị blur
            A.GaussianBlur(blur_limit=(3, 7)),
            # Gaussian Noise — mô phỏng camera giá rẻ
            # std_range: normalized [0,1] scale — (0.01, 0.03) ≈ nhẹ vừa phải
            A.GaussNoise(std_range=(0.01, 0.03)),
        ], p=0.3),

        # 4. Thay đổi màu sắc nhẹ (30% chance)
        # Ảnh thật chụp dưới nhiều điều kiện ánh sáng
        A.ColorJitter(
            brightness=0.1, contrast=0.1,
            saturation=0.1, hue=0.05,
            p=0.3,
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