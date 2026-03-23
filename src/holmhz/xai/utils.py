"""XAI utilities — image loading and gallery generation."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms


# ImageNet normalization (same as training pipeline)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image_for_gradcam(
    image_path: str | Path, image_size: int = 224
) -> tuple[torch.Tensor, np.ndarray]:
    """Load image and prepare both tensor (for model) and numpy (for display).

    Args:
        image_path: Path to image file
        image_size: Resize target (default 224)

    Returns:
        (tensor [1,3,H,W] normalized, rgb_image [H,W,3] float32 0-1)
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size))

    # For display overlay: float32 [0, 1]
    rgb_image = img_resized.astype(np.float32) / 255.0

    # For model input: normalized tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(img_resized).unsqueeze(0)  # [1, 3, H, W]

    return tensor, rgb_image


def create_comparison_grid(
    images: list[np.ndarray],
    titles: list[str],
    cols: int = 4,
    cell_size: int = 256,
) -> np.ndarray:
    """Create a grid of images with titles for gallery view.

    Args:
        images: List of [H, W, 3] uint8 images
        titles: List of title strings
        cols: Number of columns
        cell_size: Size of each cell

    Returns:
        [grid_H, grid_W, 3] uint8 image
    """
    rows = (len(images) + cols - 1) // cols
    title_h = 30
    grid = np.ones(
        (rows * (cell_size + title_h), cols * cell_size, 3), dtype=np.uint8
    ) * 255

    for idx, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(idx, cols)
        y_off = r * (cell_size + title_h)
        x_off = c * cell_size

        # Resize image to cell_size
        resized = cv2.resize(img, (cell_size, cell_size))
        grid[y_off + title_h: y_off + title_h + cell_size, x_off: x_off + cell_size] = resized

        # Add title text
        cv2.putText(
            grid, title,
            (x_off + 5, y_off + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
        )

    return grid
