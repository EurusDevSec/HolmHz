"""
Loss functions cho HolmHz.

Loss function = "thước đo sai lầm" — loss càng cao, model càng sai.
Mục tiêu training: GIẢM loss (optimizer step giảm loss mỗi iteration).

BCEWithLogitsLoss cho binary classification:
  - Binary: 2 class (Real/Fake)
  - CrossEntropy: đo khoảng cách giữa prediction và ground truth
  - WithLogits: nhận raw logits (chưa sigmoid) → numerical stability

Tại sao cần factory function?
→ Config YAML chỉ cần đổi loss.name → code tự tạo loss phù hợp
→ Sau này thêm Focal Loss (cho imbalanced data) dễ dàng
"""

import torch
import torch.nn as nn


def get_loss_fn(
    name: str = "bce_with_logits",
    pos_weight: float | None = None,
) -> nn.Module:
    """Factory tạo loss function theo tên.

    Args:
        name: tên loss function
            - "bce_with_logits": BCEWithLogitsLoss (mặc định, dùng cho HolmHz)
        pos_weight: trọng số cho class positive (Fake)
            Nếu data imbalanced (ví dụ 60% Real, 40% Fake):
            pos_weight = 60/40 = 1.5 → phạt nặng hơn khi miss Fake
            None = cân bằng (mặc định)

    Returns:
        nn.Module — loss function

    Example:
        >>> loss_fn = get_loss_fn("bce_with_logits")
        >>> logits = torch.tensor([0.5, -0.3])
        >>> labels = torch.tensor([1.0, 0.0])
        >>> loss = loss_fn(logits, labels)
    """
    if name == "bce_with_logits":
        weight = torch.tensor([pos_weight]) if pos_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=weight)

    raise ValueError(
        f"Unknown loss: '{name}'. Available: ['bce_with_logits']"
    )
