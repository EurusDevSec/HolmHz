"""
Learning Rate Scheduler factory.

Scheduler điều chỉnh learning rate trong quá trình training.
Cosine Annealing: LR giảm theo đường cong cosine từ lr_max → lr_min.

Tại sao cần factory?
→ Config YAML chỉ đổi training.scheduler: cosine → code tự tạo
→ Sau này thêm StepLR, ReduceLROnPlateau dễ dàng
"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler


def get_scheduler(
    optimizer: Optimizer,
    name: str = "cosine",
    epochs: int = 30,
    eta_min: float = 1e-6,
) -> LRScheduler:
    """Factory tạo LR scheduler theo tên.

    Args:
        optimizer: PyTorch optimizer (đã tạo trước)
        name: tên scheduler
            - "cosine": CosineAnnealingLR (mặc định)
        epochs: tổng số epochs (T_max cho CosineAnnealing)
        eta_min: learning rate tối thiểu cuối cùng

    Returns:
        LR scheduler

    Example:
        >>> optimizer = AdamW(model.parameters(), lr=0.001)
        >>> scheduler = get_scheduler(optimizer, "cosine", epochs=30)
        >>> # Mỗi epoch:
        >>> scheduler.step()  # LR giảm theo cosine curve
    """
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=eta_min,
        )

    raise ValueError(
        f"Unknown scheduler: '{name}'. Available: ['cosine']"
    )
