"""
AUC (Area Under ROC Curve) — metric CHÍNH của HolmHz.

Tại sao AUC mà không phải Accuracy?
→ AUC đo khả năng PHÂN BIỆT giữa 2 class, KHÔNG phụ thuộc threshold
→ AUC = 1.0: phân biệt hoàn hảo (P(Fake|fake) > P(Fake|real) mọi lúc)
→ AUC = 0.5: đoán ngẫu nhiên (tệ như tung đồng xu)
→ AUC < 0.5: model dự đoán ngược (đổi label sẽ tốt hơn!)

KPI dự án:
  - In-domain AUC ≥ 0.90
  - OOD AUC ≥ 0.75

Dùng sklearn.metrics.roc_auc_score — thư viện chuẩn cho ML metrics.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def compute_auc(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Tính AUC từ logits và labels.

    Args:
        logits: [N] hoặc [N, 1] — raw logits từ model
        labels: [N] — ground truth (0.0 = Real, 1.0 = Fake)

    Returns:
        auc: float ∈ [0.0, 1.0]

    Edge cases:
        - Nếu chỉ có 1 class trong batch → trả về 0.5 (không tính được)
        - Batch rất nhỏ → AUC có thể không ổn định

    Example:
        >>> logits = torch.tensor([2.0, -1.0, 3.0, -2.0])
        >>> labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        >>> compute_auc(logits, labels)
        1.0  # Phân biệt hoàn hảo
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Edge case: chỉ có 1 class trong batch (toàn Real hoặc toàn Fake)
        # sklearn sẽ raise error → trả về 0.5 (uncertain)
        if len(np.unique(labels_np)) < 2:
            return 0.5

        return float(roc_auc_score(labels_np, probs))
