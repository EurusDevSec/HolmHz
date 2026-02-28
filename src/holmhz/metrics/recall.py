"""
Recall — "Trong tất cả Fake thật, model tìm ra được bao nhiêu %?"

Recall = TP / (TP + FN)

Recall THẤP → nhiều False Negative (Fake bị bỏ sót, đoán thành Real).
→ Nguy hiểm nếu mục đích là "catch all fakes".

Ví dụ:
  Có 10 ảnh Fake thật, model chỉ tìm ra 7.
  Recall = 7/10 = 0.70 ← "bỏ sót 30% Fake"
"""

import torch


def compute_recall(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Tính Recall từ logits và labels.

    Args:
        logits: [N] hoặc [N, 1] — raw logits từ model.
        labels: [N] — ground truth (0.0 = Real, 1.0 = Fake).
        threshold: Ngưỡng phân loại (default 0.5).

    Returns:
        recall: float ∈ [0.0, 1.0].

    Edge cases:
        - Không có Fake nào trong data → Recall = 0.0
          (không có TP hay FN)
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.squeeze())
        preds = (probs >= threshold).float()

        tp = ((preds == 1) & (labels == 1)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()

        if (tp + fn) > 0:
            return float(tp / (tp + fn))
        return 0.0
