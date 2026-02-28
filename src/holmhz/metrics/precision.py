"""
Precision — "Khi model nói FAKE, nó đúng bao nhiêu %?"

Precision = TP / (TP + FP)

Precision THẤP → nhiều False Positive (Real bị đoán FAKE).
→ Đây chính là vấn đề phát hiện ở smoke test Task 1.6!

Ví dụ:
  Model đoán 10 ảnh FAKE, nhưng 4 trong đó thực ra là Real.
  Precision = 6/10 = 0.60 ← "40% dự đoán FAKE là sai"
"""

import torch


def compute_precision(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Tính Precision từ logits và labels.

    Args:
        logits: [N] hoặc [N, 1] — raw logits từ model.
        labels: [N] — ground truth (0.0 = Real, 1.0 = Fake).
        threshold: Ngưỡng phân loại (default 0.5).

    Returns:
        precision: float ∈ [0.0, 1.0].

    Edge cases:
        - Model không dự đoán FAKE nào → Precision = 0.0
          (division by zero protection)
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.squeeze())
        preds = (probs >= threshold).float()

        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()

        if (tp + fp) > 0:
            return float(tp / (tp + fp))
        return 0.0
