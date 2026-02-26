"""
Accuracy metric cho binary classification.

Accuracy = số dự đoán đúng / tổng số mẫu.
Đơn giản nhất, dễ hiểu nhất, nhưng KHÔNG PHẢI metric tốt nhất
khi data imbalanced (nhiều Real hơn Fake hoặc ngược lại).

Ví dụ: 60% Real, 40% Fake
  → Model luôn đoán "Real" → accuracy = 60% (cao nhưng vô nghĩa!)
  → Vì vậy, dùng AUC (metrics/auc.py) làm metric chính.
"""

import torch


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Tính accuracy từ logits và labels.

    Flow:
        logits → sigmoid → probs → (>threshold?) → preds → so sánh labels

    Args:
        logits: [N] hoặc [N, 1] — raw logits từ model
        labels: [N] — ground truth (0.0 = Real, 1.0 = Fake)
        threshold: ngưỡng phân loại (default 0.5)

    Returns:
        accuracy: float ∈ [0.0, 1.0]

    Example:
        >>> logits = torch.tensor([2.0, -1.0, 0.5])
        >>> labels = torch.tensor([1.0, 0.0, 1.0])
        >>> compute_accuracy(logits, labels)
        0.6667  # 2/3 đúng (2.0→Fake✓, -1.0→Real✓, 0.5→Fake nhưng sigmoid=0.62✓)
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.squeeze())
        preds = (probs >= threshold).float()
        correct = (preds == labels).sum().item()
        total = labels.numel()
        return correct / total if total > 0 else 0.0
