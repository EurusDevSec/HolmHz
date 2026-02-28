"""
F1 Score — Harmonic mean của Precision và Recall.

F1 cân bằng giữa "đoán đúng khi nói FAKE" (Precision) và
"tìm ra được bao nhiêu FAKE" (Recall).

F1 = 2 × (Precision × Recall) / (Precision + Recall)

F1 tốt khi CẢ Precision và Recall đều cao.
Nếu 1 trong 2 rất thấp → F1 cũng thấp (penalize imbalance).

Ví dụ:
  Precision=0.90, Recall=0.90 → F1=0.90 (tốt)
  Precision=0.99, Recall=0.10 → F1=0.18 (tệ! Recall quá thấp)
"""

import torch


def compute_f1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Tính F1 score từ logits và labels.

    Args:
        logits: [N] hoặc [N, 1] — raw logits từ model.
        labels: [N] — ground truth (0.0 = Real, 1.0 = Fake).
        threshold: Ngưỡng phân loại (default 0.5).

    Returns:
        f1: float ∈ [0.0, 1.0].

    Edge cases:
        - Không có TP → F1 = 0.0 (model không tìm được Fake nào).
        - Không có Positive predictions → F1 = 0.0.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.squeeze())
        preds = (probs >= threshold).float()

        # True Positives, False Positives, False Negatives
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()

        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)

        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)

        # F1 = 2 × (Precision × Recall) / (Precision + Recall)
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = torch.tensor(0.0)

        return float(f1)
