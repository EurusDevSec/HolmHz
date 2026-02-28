"""
Visualization utilities — confusion matrix, ROC curve, per-source accuracy.

Tất cả các hàm đều save file PNG trực tiếp (non-interactive backend),
phù hợp cho server/CI/CD.

Usage:
    from holmhz.utils.visualization import plot_confusion_matrix, plot_roc_curve
    plot_confusion_matrix(labels, logits, save_path="outputs/evaluation/cm.png")
    plot_roc_curve(results_dict, save_path="outputs/evaluation/roc.png")
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc

from holmhz.utils.logger import get_logger

logger = get_logger("visualization")


def plot_confusion_matrix(
    labels: torch.Tensor | np.ndarray,
    logits: torch.Tensor | np.ndarray,
    save_path: str,
    title: str = "Confusion Matrix",
    threshold: float = 0.5,
) -> str:
    """Vẽ confusion matrix và save thành PNG.

    Args:
        labels: [N] ground truth (0=Real, 1=Fake).
        logits: [N] raw logits (sẽ qua sigmoid → threshold).
        save_path: Đường dẫn file PNG output.
        title: Tiêu đề biểu đồ.
        threshold: Ngưỡng phân loại.

    Returns:
        save_path: Đường dẫn file đã save.
    """
    # Convert to numpy
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(logits, torch.Tensor):
        probs = torch.sigmoid(logits).cpu().numpy()
    else:
        probs = 1 / (1 + np.exp(-logits))  # sigmoid numpy

    preds = (probs >= threshold).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Real", "Fake"],
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Thêm annotation: tổng số, tỷ lệ sai
    total = cm.sum()
    correct = cm.diagonal().sum()
    accuracy = correct / total if total > 0 else 0
    fp = cm[0, 1]  # Real → predicted Fake
    fn = cm[1, 0]  # Fake → predicted Real
    ax.set_xlabel(
        f"Predicted Label\n\n"
        f"Total: {total} | Accuracy: {accuracy:.1%} | "
        f"FP (Real→Fake): {fp} | FN (Fake→Real): {fn}",
        fontsize=10,
    )

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Confusion matrix saved: {save_path}")
    return save_path


def plot_roc_curve(
    results_dict: dict,
    save_path: str,
    title: str = "ROC Curve — ID vs OOD",
) -> str:
    """Vẽ ROC curves cho nhiều test set trên cùng 1 biểu đồ.

    Args:
        results_dict: Dict[name → {"all_logits": tensor, "all_labels": tensor}]
            Ví dụ: {"In-Domain": id_results, "OOD": ood_results}
        save_path: Đường dẫn file PNG output.
        title: Tiêu đề biểu đồ.

    Returns:
        save_path: Đường dẫn file đã save.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800"]

    for i, (name, results) in enumerate(results_dict.items()):
        logits = results["all_logits"]
        labels = results["all_labels"]

        if isinstance(logits, torch.Tensor):
            probs = torch.sigmoid(logits).cpu().numpy()
        else:
            probs = 1 / (1 + np.exp(-logits))
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = labels

        # Kiểm tra có ít nhất 2 class
        if len(np.unique(labels_np)) < 2:
            logger.warning(f"Skipping {name}: only 1 class in data")
            continue

        fpr, tpr, _ = roc_curve(labels_np, probs)
        roc_auc = auc(fpr, tpr)

        color = colors[i % len(colors)]
        ax.plot(
            fpr, tpr,
            color=color,
            lw=2,
            label=f"{name} (AUC = {roc_auc:.4f})",
        )

    # Đường chéo (random baseline)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.5)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"ROC curve saved: {save_path}")
    return save_path


def plot_per_source_accuracy(
    per_source: dict,
    save_path: str,
    title: str = "Per-Source Accuracy",
) -> str:
    """Vẽ bar chart accuracy cho mỗi source.

    Args:
        per_source: Dict[source → {"accuracy": float, "n": int, ...}]
        save_path: Đường dẫn file PNG output.
        title: Tiêu đề biểu đồ.

    Returns:
        save_path: Đường dẫn file đã save.
    """
    sources = list(per_source.keys())
    accuracies = [per_source[s]["accuracy"] for s in sources]
    counts = [per_source[s]["n"] for s in sources]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color: xanh nếu accuracy >= 0.8, cam nếu >= 0.5, đỏ nếu < 0.5
    colors = []
    for acc in accuracies:
        if acc >= 0.8:
            colors.append("#4CAF50")   # Green
        elif acc >= 0.5:
            colors.append("#FF9800")   # Orange
        else:
            colors.append("#F44336")   # Red

    bars = ax.bar(sources, accuracies, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate mỗi bar: accuracy + count
    for bar, acc, n in zip(bars, accuracies, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.1%}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline")
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.3, label="Good threshold")
    ax.legend(fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Per-source accuracy chart saved: {save_path}")
    return save_path