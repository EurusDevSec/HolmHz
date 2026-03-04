# analysis/compare_models.py
"""
So sánh kết quả benchmark giữa các models.

Đọc predictions CSV từ outputs/benchmark/predictions/,
tính metrics, tạo:
1. Bảng so sánh (stdout + markdown)
2. ROC overlay plot (ID + OOD)
3. Per-source accuracy comparison

Usage:
    python analysis/compare_models.py
"""

import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc


def load_predictions(csv_path: str) -> dict:
    """Load predictions CSV → dict of numpy arrays."""
    labels, probs, sources, splits = [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
            probs.append(float(row["prob_fake"]))
            sources.append(row["source"])
            splits.append(row["split"])
    return {
        "labels": np.array(labels),
        "probs": np.array(probs),
        "sources": np.array(sources),
        "splits": np.array(splits),
    }


def compute_metrics(labels, probs, threshold=0.5):
    """Compute AUC, Accuracy, F1, Precision, Recall."""
    preds = (probs >= threshold).astype(int)
    acc = (preds == labels).mean()

    try:
        auc_val = roc_auc_score(labels, probs)
    except ValueError:
        auc_val = 0.5

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "auc": float(auc_val),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
    }


def per_source_accuracy(labels, probs, sources, threshold=0.5):
    """Accuracy breakdown by source."""
    preds = (probs >= threshold).astype(int)
    result = {}
    for src in sorted(set(sources)):
        mask = sources == src
        result[src] = {
            "accuracy": float((preds[mask] == labels[mask]).mean()),
            "n": int(mask.sum()),
        }
    return result


def plot_roc_overlay(all_preds: dict, output_path: str):
    """ROC overlay: ID + OOD side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "holmhz": "#2196F3",
        "cnndetection": "#FF5722",
        "universalfake": "#4CAF50",
        "deepfakebench": "#9C27B0",
    }

    for split_name, split_key, ax in [
        ("In-Domain", "id", axes[0]),
        ("OOD (Primary — 100% Fair)", "ood", axes[1]),
    ]:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (0.50)")

        for model_name, data in all_preds.items():
            mask = data["splits"] == split_key
            labels_split = data["labels"][mask]
            if len(set(labels_split)) < 2:
                continue
            probs_split = data["probs"][mask]
            fpr, tpr, _ = roc_curve(labels_split, probs_split)
            roc_auc = auc(fpr, tpr)

            color = colors.get(model_name, "#666")
            ax.plot(
                fpr, tpr, color=color, linewidth=2,
                label=f"{model_name} ({roc_auc:.4f})",
            )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {split_name}")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_per_source_bar(all_preds: dict, output_path: str):
    """Per-source OOD accuracy bar chart."""
    # Collect OOD sources
    ood_sources = set()
    for data in all_preds.values():
        ood_mask = data["splits"] == "ood"
        ood_sources.update(set(data["sources"][ood_mask]))
    ood_sources = sorted(ood_sources)

    if not ood_sources:
        return

    colors = {
        "holmhz": "#2196F3",
        "cnndetection": "#FF5722",
        "universalfake": "#4CAF50",
        "deepfakebench": "#9C27B0",
    }

    model_names = list(all_preds.keys())
    n_models = len(model_names)
    n_sources = len(ood_sources)
    bar_width = 0.8 / n_models
    x = np.arange(n_sources)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_name in enumerate(model_names):
        data = all_preds[model_name]
        ood_mask = data["splits"] == "ood"
        ps = per_source_accuracy(
            data["labels"][ood_mask],
            data["probs"][ood_mask],
            data["sources"][ood_mask],
        )
        accs = [ps.get(src, {}).get("accuracy", 0) * 100 for src in ood_sources]
        color = colors.get(model_name, "#666")
        bars = ax.bar(
            x + i * bar_width, accs, bar_width,
            label=model_name, color=color, alpha=0.85,
        )
        # Add value labels
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{acc:.0f}%", ha="center", va="bottom", fontsize=8,
                )

    ax.set_xlabel("OOD Source")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Source OOD Accuracy Comparison")
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(ood_sources)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    pred_dir = Path("outputs/benchmark/predictions")
    out_dir = Path("outputs/benchmark/comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    all_preds = {}
    for csv_file in sorted(pred_dir.glob("*_predictions.csv")):
        name = csv_file.stem.replace("_predictions", "")
        all_preds[name] = load_predictions(str(csv_file))
        print(f"  Loaded: {name} ({len(all_preds[name]['labels'])} samples)")

    if not all_preds:
        print("No predictions found in outputs/benchmark/predictions/")
        print("   Run: python scripts/benchmark_sota.py --model <name>")
        return

    # ── Overall comparison table ──
    print(f"\n{'='*70}")
    print("COMPARISON TABLE (threshold=0.5)")
    print(f"{'='*70}\n")

    header = (
        f"| {'Model':<20} | {'ID AUC':>7} | {'ID Acc':>7} "
        f"| {'OOD AUC':>8} | {'OOD Acc':>8} | {'OOD F1':>7} |"
    )
    sep = f"|{'-'*22}|{'-'*9}|{'-'*9}|{'-'*10}|{'-'*10}|{'-'*9}|"
    print(header)
    print(sep)

    md_lines = [
        "# Model Comparison — HolmHz Benchmark\n",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "> Note: OOD test set is 100% fair (disjoint from all models' training data).",
        "> ID test set contains 12.5% sources unique to HolmHz training.\n",
        "",
        "## Overall Metrics (threshold=0.5)\n",
        "| Model | ID AUC | ID Acc | OOD AUC | OOD Acc | OOD F1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for name, data in all_preds.items():
        id_mask = data["splits"] == "id"
        ood_mask = data["splits"] == "ood"

        id_m = compute_metrics(data["labels"][id_mask], data["probs"][id_mask])
        ood_m = compute_metrics(data["labels"][ood_mask], data["probs"][ood_mask])

        row = (
            f"| {name:<20} | {id_m['auc']:>7.4f} | {id_m['accuracy']*100:>6.1f}% "
            f"| {ood_m['auc']:>8.4f} | {ood_m['accuracy']*100:>7.1f}% "
            f"| {ood_m['f1']:>7.4f} |"
        )
        print(row)
        md_lines.append(
            f"| **{name}** | {id_m['auc']:.4f} | {id_m['accuracy']*100:.1f}% "
            f"| {ood_m['auc']:.4f} | {ood_m['accuracy']*100:.1f}% "
            f"| {ood_m['f1']:.4f} |"
        )

    # ── Per-source OOD accuracy ──
    print(f"\n{'='*70}")
    print("PER-SOURCE OOD ACCURACY")
    print(f"{'='*70}\n")

    ood_sources = set()
    for data in all_preds.values():
        ood_mask = data["splits"] == "ood"
        ood_sources.update(set(data["sources"][ood_mask]))
    ood_sources = sorted(ood_sources)

    md_lines.extend([
        "",
        "## Per-Source OOD Accuracy\n",
        "| Model | " + " | ".join(ood_sources) + " |",
        "| --- | " + " | ".join(["---"] * len(ood_sources)) + " |",
    ])

    for name, data in all_preds.items():
        ood_mask = data["splits"] == "ood"
        ps = per_source_accuracy(
            data["labels"][ood_mask],
            data["probs"][ood_mask],
            data["sources"][ood_mask],
        )
        row_parts = []
        for src in ood_sources:
            if src in ps:
                row_parts.append(f"{ps[src]['accuracy']*100:.1f}%")
            else:
                row_parts.append("N/A")
        print(
            f"  {name:<20}: "
            + ", ".join(f"{s}={v}" for s, v in zip(ood_sources, row_parts))
        )
        md_lines.append(f"| **{name}** | " + " | ".join(row_parts) + " |")

    # Save markdown
    md_path = out_dir / "comparison_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\n  Saved: {md_path}")

    # ── ROC plot ──
    plot_roc_overlay(all_preds, str(out_dir / "roc_overlay.png"))

    # ── Per-source bar chart ──
    plot_per_source_bar(all_preds, str(out_dir / "per_source_ood_accuracy.png"))

    print(f"\n All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
