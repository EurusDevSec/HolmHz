"""
Threshold Optimization — Strategy 2 (S2)

Tìm optimal classification threshold cho HolmHz model.
Thay vì dùng threshold=0.5 cố định, tìm threshold tối ưu trên val set,
sau đó đánh giá lại OOD performance.

3 phương pháp tìm threshold:
  1. Youden's J statistic (maximize TPR - FPR) → best overall separation
  2. Max F1 Score → best balance precision/recall
  3. Target FPR ≤ 0.15 → giảm false positives (real → fake)

Usage:
    # Tìm threshold + evaluate OOD
    python analysis/find_threshold.py

    # Custom checkpoint
    python analysis/find_threshold.py model.checkpoint=outputs/checkpoints/best_v4.pt

    # Chỉ phân tích, không evaluate OOD
    python analysis/find_threshold.py --analyze-only
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv()

import holmhz.detectors  # noqa: E402, F401
from holmhz.data import create_dataloader
from holmhz.utils.logger import get_logger
from holmhz.utils.registry import DETECTOR_REGISTRY

logger = get_logger("find_threshold")

# ═══════════════════════════════════════════════════
# Output directory
# ═══════════════════════════════════════════════════
OUTPUT_DIR = Path("outputs/threshold_analysis")


def collect_predictions(model, dataloader, device):
    """Chạy inference, trả về logits, probabilities, labels, sources."""
    model.eval()
    model.to(device)

    all_logits = []
    all_labels = []
    all_sources = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"]
            sources = batch["source"]

            logits = model(images).squeeze(-1)

            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_sources.extend(sources)

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = 1 / (1 + np.exp(-all_logits))  # sigmoid

    return all_logits, all_probs, all_labels, all_sources


def find_optimal_thresholds(y_true, y_prob):
    """
    Tìm optimal threshold bằng 3 phương pháp.

    Returns:
        dict với threshold + metrics cho mỗi phương pháp.
    """
    results = {}

    # ─── Method 1: Youden's J statistic ───
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    t_youden = float(thresholds_roc[optimal_idx])

    results["youden"] = {
        "threshold": t_youden,
        "method": "Youden's J (max TPR-FPR)",
        "j_score": float(j_scores[optimal_idx]),
        "tpr_at_threshold": float(tpr[optimal_idx]),
        "fpr_at_threshold": float(fpr[optimal_idx]),
    }

    # ─── Method 2: Max F1 Score ───
    best_f1, best_t_f1 = 0, 0.5
    for t in np.arange(0.20, 0.80, 0.005):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t_f1 = float(t)

    results["max_f1"] = {
        "threshold": best_t_f1,
        "method": "Max F1 Score",
        "f1_score": best_f1,
    }

    # ─── Method 3: Target FPR ≤ 15% ───
    # FPR ≤ 0.15 nghĩa là ≤ 15% real images bị gọi sai là fake
    target_fpr = 0.15
    valid = fpr <= target_fpr
    if valid.any():
        idx = np.where(valid)[0][-1]  # lấy threshold thấp nhất mà FPR ≤ target
        t_fpr = float(thresholds_roc[idx])
    else:
        t_fpr = 0.5

    results["low_fpr"] = {
        "threshold": t_fpr,
        "method": f"FPR ≤ {target_fpr:.0%}",
        "fpr_at_threshold": float(fpr[idx]) if valid.any() else None,
        "tpr_at_threshold": float(tpr[idx]) if valid.any() else None,
    }

    return results


def evaluate_with_threshold(y_true, y_prob, threshold, sources=None):
    """Tính metrics với một threshold cụ thể."""
    preds = (y_prob >= threshold).astype(int)

    overall = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }

    # Per-source
    per_source = {}
    if sources is not None:
        unique_sources = sorted(set(sources))
        for src in unique_sources:
            mask = np.array([s == src for s in sources])
            src_y = y_true[mask]
            src_p = preds[mask]
            src_prob = y_prob[mask]

            per_source[src] = {
                "accuracy": float(accuracy_score(src_y, src_p)),
                "n": int(mask.sum()),
                "n_real": int((src_y == 0).sum()),
                "n_fake": int((src_y == 1).sum()),
                "mean_prob": float(src_prob.mean()),
                "median_prob": float(np.median(src_prob)),
                "std_prob": float(src_prob.std()),
            }

    return overall, per_source


def analyze_probability_distribution(y_prob, y_true, sources):
    """Phân tích distribution P(fake) cho mỗi source → hiểu model behaviour."""
    analysis = {}
    unique_sources = sorted(set(sources))

    for src in unique_sources:
        mask = np.array([s == src for s in sources])
        src_prob = y_prob[mask]
        src_labels = y_true[mask]

        # Label info
        label = "Fake" if src_labels.mean() > 0.5 else "Real"

        analysis[src] = {
            "label": label,
            "n": int(mask.sum()),
            "prob_mean": float(src_prob.mean()),
            "prob_median": float(np.median(src_prob)),
            "prob_std": float(src_prob.std()),
            "prob_min": float(src_prob.min()),
            "prob_max": float(src_prob.max()),
            "prob_q25": float(np.percentile(src_prob, 25)),
            "prob_q75": float(np.percentile(src_prob, 75)),
            # Bao nhiêu % nằm trong vùng "uncertain" (0.4 - 0.6)?
            "pct_uncertain": float(
                ((src_prob >= 0.4) & (src_prob <= 0.6)).mean() * 100
            ),
        }

    return analysis


def plot_threshold_analysis(
    val_analysis, ood_analysis, thresholds, ood_comparison, output_dir
):
    """Vẽ charts phân tích threshold."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Chart 1: Probability Distribution per OOD source ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("P(Fake) Distribution — OOD Sources", fontsize=14, fontweight="bold")

    sources_to_plot = list(ood_analysis.keys())[:4]
    for i, src in enumerate(sources_to_plot):
        ax = axes[i // 2][i % 2]
        info = ood_analysis[src]

        # Label=1 → Fake, Label=0 → Real
        is_fake = info["label"] == "Fake"
        color = "#e74c3c" if is_fake else "#2ecc71"
        label_text = f"{src} ({info['label']}, N={info['n']})"

        # Draw histogram (approximate from stats)
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.barh(
            [0],
            [info["prob_mean"]],
            height=0.3,
            color=color,
            alpha=0.7,
            label=label_text,
        )
        ax.errorbar(
            info["prob_mean"],
            0,
            xerr=info["prob_std"],
            fmt="o",
            color="black",
            capsize=5,
        )

        # Threshold lines
        for name, t_info in thresholds.items():
            t = t_info["threshold"]
            ax.axvline(x=t, color="gray", linestyle="--", alpha=0.5)
            ax.text(t, 0.2, f'{name}\n{t:.3f}', ha="center", fontsize=8)

        ax.axvline(x=0.5, color="blue", linestyle="-", linewidth=2, alpha=0.3, label="Default 0.5")

        ax.set_xlim(0, 1)
        ax.set_title(label_text, fontsize=11)
        ax.set_xlabel("P(Fake)")
        ax.set_yticks([])
        ax.text(
            0.02,
            0.95,
            f"mean={info['prob_mean']:.3f}\n"
            f"median={info['prob_median']:.3f}\n"
            f"std={info['prob_std']:.3f}\n"
            f"uncertain={info['pct_uncertain']:.1f}%",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "ood_probability_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")

    # ─── Chart 2: Threshold Comparison Table (visual) ───
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        "OOD Accuracy Comparison — Different Thresholds",
        fontsize=14,
        fontweight="bold",
    )

    methods = list(ood_comparison.keys())
    sources = list(next(iter(ood_comparison.values()))["per_source"].keys())
    x = np.arange(len(sources))
    width = 0.8 / len(methods)

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for i, method in enumerate(methods):
        accs = [
            ood_comparison[method]["per_source"][src]["accuracy"] * 100
            for src in sources
        ]
        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            accs,
            width,
            label=f'{method} (t={ood_comparison[method]["overall"]["threshold"]:.3f})',
            color=colors[i % len(colors)],
            alpha=0.8,
        )
        for bar, acc in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{acc:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("OOD Source")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(sources, fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "threshold_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def main():
    """Main threshold analysis pipeline."""
    # ─── Load config ───
    config = OmegaConf.load("configs/test.yaml")

    # CLI overrides
    cli_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if cli_args:
        cli_overrides = OmegaConf.from_cli(cli_args)
        config = OmegaConf.merge(config, cli_overrides)

    analyze_only = "--analyze-only" in sys.argv

    # ─── Device ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ─── Load model ───
    checkpoint_path = config.model.checkpoint
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    model = DETECTOR_REGISTRY.build(
        config.model.name,
        pretrained=False,
        dropout=config.model.get("dropout", 0.3),
        freeze_backbone=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "N/A")
    logger.info(f"Model loaded: {checkpoint_path} (epoch {epoch})")

    # ─── DataLoaders ───
    batch_size = config.data.get("batch_size", 32)
    num_workers = config.data.get("num_workers", 0)

    val_loader = create_dataloader(
        manifest_path="data/manifests/val.json",
        batch_size=batch_size,
        image_size=config.data.image_size,
        is_training=False,
        num_workers=num_workers,
    )

    ood_loader = create_dataloader(
        manifest_path=config.data.ood_manifest,
        batch_size=batch_size,
        image_size=config.data.image_size,
        is_training=False,
        num_workers=num_workers,
    )

    # ═══════════════════════════════════════════════════
    # STEP 1: Collect predictions on VAL set
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1: COLLECTING VAL PREDICTIONS")
    print("=" * 70)

    val_logits, val_probs, val_labels, val_sources = collect_predictions(
        model, val_loader, device
    )
    logger.info(f"Val set: {len(val_labels)} samples")

    # ═══════════════════════════════════════════════════
    # STEP 2: Find optimal thresholds on VAL set
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2: FINDING OPTIMAL THRESHOLDS (on Val set)")
    print("=" * 70)

    thresholds = find_optimal_thresholds(val_labels, val_probs)

    print(f"\n{'Method':<30} {'Threshold':<12} {'Details'}")
    print("-" * 70)
    print(f"{'Default':<30} {'0.5000':<12} Fixed baseline")
    for name, info in thresholds.items():
        detail = info.get("j_score", info.get("f1_score", ""))
        if isinstance(detail, float):
            detail = f"{detail:.4f}"
        print(f"{info['method']:<30} {info['threshold']:<12.4f} {detail}")

    # ═══════════════════════════════════════════════════
    # STEP 3: Evaluate VAL set with each threshold
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3: VAL SET COMPARISON")
    print("=" * 70)

    val_comparison = {}

    # Default 0.5
    overall_default, _ = evaluate_with_threshold(val_labels, val_probs, 0.5, val_sources)
    val_comparison["default"] = {"overall": overall_default}
    print(f"\n  Default (0.5):    Acc={overall_default['accuracy']:.4f}  F1={overall_default['f1']:.4f}")

    for name, info in thresholds.items():
        t = info["threshold"]
        overall, _ = evaluate_with_threshold(val_labels, val_probs, t, val_sources)
        val_comparison[name] = {"overall": overall}
        print(f"  {info['method']:<27} Acc={overall['accuracy']:.4f}  F1={overall['f1']:.4f}")

    # ═══════════════════════════════════════════════════
    # STEP 4: Collect OOD predictions + evaluate with all thresholds
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 4: OOD EVALUATION WITH DIFFERENT THRESHOLDS")
    print("=" * 70)

    ood_logits, ood_probs, ood_labels, ood_sources = collect_predictions(
        model, ood_loader, device
    )
    logger.info(f"OOD set: {len(ood_labels)} samples")

    # Analyze probability distribution
    ood_analysis = analyze_probability_distribution(ood_probs, ood_labels, ood_sources)

    print("\n📊 OOD Probability Distribution:")
    print(f"\n  {'Source':<22} {'Label':<6} {'Mean P(F)':<10} {'Median':<10} {'Std':<8} {'Uncertain%':<10}")
    print("  " + "-" * 65)
    for src, info in sorted(ood_analysis.items()):
        uncertain_marker = " ⚠️" if info["pct_uncertain"] > 20 else ""
        print(
            f"  {src:<22} {info['label']:<6} "
            f"{info['prob_mean']:<10.4f} {info['prob_median']:<10.4f} "
            f"{info['prob_std']:<8.4f} {info['pct_uncertain']:<10.1f}{uncertain_marker}"
        )

    # Evaluate OOD with each threshold
    ood_comparison = {}

    # Default 0.5
    overall_d, per_source_d = evaluate_with_threshold(ood_labels, ood_probs, 0.5, ood_sources)
    ood_comparison["default"] = {"overall": overall_d, "per_source": per_source_d}

    for name, info in thresholds.items():
        t = info["threshold"]
        overall, per_source = evaluate_with_threshold(ood_labels, ood_probs, t, ood_sources)
        ood_comparison[name] = {"overall": overall, "per_source": per_source}

    # Print comparison
    print(f"\n\n{'='*70}")
    print("OOD RESULTS WITH DIFFERENT THRESHOLDS")
    print(f"{'='*70}")

    header_sources = sorted(set(ood_sources))
    print(f"\n  {'Method':<22} {'Threshold':<10} {'OOD Acc':<10}", end="")
    for src in header_sources:
        print(f" {src:<18}", end="")
    print()
    print("  " + "-" * (42 + 18 * len(header_sources)))

    for method_name, data in ood_comparison.items():
        t = data["overall"]["threshold"]
        acc = data["overall"]["accuracy"]
        print(f"  {method_name:<22} {t:<10.4f} {acc:<10.4f}", end="")
        for src in header_sources:
            src_acc = data["per_source"].get(src, {}).get("accuracy", 0)
            print(f" {src_acc:<18.4f}", end="")
        print()

    # ═══════════════════════════════════════════════════
    # STEP 5: RECOMMENDATION
    # ═══════════════════════════════════════════════════
    print(f"\n\n{'='*70}")
    print("📋 RECOMMENDATION")
    print(f"{'='*70}")

    # Find best threshold for overall OOD accuracy
    best_method = max(
        ood_comparison.items(), key=lambda x: x[1]["overall"]["accuracy"]
    )
    best_name = best_method[0]
    best_data = best_method[1]

    # Find best for real_camera specifically
    best_for_camera = max(
        ood_comparison.items(),
        key=lambda x: x[1]["per_source"].get("real_camera", {}).get("accuracy", 0),
    )
    camera_name = best_for_camera[0]
    camera_data = best_for_camera[1]

    print(f"\n  ⭐ Best OVERALL OOD Accuracy:")
    print(f"     Method:    {best_name}")
    print(f"     Threshold: {best_data['overall']['threshold']:.4f}")
    print(f"     OOD Acc:   {best_data['overall']['accuracy']:.4f}")
    for src in header_sources:
        src_acc = best_data["per_source"].get(src, {}).get("accuracy", 0)
        print(f"     {src}: {src_acc:.4f}")

    print(f"\n  🎯 Best for real_camera:")
    print(f"     Method:    {camera_name}")
    print(f"     Threshold: {camera_data['overall']['threshold']:.4f}")
    camera_acc = camera_data["per_source"].get("real_camera", {}).get("accuracy", 0)
    print(f"     real_camera Acc: {camera_acc:.4f}")
    print(f"     OOD Overall Acc: {camera_data['overall']['accuracy']:.4f}")

    # Check if threshold significantly helps real_camera
    default_camera = ood_comparison["default"]["per_source"].get("real_camera", {}).get("accuracy", 0)
    improvement = camera_acc - default_camera
    print(f"\n  📈 real_camera improvement: {default_camera:.4f} → {camera_acc:.4f} (Δ={improvement:+.4f})")

    if improvement > 0.10:
        print("  ✅ Threshold tuning SIGNIFICANTLY helps real_camera!")
        print(f"  → Khuyến nghị dùng threshold = {camera_data['overall']['threshold']:.4f}")
        print(f"  → Cập nhật configs/test.yaml: threshold: {camera_data['overall']['threshold']:.4f}")
    elif improvement > 0.05:
        print("  ↗️ Threshold tuning helps but marginal. Consider retrain (Strategy 1).")
    else:
        print("  ⚠️ Threshold tuning does NOT significantly help real_camera.")
        print("  → Model P(fake) for real_camera is too high (>0.7)")
        print("  → Need to retrain with more real camera data (Strategy 1)")

    # ═══════════════════════════════════════════════════
    # STEP 6: Save results
    # ═══════════════════════════════════════════════════
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": epoch,
        "thresholds": thresholds,
        "val_comparison": val_comparison,
        "ood_comparison": {
            method: {
                "overall": data["overall"],
                "per_source": data["per_source"],
            }
            for method, data in ood_comparison.items()
        },
        "ood_probability_distribution": ood_analysis,
        "recommendation": {
            "best_overall": {
                "method": best_name,
                "threshold": best_data["overall"]["threshold"],
                "ood_accuracy": best_data["overall"]["accuracy"],
            },
            "best_for_real_camera": {
                "method": camera_name,
                "threshold": camera_data["overall"]["threshold"],
                "real_camera_accuracy": camera_acc,
            },
            "real_camera_improvement": improvement,
            "verdict": (
                "threshold_helps"
                if improvement > 0.10
                else "marginal" if improvement > 0.05 else "need_retrain"
            ),
        },
    }

    report_path = OUTPUT_DIR / "threshold_analysis.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report saved: {report_path}")

    # Plot
    val_analysis = analyze_probability_distribution(val_probs, val_labels, val_sources)
    plot_threshold_analysis(
        val_analysis, ood_analysis, thresholds, ood_comparison, OUTPUT_DIR
    )

    print(f"\n  📁 All outputs saved to: {OUTPUT_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
