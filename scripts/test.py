"""
HolmHz Evaluation Script — đánh giá model trên ID + OOD test set.

Chạy inference trên cả 2 test sets, tính metrics đa chiều,
vẽ confusion matrix + ROC curve, lưu JSON report.

Usage:
    # Default config
    python scripts/test.py

    # Custom checkpoint
    python scripts/test.py model.checkpoint=outputs/checkpoints/best.pt

    # Adjust for local machine (RTX 3050)
    python scripts/test.py data.batch_size=32 data.num_workers=0

Example output:
    outputs/evaluation/
    ├── eval_report.json
    ├── confusion_matrix_id.png
    ├── confusion_matrix_ood.png
    ├── roc_curve.png
    └── per_source_accuracy.png
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv()

import holmhz.detectors  # noqa: E402, F401
from holmhz.data import create_dataloader
from holmhz.evaluation import Evaluator
from holmhz.utils.logger import get_logger
from holmhz.utils.registry import DETECTOR_REGISTRY
from holmhz.utils.visualization import (
    plot_confusion_matrix,
    plot_per_source_accuracy,
    plot_roc_curve,
)

logger = get_logger("test")


def main():
    """Main evaluation entry point."""
    # ─── Load config ───
    config_path = "configs/test.yaml"

    if (
        len(sys.argv) > 1
        and not sys.argv[1].startswith("--")
        and "=" not in sys.argv[1]
        and sys.argv[1].endswith(".yaml")
    ):
        config_path = sys.argv[1]

    config = OmegaConf.load(config_path)

    # CLI overrides
    cli_args = [a for a in sys.argv[1:] if a != config_path]
    if cli_args:
        cli_overrides = OmegaConf.from_cli(cli_args)
        config = OmegaConf.merge(config, cli_overrides)

    logger.info(f"Config: {config_path}")

    # ─── Device ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ─── Load model ───
    checkpoint_path = config.model.checkpoint
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Train first: python scripts/train.py")
        sys.exit(1)

    # Flexible build — skip unsupported kwargs (e.g. FrequencyDetector has no pretrained/freeze_backbone)
    build_kwargs = {"dropout": config.model.get("dropout", 0.3)}
    if OmegaConf.select(config, "model.use_phase", default=None) is not None:
        build_kwargs["use_phase"] = config.model.use_phase
    # Only pass pretrained/freeze_backbone for models that support them
    model_name = config.model.name
    if model_name not in ("freq_fft",):
        build_kwargs["pretrained"] = False
        build_kwargs["freeze_backbone"] = False

    model = DETECTOR_REGISTRY.build(model_name, **build_kwargs)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Strip DataParallel 'module.' prefix if present (saved with nn.DataParallel on multi-GPU)
    if any(k.startswith("module.") for k in state_dict):
        logger.info("Stripping DataParallel 'module.' prefix from checkpoint keys")
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "N/A")
    best_auc = checkpoint.get("best_metric", "N/A")
    logger.info(f"Loaded: {checkpoint_path} (epoch {epoch}, val_auc {best_auc})")

    # ─── Dataloaders ───
    threshold = config.evaluation.get("threshold", 0.5)
    batch_size = config.data.get("batch_size", 32)
    num_workers = config.data.get("num_workers", 0)

    id_loader = create_dataloader(
        manifest_path=config.data.test_manifest,
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

    logger.info(f"ID test:  {len(id_loader.dataset)} samples")
    logger.info(f"OOD test: {len(ood_loader.dataset)} samples")

    # ─── Evaluate ID ───
    print("\n" + "=" * 60)
    print("IN-DOMAIN EVALUATION")
    print("=" * 60)

    id_evaluator = Evaluator(model, id_loader, device, threshold)
    id_results = id_evaluator.evaluate()

    # ─── Evaluate OOD ───
    print("\n" + "=" * 60)
    print("OOD EVALUATION")
    print("=" * 60)

    ood_evaluator = Evaluator(model, ood_loader, device, threshold)
    ood_results = ood_evaluator.evaluate()

    # ─── Output dir ───
    output_dir = Path(config.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Visualization ───
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Confusion matrices
    plot_confusion_matrix(
        id_results["all_labels"],
        id_results["all_logits"],
        save_path=str(output_dir / "confusion_matrix_id.png"),
        title="Confusion Matrix - In-Domain Test",
        threshold=threshold,
    )

    plot_confusion_matrix(
        ood_results["all_labels"],
        ood_results["all_logits"],
        save_path=str(output_dir / "confusion_matrix_ood.png"),
        title="Confusion Matrix - OOD Test",
        threshold=threshold,
    )

    # ROC curve (both ID and OOD on same plot)
    plot_roc_curve(
        {
            "In-Domain": id_results,
            "Out-of-Distribution": ood_results,
        },
        save_path=str(output_dir / "roc_curve.png"),
        title="ROC Curve - In-Domain vs OOD",
    )

    # Per-source accuracy (combine ID + OOD)
    all_per_source = {}
    for src, metrics in id_results["per_source"].items():
        all_per_source[f"ID: {src}"] = metrics
    for src, metrics in ood_results["per_source"].items():
        all_per_source[f"OOD: {src}"] = metrics

    plot_per_source_accuracy(
        all_per_source,
        save_path=str(output_dir / "per_source_accuracy.png"),
        title="Per-Source Accuracy - All Test Sets",
    )

    # ─── JSON Report ───
    report = {
        "model": config.model.name,
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": epoch,
        "checkpoint_val_auc": float(best_auc) if isinstance(best_auc, (int, float)) else best_auc,
        "threshold": threshold,
        "timestamp": datetime.now().isoformat(),
        "in_domain": {
            "manifest": config.data.test_manifest,
            "total": id_results["total"],
            "overall": id_results["overall"],
            "per_source": {
                src: {k: v for k, v in metrics.items()}
                for src, metrics in id_results["per_source"].items()
            },
        },
        "ood": {
            "manifest": config.data.ood_manifest,
            "total": ood_results["total"],
            "overall": ood_results["overall"],
            "per_source": {
                src: {k: v for k, v in metrics.items()}
                for src, metrics in ood_results["per_source"].items()
            },
        },
    }

    report_path = output_dir / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Report saved: {report_path}")

    # ─── Summary table ───
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nModel: {config.model.name}")
    print(f"Checkpoint: {checkpoint_path} (epoch {epoch})")
    print(f"Threshold: {threshold}")

    print(f"\n{'Set':<25} {'AUC':<8} {'Acc':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'N':<6}")
    print("-" * 70)

    for name, results in [("In-Domain", id_results), ("OOD", ood_results)]:
        o = results["overall"]
        print(
            f"{name:<25} "
            f"{o['auc']:<8.4f} "
            f"{o['accuracy']:<8.4f} "
            f"{o['f1']:<8.4f} "
            f"{o['precision']:<8.4f} "
            f"{o['recall']:<8.4f} "
            f"{results['total']:<6}"
        )

    print(f"\n{'Source':<25} {'Acc':<8} {'AUC':<8} {'N':<6} {'Real':<6} {'Fake':<6}")
    print("-" * 70)

    for src, metrics in sorted(id_results["per_source"].items()):
        print(
            f"  ID/{src:<22} "
            f"{metrics['accuracy']:<8.4f} "
            f"{metrics['auc']:<8.4f} "
            f"{metrics['n']:<6} "
            f"{metrics['n_real']:<6} "
            f"{metrics['n_fake']:<6}"
        )

    for src, metrics in sorted(ood_results["per_source"].items()):
        print(
            f"  OOD/{src:<21} "
            f"{metrics['accuracy']:<8.4f} "
            f"{metrics['auc']:<8.4f} "
            f"{metrics['n']:<6} "
            f"{metrics['n_real']:<6} "
            f"{metrics['n_fake']:<6}"
        )

    print("=" * 70)

    # ─── OOD Analysis ───
    print("\nOOD FAILURE ANALYSIS")
    print("-" * 50)

    ood_overall = ood_results["overall"]
    id_overall = id_results["overall"]
    gap = id_overall["auc"] - ood_overall["auc"]

    print(f"ID AUC:  {id_overall['auc']:.4f}")
    print(f"OOD AUC: {ood_overall['auc']:.4f}")
    print(f"Gap:     {gap:.4f} ({'WARNING: LARGE GAP' if gap > 0.15 else 'acceptable'})")

    # Tìm source yếu nhất
    worst_source = min(
        ood_results["per_source"].items(),
        key=lambda x: x[1]["accuracy"],
    )
    print(f"\nWeakest OOD source: {worst_source[0]} "
          f"(Acc: {worst_source[1]['accuracy']:.4f}, N: {worst_source[1]['n']})")

    # False Positive analysis (Real → FAKE)
    ood_precision = ood_overall["precision"]
    ood_recall = ood_overall["recall"]
    if ood_precision < ood_recall:
        print("\nWARNING: False Positive dominant: Model bias predicts FAKE")
        print("   -> Many Real OOD images misclassified as FAKE")
        print("   -> Consistent with smoke test Task 1.6 (5/5 Real -> FAKE)")
    else:
        print("\nWARNING: False Negative dominant: Model misses many Fakes")
        print("   -> Many Fake OOD images misclassified as Real")

    print(f"\nAll outputs saved to: {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()